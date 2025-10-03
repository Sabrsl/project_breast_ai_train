"""
Script d'entraînement pour filtre d'images breast cancer (3 classes)
Architecture: EfficientNetV2-S avec PyTorch
Classes: non_medical (0) / medical_other (1) / breast (2)
FILTRE: Seule la classe 'breast' passe au modèle principal
Les autres classes améliorent la précision et réduisent les faux positifs
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix, classification_report, f1_score
import numpy as np
from pathlib import Path
import time
from collections import defaultdict

# ======================== CONFIGURATION ========================
class Config:
    # Chemins
    DATA_DIR = r"C:\Users\badza\Desktop\project_breast_ai\filtre_ai\data"
    CHECKPOINT_DIR = r"C:\Users\badza\Desktop\project_breast_ai\filtre_ai\checkpoints"
    
    # Hyperparamètres
    IMG_SIZE = 224
    BATCH_SIZE = 8  # Optimisé pour CPU 8GB RAM
    NUM_EPOCHS = 50
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    GRADIENT_ACCUMULATION_STEPS = 2  # Simuler batch_size=16
    
    # Early stopping
    PATIENCE = 7
    MIN_DELTA = 0.0005
    
    # Checkpointing
    SAVE_EVERY = 5
    
    # Backbone freezing
    FREEZE_EPOCHS = 5
    
    # ⚠️ CONFIGURATION 3 CLASSES (seul breast passe le filtre)
    CLASSES = ['non_medical', 'medical_other', 'breast']
    NUM_CLASSES = 3
    TARGET_CLASS_IDX = 2  # Index de la classe "breast" (SEULE CLASSE QUI PASSE)
    TARGET_CLASS_NAME = 'breast'
    
    # ⚠️ MÉTRIQUES CRITIQUES pour le filtrage
    # Le but: maximiser le recall sur breast (ne pas rater les vrais breast)
    # tout en gardant une bonne précision (éviter faux positifs)
    MIN_BREAST_RECALL = 0.95     # Minimum 95% de recall sur breast
    TARGET_BREAST_PRECISION = 0.85  # Objectif 85% de précision sur breast
    MIN_BREAST_F1 = 0.90           # Objectif F1-score pour équilibre
    
    # Device
    DEVICE = torch.device('cpu')
    NUM_WORKERS = 0

# ======================== FOCAL LOSS ========================
class FocalLoss(nn.Module):
    """
    Focal Loss pour gérer le déséquilibre de classes
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ======================== TRANSFORMATIONS ========================
def get_transforms(phase='train'):
    """
    Transformations d'images
    """
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

# ======================== DATASET ET DATALOADER ========================
def create_dataloaders():
    """
    Crée les DataLoaders pour l'entraînement et la validation
    Structure attendue:
        data/
            train/
                non_medical/     (images non médicales)
                medical_other/   (images médicales mais pas breast)
                breast/          (images breast - SEULE CLASSE QUI PASSE)
            val/
                non_medical/
                medical_other/
                breast/
    """
    train_dir = os.path.join(Config.DATA_DIR, 'train')
    val_dir = os.path.join(Config.DATA_DIR, 'val')
    
    # Vérification de l'existence des dossiers
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("⚠️  Dossiers train/val non trouvés. Utilisation du dataset complet.")
        full_dataset = datasets.ImageFolder(Config.DATA_DIR, transform=get_transforms('train'))
        
        # Split manuel 80/20
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        val_dataset.dataset.transform = get_transforms('val')
    else:
        train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms('train'))
        val_dataset = datasets.ImageFolder(val_dir, transform=get_transforms('val'))
    
    # Calcul des poids pour le sampler (gérer le déséquilibre)
    if hasattr(train_dataset, 'targets'):
        targets = train_dataset.targets
    else:
        targets = [train_dataset.dataset.targets[i] for i in train_dataset.indices]
    
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    
    # ⚠️ BOOST pour la classe breast (focus sur ne pas la rater)
    class_weights[Config.TARGET_CLASS_IDX] *= 1.5  # Augmenter l'importance de breast
    
    sample_weights = [class_weights[t] for t in targets]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False
    )
    
    # Calcul des poids pour la loss
    loss_weights = torch.tensor(class_weights, dtype=torch.float32).to(Config.DEVICE)
    
    print(f"✓ Dataset chargé: {len(train_dataset)} train, {len(val_dataset)} val")
    print(f"✓ Distribution des classes: {dict(zip(Config.CLASSES, class_counts))}")
    print(f"✓ Poids de la loss: {dict(zip(Config.CLASSES, class_weights))}")
    print(f"\n⚠️  RAPPEL: Seule la classe '{Config.TARGET_CLASS_NAME}' passera le filtre!\n")
    
    return train_loader, val_loader, loss_weights

# ======================== MODÈLE ========================
def create_model():
    """
    Crée le modèle EfficientNetV2-S pré-entraîné
    """
    model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
    
    # Modifier la dernière couche pour 3 classes
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, Config.NUM_CLASSES)
    
    model = model.to(Config.DEVICE)
    
    print(f"✓ Modèle EfficientNetV2-S chargé sur {Config.DEVICE}")
    
    return model

def freeze_backbone(model, freeze=True):
    """
    Gèle ou dégèle le backbone du modèle
    """
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = not freeze
    
    status = "gelé" if freeze else "dégelé"
    print(f"✓ Backbone {status}")

# ======================== MÉTRIQUES DÉTAILLÉES ========================
def compute_breast_metrics(all_labels, all_preds):
    """
    Calcule les métriques détaillées pour la classe breast
    """
    # Conversion en binaire: breast (1) vs non-breast (0)
    binary_labels = (np.array(all_labels) == Config.TARGET_CLASS_IDX).astype(int)
    binary_preds = (np.array(all_preds) == Config.TARGET_CLASS_IDX).astype(int)
    
    # Métriques sur classe breast
    breast_recall = recall_score(binary_labels, binary_preds, zero_division=0)
    breast_precision = precision_score(binary_labels, binary_preds, zero_division=0)
    breast_f1 = f1_score(binary_labels, binary_preds, zero_division=0)
    
    # Métriques multi-classes
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Recall par classe
    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'breast_recall': breast_recall,
        'breast_precision': breast_precision,
        'breast_f1': breast_f1,
        'per_class_recall': per_class_recall
    }

# ======================== ENTRAÎNEMENT ========================
def train_epoch(model, dataloader, criterion, optimizer, epoch):
    """
    Entraîne le modèle pour une epoch avec gradient accumulation
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Normaliser la loss pour gradient accumulation
        loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
        
        # Backward
        loss.backward()
        
        # Update weights après accumulation
        if (batch_idx + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # Métriques
        running_loss += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Affichage progressif
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(dataloader):
            progress = (batch_idx + 1) / len(dataloader) * 100
            print(f"\r  [{progress:5.1f}%] Batch {batch_idx + 1}/{len(dataloader)} - Loss: {loss.item() * Config.GRADIENT_ACCUMULATION_STEPS:.4f}", end='')
    
    # Calcul des métriques finales
    epoch_loss = running_loss / len(dataloader)
    metrics = compute_breast_metrics(all_labels, all_preds)
    
    print()  # Nouvelle ligne après la barre de progression
    
    return epoch_loss, metrics

def validate(model, dataloader, criterion):
    """
    Évalue le modèle sur le dataset de validation avec rapport détaillé
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(dataloader)
    metrics = compute_breast_metrics(all_labels, all_preds)
    
    # Matrice de confusion pour analyse détaillée
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return val_loss, metrics, conf_matrix

# ======================== EARLY STOPPING ========================
class EarlyStopping:
    """
    Early stopping basé sur le F1-score de la classe breast
    (équilibre entre recall et precision)
    """
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0

# ======================== CHECKPOINT ========================
def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath):
    """
    Sauvegarde un checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'config': {
            'num_classes': Config.NUM_CLASSES,
            'classes': Config.CLASSES,
            'target_class_idx': Config.TARGET_CLASS_IDX,
            'img_size': Config.IMG_SIZE
        }
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """
    Charge un checkpoint
    """
    checkpoint = torch.load(filepath, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['metrics']

# ======================== AFFICHAGE DES RÉSULTATS ========================
def print_detailed_results(metrics, conf_matrix, phase='Val'):
    """
    Affiche les résultats détaillés avec focus sur la classe breast
    """
    print(f"  {phase} → Loss: {metrics.get('loss', 0):.4f} | Acc: {metrics['accuracy']:.3f}")
    print(f"  🎯 CLASSE BREAST (seule classe qui passe):")
    print(f"     • Recall (sensibilité):  {metrics['breast_recall']:.3f} {'✓' if metrics['breast_recall'] >= Config.MIN_BREAST_RECALL else '⚠️'}")
    print(f"     • Precision:             {metrics['breast_precision']:.3f} {'✓' if metrics['breast_precision'] >= Config.TARGET_BREAST_PRECISION else '⚠️'}")
    print(f"     • F1-Score:              {metrics['breast_f1']:.3f} {'✓' if metrics['breast_f1'] >= Config.MIN_BREAST_F1 else '⚠️'}")
    
    # Recall par classe
    print(f"  📊 Recall par classe:")
    for i, class_name in enumerate(Config.CLASSES):
        marker = "🎯" if i == Config.TARGET_CLASS_IDX else "  "
        print(f"     {marker} {class_name:15s}: {metrics['per_class_recall'][i]:.3f}")
    
    # Matrice de confusion
    if conf_matrix is not None:
        print(f"  📋 Matrice de confusion:")
        print(f"     Prédictions →")
        header = "     Réel ↓         " + "  ".join([f"{c[:10]:>10s}" for c in Config.CLASSES])
        print(header)
        for i, row in enumerate(conf_matrix):
            row_str = "     " + f"{Config.CLASSES[i][:15]:15s}: " + "  ".join([f"{val:>10d}" for val in row])
            print(row_str)

# ======================== MAIN ========================
def main():
    """
    Fonction principale d'entraînement
    """
    print("=" * 80)
    print("ENTRAÎNEMENT FILTRE BREAST CANCER (3 classes)")
    print("=" * 80)
    print(f"⚠️  RAPPEL: Seule la classe '{Config.TARGET_CLASS_NAME}' passera au modèle principal")
    print(f"   Les classes {Config.CLASSES[:2]} servent à améliorer la précision du filtre")
    print("=" * 80)
    
    # Créer le dossier de checkpoints
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    # Charger les données
    train_loader, val_loader, class_weights = create_dataloaders()
    
    # Créer le modèle
    model = create_model()
    
    # Loss avec pondération
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    # Optimiseur
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, 
                           weight_decay=Config.WEIGHT_DECAY)
    
    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    
    # Early stopping basé sur F1-score breast
    early_stopping = EarlyStopping(patience=Config.PATIENCE, min_delta=Config.MIN_DELTA)
    
    # Historique
    history = defaultdict(list)
    best_f1 = 0.0
    start_epoch = 0
    
    # Reprendre l'entraînement si checkpoint existe
    latest_checkpoint = os.path.join(Config.CHECKPOINT_DIR, 'latest.pth')
    if os.path.exists(latest_checkpoint):
        print(f"\n📂 Chargement du checkpoint: {latest_checkpoint}")
        start_epoch, saved_metrics = load_checkpoint(latest_checkpoint, model, optimizer, scheduler)
        best_f1 = saved_metrics.get('best_breast_f1', 0.0)
        print(f"✓ Reprise à l'epoch {start_epoch + 1}")
    
    # Boucle d'entraînement
    print(f"\n🚀 Début de l'entraînement pour {Config.NUM_EPOCHS} epochs\n")
    
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        epoch_start = time.time()
        
        # Geler/dégeler le backbone
        if epoch == 0:
            freeze_backbone(model, freeze=True)
        elif epoch == Config.FREEZE_EPOCHS:
            freeze_backbone(model, freeze=False)
        
        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        print("-" * 80)
        
        # Entraînement
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, epoch
        )
        train_metrics['loss'] = train_loss
        
        # Validation
        val_loss, val_metrics, conf_matrix = validate(model, val_loader, criterion)
        val_metrics['loss'] = val_loss
        
        # Scheduler step
        scheduler.step()
        
        # Temps
        epoch_time = time.time() - epoch_start
        
        # Affichage détaillé
        print_detailed_results(train_metrics, None, phase='Train')
        print_detailed_results(val_metrics, conf_matrix, phase='Val')
        print(f"  ⏱️  Temps: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.2e}\n")
        
        # Historique
        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                history[f'train_{key}'].append(value)
        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                history[f'val_{key}'].append(value)
        
        # Sauvegarder le meilleur modèle basé sur F1-score breast
        current_f1 = val_metrics['breast_f1']
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, 
                          {'best_breast_f1': best_f1, 
                           'breast_recall': val_metrics['breast_recall'],
                           'breast_precision': val_metrics['breast_precision']}, 
                          best_path)
            print(f"  💾 Meilleur modèle sauvegardé (F1: {best_f1:.3f})")
        
        # Sauvegarder checkpoint régulier
        if (epoch + 1) % Config.SAVE_EVERY == 0:
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch,
                          {'breast_f1': current_f1}, checkpoint_path)
        
        # Sauvegarder le dernier checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch,
                      {'best_breast_f1': best_f1}, latest_checkpoint)
        
        # Early stopping basé sur F1-score
        early_stopping(current_f1)
        if early_stopping.early_stop:
            print(f"\n⏹️  Early stopping déclenché à l'epoch {epoch + 1}")
            break
    
    # Sauvegarder l'historique
    history_path = os.path.join(Config.CHECKPOINT_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        # Convertir les arrays numpy en listes
        history_serializable = {}
        for key, value in history.items():
            if isinstance(value, list):
                history_serializable[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in value]
            else:
                history_serializable[key] = value
        json.dump(history_serializable, f, indent=4)
    
    print("\n" + "=" * 80)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print(f"🎯 Meilleur F1-score (breast): {best_f1:.3f}")
    print(f"⚠️  RAPPEL: Ce modèle filtre les images - seule '{Config.TARGET_CLASS_NAME}' passe!")
    print(f"💾 Modèles sauvegardés dans: {Config.CHECKPOINT_DIR}")
    print("=" * 80)

if __name__ == '__main__':
    main()