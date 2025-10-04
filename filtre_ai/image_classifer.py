"""
Script d'entraînement PRODUCTION pour filtre breast cancer (BINAIRE)
Architecture: EfficientNetV2-S avec PyTorch
Objectif: RECALL maximal (ne jamais rater une vraie image breast)

CORRECTIONS APPLIQUÉES:
- Focal Loss avec alpha inversé (pondère classe minoritaire)
- Calibration de seuil post-entraînement
- Train/Val/Test/Calib stratifiés
- Métriques de production (ROC, PR curves)
- Discriminative learning rates
- Seed reproducibility
- Guards sécurité (bincount, confusion_matrix, best_model)
"""

import os
import json
import time
import warnings
import logging
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import (
    recall_score, precision_score, accuracy_score, confusion_matrix,
    f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# ======================== CONFIGURATION ========================
class Config:
    # Chemins
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
    LOG_DIR = BASE_DIR / 'logs'
    PLOTS_DIR = BASE_DIR / 'plots'
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = min(4, os.cpu_count() or 4)
    
    # Reproducibility
    SEED = 42
    
    # Hyperparamètres
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    BACKBONE_LR = 1e-5
    CLASSIFIER_LR = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10
    MIN_DELTA = 0.001
    SAVE_EVERY = 5
    UNFREEZE_EPOCH = 5
    
    # Classes
    CLASSES = ['not_breast', 'breast']
    NUM_CLASSES = 2
    TARGET_CLASS_IDX = 1
    TARGET_CLASS_NAME = 'breast'
    
    # Métriques cibles
    MIN_BREAST_RECALL = 0.98
    TARGET_BREAST_PRECISION = 0.85
    MIN_BREAST_F1 = 0.90
    
    # Focal Loss (alpha calculé dynamiquement - INVERSÉ pour pondérer minoritaire)
    FOCAL_ALPHA = None
    FOCAL_GAMMA = 2.0
    
    # Splits
    VAL_RATIO = 0.12
    CALIB_RATIO = 0.08
    TEST_RATIO = 0.15
    
    # Early stopping
    EARLY_STOP_METRIC = 'f1'
    IGNORE_CORRUPTED = False
    
    # Sampler (désactivé par défaut - FocalLoss suffit)
    USE_WEIGHTED_SAMPLER = False  # Mettre True si besoin, mais surveiller FP


logger = None


# ======================== REPRODUCIBILITY ========================
def set_seed(seed=42):
    """Fixe les seeds pour reproductibilité"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================== LOGGING ========================
def setup_logging():
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = Config.LOG_DIR / f'training_{time.strftime("%Y%m%d_%H%M%S")}.log'
    
    log = logging.getLogger('BreastFilter')
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    
    log.addHandler(fh)
    log.addHandler(ch)
    log.info(f"Logging: {log_file}")
    
    return log


# ======================== FOCAL LOSS ========================
class FocalLoss(nn.Module):
    """
    Focal Loss - alpha INVERSÉ pour pondérer la classe minoritaire
    alpha > 0.5 = plus de poids sur classe positive (breast)
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        p = torch.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.view(-1, 1)).squeeze()
        modulating_factor = (1 - p_t) ** self.gamma
        
        # Alpha balancing - INVERSÉ: alpha pour classe 1 (breast)
        alpha_t = torch.full_like(targets, 1 - self.alpha, dtype=torch.float, device=inputs.device)
        alpha_t[targets == 1] = self.alpha
        
        focal_loss = alpha_t * modulating_factor * ce_loss
        return focal_loss.mean()


# ======================== DATASET VALIDATION ========================
def validate_dataset():
    logger.info("=" * 80)
    logger.info("VALIDATION DU DATASET")
    logger.info("=" * 80)
    
    corrupted_files = []
    file_stats = defaultdict(lambda: {'count': 0, 'formats': defaultdict(int)})
    
    for split in ['train', 'val']:
        split_dir = Config.DATA_DIR / split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Dossier {split} introuvable: {split_dir}")
        
        logger.info(f"\n{split}/")
        
        for cls in Config.CLASSES:
            cls_dir = split_dir / cls
            
            if not cls_dir.exists():
                raise FileNotFoundError(f"Dossier {split}/{cls} introuvable")
            
            images = [f for f in cls_dir.glob('*') if not f.name.startswith('.') and f.is_file()]
            
            if not images:
                logger.warning(f"Aucune image dans {split}/{cls}")
            
            for img_path in images:
                try:
                    img = Image.open(img_path)
                    img.verify()
                    file_stats[f"{split}/{cls}"]['count'] += 1
                    file_stats[f"{split}/{cls}"]['formats'][img.format] += 1
                except Exception as e:
                    corrupted_files.append({
                        'path': str(img_path),
                        'split': split,
                        'class': cls,
                        'error': str(e)
                    })
    
    logger.info("\nStatistiques:")
    logger.info("-" * 80)
    for key, stats in sorted(file_stats.items()):
        count = stats['count']
        formats = ', '.join([f"{fmt}: {cnt}" for fmt, cnt in stats['formats'].items()])
        logger.info(f"   {key:20s}: {count:5d} images ({formats})")
    
    if corrupted_files:
        logger.warning(f"\n{len(corrupted_files)} image(s) corrompue(s)")
        corrupted_log = Config.LOG_DIR / 'corrupted_images.json'
        with open(corrupted_log, 'w') as f:
            json.dump(corrupted_files, f, indent=4)
        logger.warning(f"   Liste: {corrupted_log}")
        
        if not Config.IGNORE_CORRUPTED:
            raise RuntimeError(f"{len(corrupted_files)} images corrompues. Utilisez --ignore-corrupted.")
    else:
        logger.info("\nToutes les images sont valides")
    
    logger.info("=" * 80)
    return file_stats


# ======================== TRANSFORMATIONS ========================
def get_transforms(phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# ======================== DATALOADERS STRATIFIÉS ========================
def create_stratified_dataloaders():
    train_dir = Config.DATA_DIR / 'train'
    full_dataset = datasets.ImageFolder(str(train_dir), transform=None)
    
    targets = np.array(full_dataset.targets)
    indices = np.arange(len(targets))
    
    # Guard: vérifier au moins 2 classes
    class_counts = np.bincount(targets, minlength=2)
    if len(class_counts) < 2:
        raise ValueError("Dataset doit contenir au moins 2 classes")
    
    n_not_breast, n_breast = class_counts[0], class_counts[1]
    
    if n_breast == 0:
        raise ValueError("Aucune image 'breast' trouvée dans le dataset")
    
    # Alpha INVERSÉ: plus de poids sur classe minoritaire (breast)
    # Si breast = minoritaire, alpha doit être > 0.5
    Config.FOCAL_ALPHA = n_not_breast / (n_not_breast + n_breast)
    
    logger.info(f"\nDistribution des classes:")
    logger.info(f"   not_breast: {n_not_breast:5d} ({n_not_breast/len(targets)*100:.1f}%)")
    logger.info(f"   breast:     {n_breast:5d} ({n_breast/len(targets)*100:.1f}%)")
    logger.info(f"   Ratio: {n_not_breast/n_breast:.2f}:1")
    logger.info(f"   Alpha Focal (INVERSÉ): {Config.FOCAL_ALPHA:.4f} (pondère classe breast)")
    
    # Splits stratifiés
    total_other = Config.VAL_RATIO + Config.CALIB_RATIO + Config.TEST_RATIO
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=total_other, random_state=Config.SEED)
    train_idx, temp_idx = next(sss1.split(indices, targets))
    
    temp_targets = targets[temp_idx]
    test_ratio_adjusted = Config.TEST_RATIO / total_other
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio_adjusted, random_state=Config.SEED)
    val_calib_idx_temp, test_idx_temp = next(sss2.split(temp_idx, temp_targets))
    
    val_calib_idx = temp_idx[val_calib_idx_temp]
    test_idx = temp_idx[test_idx_temp]
    val_calib_targets = targets[val_calib_idx]
    calib_ratio_adjusted = Config.CALIB_RATIO / (Config.VAL_RATIO + Config.CALIB_RATIO)
    sss3 = StratifiedShuffleSplit(n_splits=1, test_size=calib_ratio_adjusted, random_state=Config.SEED)
    val_idx_temp, calib_idx_temp = next(sss3.split(val_calib_idx, val_calib_targets))
    val_idx = val_calib_idx[val_idx_temp]
    calib_idx = val_calib_idx[calib_idx_temp]
    
    # Créer datasets avec transformations
    train_dataset = datasets.ImageFolder(str(train_dir), transform=get_transforms('train'))
    val_dataset = datasets.ImageFolder(str(train_dir), transform=get_transforms('val'))
    calib_dataset = datasets.ImageFolder(str(train_dir), transform=get_transforms('val'))
    test_dataset = datasets.ImageFolder(str(train_dir), transform=get_transforms('val'))
    
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    calib_subset = Subset(calib_dataset, calib_idx)
    test_subset = Subset(test_dataset, test_idx)
    
    # DataLoaders
    pin_memory = Config.DEVICE.type == 'cuda'
    
    # Option: WeightedRandomSampler (désactivé par défaut car FocalLoss suffit)
    if Config.USE_WEIGHTED_SAMPLER:
        train_targets = targets[train_idx]
        sample_weights = np.array([1.0 / class_counts[t] for t in train_targets])
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(train_subset, batch_size=Config.BATCH_SIZE,
                                 sampler=sampler, num_workers=Config.NUM_WORKERS,
                                 pin_memory=pin_memory)
        logger.info(f"   WeightedRandomSampler activé (attention aux FP)")
    else:
        train_loader = DataLoader(train_subset, batch_size=Config.BATCH_SIZE,
                                 shuffle=True, num_workers=Config.NUM_WORKERS,
                                 pin_memory=pin_memory)
        logger.info(f"   Shuffle standard (FocalLoss gère déséquilibre)")
    
    val_loader = DataLoader(val_subset, batch_size=Config.BATCH_SIZE,
                           shuffle=False, num_workers=Config.NUM_WORKERS,
                           pin_memory=pin_memory)
    calib_loader = DataLoader(calib_subset, batch_size=Config.BATCH_SIZE,
                             shuffle=False, num_workers=Config.NUM_WORKERS,
                             pin_memory=pin_memory)
    test_loader = DataLoader(test_subset, batch_size=Config.BATCH_SIZE,
                            shuffle=False, num_workers=Config.NUM_WORKERS,
                            pin_memory=pin_memory)
    
    # Stats
    val_targets = targets[val_idx]
    calib_targets = targets[calib_idx]
    test_targets = targets[test_idx]
    
    train_counts = np.bincount(targets[train_idx], minlength=2)
    val_counts = np.bincount(val_targets, minlength=2)
    calib_counts = np.bincount(calib_targets, minlength=2)
    test_counts = np.bincount(test_targets, minlength=2)
    
    logger.info(f"\nSplits stratifiés:")
    logger.info(f"   Train: {len(train_idx):5d} - breast={train_counts[1]:4d} ({train_counts[1]/len(train_idx)*100:.1f}%)")
    logger.info(f"   Val:   {len(val_idx):5d} - breast={val_counts[1]:4d} ({val_counts[1]/len(val_idx)*100:.1f}%)")
    logger.info(f"   Calib: {len(calib_idx):5d} - breast={calib_counts[1]:4d} ({calib_counts[1]/len(calib_idx)*100:.1f}%)")
    logger.info(f"   Test:  {len(test_idx):5d} - breast={test_counts[1]:4d} ({test_counts[1]/len(test_idx)*100:.1f}%)")
    
    return train_loader, val_loader, calib_loader, test_loader


# ======================== MODÈLE ========================
def create_model():
    model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, Config.NUM_CLASSES)
    model = model.to(Config.DEVICE)
    logger.info(f"EfficientNetV2-S sur {Config.DEVICE}")
    return model


def get_param_groups(model):
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    
    return [
        {'params': backbone_params, 'lr': Config.BACKBONE_LR},
        {'params': classifier_params, 'lr': Config.CLASSIFIER_LR}
    ]


def unfreeze_backbone_progressive(model, epoch):
    if epoch < Config.UNFREEZE_EPOCH:
        for name, param in model.named_parameters():
            param.requires_grad = 'classifier' in name
        return "gelé"
    else:
        for param in model.parameters():
            param.requires_grad = True
        return "dégelé"


# ======================== MÉTRIQUES ========================
def compute_metrics(labels, preds, probs=None):
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'breast_recall': recall_score(labels, preds, pos_label=1, zero_division=0),
        'breast_precision': precision_score(labels, preds, pos_label=1, zero_division=0),
        'breast_f1': f1_score(labels, preds, pos_label=1, zero_division=0),
    }
    
    # Guard: vérifier shape confusion_matrix
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    if cm.shape != (2, 2):
        logger.warning(f"Confusion matrix shape inattendue: {cm.shape}")
        # Fallback sécuritaire
        metrics.update({'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'specificity': 0})
    else:
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
        })
    
    if probs is not None:
        fpr, tpr, _ = roc_curve(labels, probs)
        metrics['auc_roc'] = auc(fpr, tpr)
        metrics['auc_pr'] = average_precision_score(labels, probs)
    
    return metrics, cm


# ======================== ENTRAÎNEMENT ========================
def train_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if (batch_idx + 1) % 50 == 0:
            progress = (batch_idx + 1) / len(loader) * 100
            avg_loss = running_loss / (batch_idx + 1)
            print(f"\r  [{progress:5.1f}%] Batch {batch_idx+1}/{len(loader)} - Loss: {avg_loss:.4f}", end='')
    
    print()
    epoch_loss = running_loss / len(loader)
    metrics, _ = compute_metrics(all_labels, all_preds)
    return epoch_loss, metrics


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    val_loss = running_loss / len(loader)
    metrics, cm = compute_metrics(all_labels, all_preds, all_probs)
    return val_loss, metrics, cm, all_probs, all_labels


# ======================== CALIBRATION ========================
def calibrate_threshold(probs, labels, target_recall=0.98):
    precision_curve, recall_curve, thresholds = precision_recall_curve(labels, probs)
    valid_indices = np.where(recall_curve >= target_recall)[0]
    
    if len(valid_indices) == 0:
        logger.warning(f"Impossible d'atteindre recall={target_recall}")
        return 0.5, 0.0, 0.0
    
    best_idx = valid_indices[np.argmax(precision_curve[valid_indices])]
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    
    calibrated_preds = (np.array(probs) >= optimal_threshold).astype(int)
    precision = precision_score(labels, calibrated_preds, pos_label=1)
    recall = recall_score(labels, calibrated_preds, pos_label=1)
    
    logger.info(f"\nCalibration de seuil:")
    logger.info(f"   Seuil optimal: {optimal_threshold:.4f}")
    logger.info(f"   Recall: {recall:.4f}")
    logger.info(f"   Precision: {precision:.4f}")
    
    return optimal_threshold, precision, recall


# ======================== VISUALISATIONS ========================
def plot_metrics(history, save_path):
    Config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['train_breast_recall'], label='Train')
    axes[0, 1].plot(history['val_breast_recall'], label='Val')
    axes[0, 1].axhline(y=Config.MIN_BREAST_RECALL, color='r', linestyle='--', label='Target')
    axes[0, 1].set_title('Breast Recall')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[0, 2].plot(history['train_breast_precision'], label='Train')
    axes[0, 2].plot(history['val_breast_precision'], label='Val')
    axes[0, 2].set_title('Breast Precision')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    axes[1, 0].plot(history['train_breast_f1'], label='Train')
    axes[1, 0].plot(history['val_breast_f1'], label='Val')
    axes[1, 0].set_title('Breast F1-Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    if 'train_specificity' in history:
        axes[1, 1].plot(history['train_specificity'], label='Train')
        axes[1, 1].plot(history['val_specificity'], label='Val')
        axes[1, 1].set_title('Specificity')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    if 'learning_rate' in history:
        axes[1, 2].plot(history['learning_rate'], label='LR')
        axes[1, 2].set_title('Learning Rate')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_yscale('log')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"   Graphique: {save_path}")


def plot_roc_pr_curves(probs, labels, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(True)
    
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    axes[1].plot(recall, precision, label=f'AP = {pr_auc:.3f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"   ROC/PR: {save_path}")


# ======================== EARLY STOPPING ========================
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001, mode='max', metric='f1'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.metric = metric
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, metrics):
        if self.metric == 'f1':
            score = metrics['breast_f1']
        elif self.metric == 'recall':
            score = metrics['breast_recall']
        elif self.metric == 'balanced_accuracy':
            score = (metrics['breast_recall'] + metrics['specificity']) / 2
        else:
            score = metrics['breast_f1']
        
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
        
        return score


# ======================== CHECKPOINTING ========================
def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath, threshold=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'optimal_threshold': threshold,
        'config': {'num_classes': Config.NUM_CLASSES, 'img_size': Config.IMG_SIZE}
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(filepath, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['metrics'], checkpoint.get('optimal_threshold')


# ======================== AFFICHAGE ========================
def print_results(metrics, cm, phase='Val'):
    logger.info(f"\n  {phase} -> Loss: {metrics.get('loss', 0):.4f} | Acc: {metrics['accuracy']:.3f}")
    
    if 'auc_roc' in metrics:
        logger.info(f"   AUC-ROC: {metrics['auc_roc']:.3f} | AUC-PR: {metrics['auc_pr']:.3f}")
    
    status = 'OK' if metrics['breast_recall'] >= Config.MIN_BREAST_RECALL else 'NON'
    logger.info(f"  Classe BREAST:")
    logger.info(f"     Recall:      {metrics['breast_recall']:.4f} [{status}]")
    logger.info(f"     Precision:   {metrics['breast_precision']:.4f}")
    logger.info(f"     F1-Score:    {metrics['breast_f1']:.4f}")
    logger.info(f"     Specificity: {metrics['specificity']:.4f}")
    
    logger.info(f"  Confusion:")
    logger.info(f"     TP={metrics['tp']:4d} | FP={metrics['fp']:4d}")
    logger.info(f"     FN={metrics['fn']:4d} | TN={metrics['tn']:4d}")
    
    if metrics['tp'] + metrics['fn'] > 0:
        miss_rate = metrics['fn'] / (metrics['tp'] + metrics['fn']) * 100
        logger.info(f"     Taux de perte: {miss_rate:.2f}%")


# ======================== MAIN ========================
def main():
    global logger
    logger = setup_logging()
    
    # Reproducibility
    set_seed(Config.SEED)
    logger.info(f"Seed: {Config.SEED}")
    
    logger.info("=" * 80)
    logger.info("FILTRE BREAST CANCER - PRODUCTION")
    logger.info("=" * 80)
    logger.info(f"Device: {Config.DEVICE}")
    logger.info(f"Batch: {Config.BATCH_SIZE} | Epochs: {Config.NUM_EPOCHS}")
    logger.info(f"LR: Backbone={Config.BACKBONE_LR:.2e} | Classifier={Config.CLASSIFIER_LR:.2e}")
    logger.info("=" * 80)
    
    try:
        validate_dataset()
    except Exception as e:
        logger.error(f"Validation echouee: {e}")
        return
    
    Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Dataloaders
    train_loader, val_loader, calib_loader, test_loader = create_stratified_dataloaders()
    
    logger.info(f"\nFocal Loss: alpha={Config.FOCAL_ALPHA:.4f} (INVERSE) | gamma={Config.FOCAL_GAMMA}")
    logger.info(f"Early stopping: metrique='{Config.EARLY_STOP_METRIC}' | patience={Config.PATIENCE}")
    logger.info(f"Objectif: Recall >= {Config.MIN_BREAST_RECALL}")
    if Config.USE_WEIGHTED_SAMPLER:
        logger.warning(f"ATTENTION: Sampler + FocalLoss = double correction. Surveiller FP!\n")
    else:
        logger.info(f"Sampler desactive, FocalLoss seul gere desequilibre\n")
    
    # Modele et optimisation
    model = create_model()
    param_groups = get_param_groups(model)
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)
    optimizer = optim.AdamW(param_groups, weight_decay=Config.WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE,
        min_delta=Config.MIN_DELTA,
        metric=Config.EARLY_STOP_METRIC
    )
    
    # Etat initial
    history = defaultdict(list)
    best_recall = 0.0
    best_f1 = 0.0
    start_epoch = 0
    optimal_threshold = 0.5
    
    # Reprise checkpoint
    latest_checkpoint = Config.CHECKPOINT_DIR / 'latest.pth'
    if latest_checkpoint.exists():
        logger.info(f"\nChargement: {latest_checkpoint}")
        try:
            start_epoch, saved_metrics, optimal_threshold = load_checkpoint(
                latest_checkpoint, model, optimizer, scheduler
            )
            best_recall = saved_metrics.get('best_breast_recall', 0.0)
            best_f1 = saved_metrics.get('best_breast_f1', 0.0)
            if optimal_threshold:
                logger.info(f"Seuil optimal: {optimal_threshold:.4f}")
            logger.info(f"Reprise epoch {start_epoch + 1}")
            logger.info(f"Meilleur recall: {best_recall:.4f} | F1: {best_f1:.4f}")
        except Exception as e:
            logger.warning(f"Erreur chargement: {e}")
            start_epoch = 0
    
    # Entrainement
    logger.info(f"\nDebut entrainement\n")
    
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        epoch_start = time.time()
        
        freeze_status = unfreeze_backbone_progressive(model, epoch)
        if epoch == Config.UNFREEZE_EPOCH:
            logger.info(f"Backbone {freeze_status}\n")
        
        logger.info(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        logger.info("-" * 80)
        
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, epoch)
        train_metrics['loss'] = train_loss
        
        # Validation
        val_loss, val_metrics, conf_matrix, val_probs, val_labels = validate(
            model, val_loader, criterion
        )
        val_metrics['loss'] = val_loss
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        # Affichage
        print_results(train_metrics, None, phase='Train')
        print_results(val_metrics, conf_matrix, phase='Val')
        
        # Learning rate robuste
        current_lr = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]['lr']
        logger.info(f"  Temps: {epoch_time:.1f}s | LR: {current_lr:.2e}")
        
        # Historique (inclut LR)
        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                history[f'train_{key}'].append(value)
        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                history[f'val_{key}'].append(value)
        history['learning_rate'].append(current_lr)
        
        # Meilleur modele
        current_recall = val_metrics['breast_recall']
        current_f1 = val_metrics['breast_f1']
        early_stop_score = early_stopping(val_metrics)
        
        save_best = False
        if current_recall > best_recall:
            best_recall = current_recall
            save_best = True
        if current_f1 > best_f1:
            best_f1 = current_f1
            save_best = True
        
        if save_best:
            best_path = Config.CHECKPOINT_DIR / 'best_model.pth'
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {
                    'best_breast_recall': best_recall,
                    'best_breast_f1': best_f1,
                    'breast_precision': val_metrics['breast_precision'],
                    'auc_roc': val_metrics.get('auc_roc', 0),
                    'auc_pr': val_metrics.get('auc_pr', 0),
                    'fn': val_metrics['fn'],
                    'fp': val_metrics['fp']
                },
                best_path,
                threshold=None
            )
            logger.info(f"\nMeilleur modele: Recall={best_recall:.4f} | F1={best_f1:.4f}")
        
        # Checkpoints reguliers
        if (epoch + 1) % Config.SAVE_EVERY == 0:
            checkpoint_path = Config.CHECKPOINT_DIR / f'checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'breast_recall': current_recall, 'breast_f1': current_f1},
                checkpoint_path
            )
        
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            {'best_breast_recall': best_recall, 'best_breast_f1': best_f1},
            latest_checkpoint,
            threshold=None
        )
        
        # Early stopping
        if early_stopping.early_stop:
            logger.info(f"\nEarly stopping (epoch {epoch + 1})")
            logger.info(f"Metrique '{Config.EARLY_STOP_METRIC}' stable depuis {Config.PATIENCE} epochs")
            logger.info(f"Meilleur score: {early_stopping.best_score:.4f}")
            break
        
        logger.info("")
    
    # ========== CALIBRATION ==========
    logger.info("\n" + "=" * 80)
    logger.info("CALIBRATION DU SEUIL (calib set)")
    logger.info("=" * 80)
    
    best_model_path = Config.CHECKPOINT_DIR / 'best_model.pth'
    
    # Guard: verifier existence best_model.pth
    if not best_model_path.exists():
        logger.error(f"ERREUR: {best_model_path} introuvable!")
        logger.error("Aucun meilleur modele sauvegarde. Verifier l'entrainement.")
        return
    
    _, best_metrics, _ = load_checkpoint(best_model_path, model)
    logger.info(f"Meilleur modele charge")
    
    _, calib_metrics, _, calib_probs, calib_labels = validate(model, calib_loader, criterion)
    
    logger.info(f"\nMetriques calib (seuil=0.5):")
    logger.info(f"   Recall: {calib_metrics['breast_recall']:.4f}")
    logger.info(f"   Precision: {calib_metrics['breast_precision']:.4f}")
    
    optimal_threshold, calib_precision, calib_recall = calibrate_threshold(
        calib_probs, calib_labels, target_recall=Config.MIN_BREAST_RECALL
    )
    
    save_checkpoint(
        model, optimizer, scheduler, epoch,
        {
            'best_breast_recall': best_recall,
            'best_breast_f1': best_f1,
            'calibrated_threshold': optimal_threshold,
            'calibrated_precision': calib_precision,
            'calibrated_recall': calib_recall
        },
        best_model_path,
        threshold=optimal_threshold
    )
    
    logger.info(f"Seuil optimal sauvegarde: {optimal_threshold:.4f}")
    
    # ========== TEST ==========
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION FINALE (test set)")
    logger.info("=" * 80)
    
    logger.info(f"Seuil calibre: {optimal_threshold:.4f}")
    
    # Test standard
    logger.info("\nTest seuil=0.5:")
    test_loss, test_metrics, test_cm, test_probs, test_labels = validate(
        model, test_loader, criterion
    )
    test_metrics['loss'] = test_loss
    print_results(test_metrics, test_cm, phase='Test')
    
    # Test calibre
    logger.info(f"\nTest seuil={optimal_threshold:.4f}:")
    calibrated_preds = (np.array(test_probs) >= optimal_threshold).astype(int)
    calib_metrics, calib_cm = compute_metrics(test_labels, calibrated_preds, test_probs)
    calib_metrics['loss'] = test_loss
    print_results(calib_metrics, calib_cm, phase='Test (calibre)')
    
    # Visualisations
    logger.info("\nGeneration graphiques...")
    plot_metrics(history, Config.PLOTS_DIR / 'training_curves.png')
    plot_roc_pr_curves(test_probs, test_labels, Config.PLOTS_DIR / 'roc_pr_curves.png')
    
    # Historique
    history_path = Config.CHECKPOINT_DIR / 'training_history.json'
    with open(history_path, 'w') as f:
        history_clean = {}
        for key, value in history.items():
            if isinstance(value, list):
                history_clean[key] = [
                    float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for v in value
                ]
        json.dump(history_clean, f, indent=4)
    
    # Rapport final
    logger.info("\n" + "=" * 80)
    logger.info("ENTRAINEMENT TERMINE")
    logger.info("=" * 80)
    logger.info(f"\nRESULTATS FINAUX (Test set):")
    logger.info(f"   Standard (seuil=0.5):")
    logger.info(f"     Recall:    {test_metrics['breast_recall']:.4f}")
    logger.info(f"     Precision: {test_metrics['breast_precision']:.4f}")
    logger.info(f"     F1-Score:  {test_metrics['breast_f1']:.4f}")
    logger.info(f"     AUC-ROC:   {test_metrics.get('auc_roc', 0):.4f}")
    
    logger.info(f"\n   Calibre (seuil={optimal_threshold:.4f}):")
    logger.info(f"     Recall:    {calib_metrics['breast_recall']:.4f}")
    logger.info(f"     Precision: {calib_metrics['breast_precision']:.4f}")
    logger.info(f"     F1-Score:  {calib_metrics['breast_f1']:.4f}")
    logger.info(f"     FN: {calib_metrics['fn']} | FP: {calib_metrics['fp']}")
    
    # Verdict
    if calib_metrics['breast_recall'] >= Config.MIN_BREAST_RECALL:
        total_breast = calib_metrics['fn'] + calib_metrics['tp']
        miss_rate = calib_metrics['fn'] / total_breast * 100 if total_breast > 0 else 0
        logger.info(f"\nOBJECTIF ATTEINT")
        logger.info(f"   Images breast manquees: {calib_metrics['fn']}/{total_breast}")
        logger.info(f"   Taux de perte: {miss_rate:.2f}%")
    else:
        logger.warning(f"\nOBJECTIF NON ATTEINT")
        logger.warning(f"   Recall: {calib_metrics['breast_recall']:.4f} < {Config.MIN_BREAST_RECALL}")
        logger.info(f"\nRecommandations:")
        logger.info(f"     1. Augmenter FOCAL_ALPHA a 0.85")
        logger.info(f"     2. Augmenter NUM_EPOCHS a 150")
        logger.info(f"     3. Verifier qualite annotations")
        logger.info(f"     4. Ajouter plus de donnees breast")
    
    logger.info(f"\nFichiers generes:")
    logger.info(f"   {Config.CHECKPOINT_DIR}/best_model.pth")
    logger.info(f"   {Config.CHECKPOINT_DIR}/training_history.json")
    logger.info(f"   {Config.PLOTS_DIR}/training_curves.png")
    logger.info(f"   {Config.PLOTS_DIR}/roc_pr_curves.png")
    
    logger.info(f"\nUtilisation production:")
    logger.info(f"   1. Charger best_model.pth")
    logger.info(f"   2. Utiliser seuil: {optimal_threshold:.4f}")
    logger.info(f"   3. Classifier 'breast' si P(breast) >= {optimal_threshold:.4f}")
    logger.info("=" * 80)
    
    # Config production
    prod_config = {
        'model_path': str(Config.CHECKPOINT_DIR / 'best_model.pth'),
        'optimal_threshold': float(optimal_threshold),
        'img_size': Config.IMG_SIZE,
        'normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'test_metrics': {
            'recall': float(calib_metrics['breast_recall']),
            'precision': float(calib_metrics['breast_precision']),
            'f1': float(calib_metrics['breast_f1']),
            'auc_roc': float(calib_metrics.get('auc_roc', 0)),
            'fn': int(calib_metrics['fn']),
            'fp': int(calib_metrics['fp'])
        }
    }
    
    prod_config_path = Config.CHECKPOINT_DIR / 'production_config.json'
    with open(prod_config_path, 'w') as f:
        json.dump(prod_config, f, indent=4)
    
    logger.info(f"\nConfig production: {prod_config_path}")
    logger.info("\nEntrainement termine!")


if __name__ == '__main__':
    main()