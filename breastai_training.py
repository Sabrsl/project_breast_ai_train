#!/usr/bin/env python3
"""
BreastAI Training System v3.3.0 - SIMPLIFIÉ ET FONCTIONNEL
Architecture médicale EfficientNetV2 + CBAM pour classification cancer du sein
Réécriture complète pour compatibilité totale avec l'interface web
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, cohen_kappa_score, roc_auc_score,
    f1_score, precision_score, recall_score  # Pour validation complète
)

# Configuration de base
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/training.log')
    ]
)
logger = logging.getLogger(__name__)

# ==================================================================================
# EARLY STOPPING
# ==================================================================================

class EarlyStopping:
    """
    Early Stopping pour arrêter l'entraînement si pas d'amélioration
    
    Args:
        patience (int): Nombre d'epochs sans amélioration avant d'arrêter
        min_delta (float): Amélioration minimale pour considérer un progrès
        mode (str): 'min' pour loss, 'max' pour accuracy/f1
    """
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Vérifie si l'entraînement doit s'arrêter
        
        Args:
            score: Métrique à surveiller (accuracy, f1, loss, etc.)
            
        Returns:
            True si l'entraînement doit s'arrêter
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        # Calculer si amélioration
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:  # mode == 'min'
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"⏹️ EARLY STOPPING : pas d'amélioration depuis {self.patience} epochs")
                return True
            else:
                logger.info(f"⏳ Early stopping counter: {self.counter}/{self.patience}")
                return False
    
    def reset(self):
        """Réinitialise le compteur"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False

# ==================================================================================
# CONFIGURATION
# ==================================================================================

class Config:
    """Configuration centralisée et simple"""
    
    def __init__(self, config_dict: Optional[Dict] = None):
        # Charger config.json par défaut
        config_path = Path('config.json')
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        # Fusionner avec config custom
        if config_dict:
            self._merge_config(config_dict)
    
    def _merge_config(self, custom: Dict):
        """Fusion simple des configs"""
        for key, value in custom.items():
            if isinstance(value, dict) and key in self.config:
                self.config[key].update(value)
            else:
                self.config[key] = value
    
    def get(self, *keys, default=None):
        """Accès simple aux valeurs imbriquées"""
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default

# ==================================================================================
# MODULES CBAM
# ==================================================================================

class ChannelAttention(nn.Module):
    """Attention sur les canaux"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        mid_channels = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    """Attention spatiale"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(out))

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

# ==================================================================================
# FOCAL LOSS
# ==================================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss pour gérer le déséquilibre des classes
    FL(pt) = -alpha * (1-pt)^gamma * log(pt)
    """
    def __init__(self, alpha=None, gamma=2.5, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, list):
                self.alpha = torch.tensor(alpha)
            else:
                self.alpha = alpha
        else:
            self.alpha = None
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probabilité de la vraie classe
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ==================================================================================
# MODÈLE
# ==================================================================================

class BreastAIModel(nn.Module):
    """Modèle EfficientNetV2 + CBAM simple et efficace"""
    
    def __init__(self, architecture: str = 'efficientnetv2_s', num_classes: int = 3,
                 use_cbam: bool = True, dropout: float = 0.4):
        super().__init__()
        
        # Backbone
        if architecture == 'efficientnetv2_s':
            self.backbone = models.efficientnet_v2_s(weights='DEFAULT')
            num_features = 1280
        elif architecture == 'efficientnetv2_m':
            self.backbone = models.efficientnet_v2_m(weights='DEFAULT')
            num_features = 1280
        elif architecture == 'efficientnetv2_l':
            self.backbone = models.efficientnet_v2_l(weights='DEFAULT')
            num_features = 1280
        else:
            raise ValueError(f"Architecture non supportée: {architecture}")
        
        # Retirer le classifier
        self.backbone.classifier = nn.Identity()
        
        # CBAM optionnel
        self.cbam = CBAM(num_features) if use_cbam else None
        
        # Classifier médical
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.3),
            nn.Linear(256, num_classes)
        )
        
        logger.info(f"Modèle créé: {architecture}, classes={num_classes}, CBAM={use_cbam}")
    
    def forward(self, x):
        # Features du backbone
        x = self.backbone.features(x)
        
        # CBAM
        if self.cbam is not None:
            x = self.cbam(x)
        
        # Global average pooling
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        return x

# ==================================================================================
# DATASET
# ==================================================================================

class MedicalDataset(Dataset):
    """Dataset médical avec CLAHE et augmentations"""
    
    def __init__(self, root_dir: Path, transform=None, use_clahe: bool = True):
        self.root_dir = root_dir
        self.transform = transform
        self.use_clahe = use_clahe
        
        # Scanner les classes
        self.classes = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Scanner les images
        self.samples = []
        for class_name in self.classes:
            class_dir = root_dir / class_name
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        logger.info(f"Dataset: {len(self.samples)} images, {len(self.classes)} classes")
    
    def apply_clahe(self, image):
        """Applique CLAHE pour améliorer le contraste"""
        if len(image.shape) == 2:  # Grayscale
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        else:  # RGB
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Charger l'image
            image = cv2.imread(img_path)
            
            # 🛡️ Validation : image corrompue
            if image is None:
                raise IOError(f"Image corrompue ou inaccessible: {img_path}")
            
            # 🛡️ Validation : dimensions minimales
            if image.shape[0] < 50 or image.shape[1] < 50:
                raise ValueError(f"Image trop petite ({image.shape[0]}x{image.shape[1]}): {img_path}")
            
            # 🛡️ Validation : image entièrement noire (possiblement corrompue)
            if np.all(image == 0):
                raise ValueError(f"Image entièrement noire: {img_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # CLAHE
            if self.use_clahe:
                image = self.apply_clahe(image)
            
            # Convertir en PIL
            image = Image.fromarray(image)
            
            # Appliquer les transformations
            if self.transform:
                image = self.transform(image)
            
            # 🛡️ Validation finale : vérifier qu'il n'y a pas de NaN après transformations
            if torch.isnan(image).any():
                raise ValueError(f"NaN détecté après transformations: {img_path}")
            
            return image, label
            
        except (IOError, OSError) as e:
            # Erreur d'accès fichier - propager l'exception
            logger.error(f"❌ Erreur I/O: {img_path} - {e}")
            raise
        
        except ValueError as e:
            # Erreur de validation - propager l'exception
            logger.error(f"❌ Validation échouée: {img_path} - {e}")
            raise
        
        except Exception as e:
            # Erreur inattendue - propager avec contexte
            logger.error(f"❌ Erreur inattendue: {img_path} - {type(e).__name__}: {e}")
            raise RuntimeError(f"Erreur chargement image: {img_path}") from e

# ==================================================================================
# SYSTÈME D'ENTRAÎNEMENT
# ==================================================================================

class TrainingSystem:
    """Système d'entraînement simple et direct"""
    
    def __init__(self, config: Config, callback: Optional[Callable] = None):
        self.config = config
        self.callback = callback  # Fonction pour envoyer des updates
        
        # 🚀 Détection automatique GPU/CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"🎮 GPU détecté: {torch.cuda.get_device_name(0)}")
            logger.info(f"💾 VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            self.device = torch.device('cpu')
            logger.warning("WARNING: Aucun GPU detecte - Utilisation du CPU (entrainement plus lent)")
        
        self.is_training = False
        
        # Composants
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # 🆕 Gradient Accumulation & EMA
        self.gradient_accumulation_steps = 1  # Sera mis à jour au setup
        self.accumulation_counter = 0
        self.use_ema = False
        self.ema_decay = 0.9998
        self.model_ema = None
        
        # 🆕 Early Stopping (classe dédiée)
        early_stop_config = config.config.get('training', {}).get('early_stopping', {})
        early_stop_patience = early_stop_config.get('patience', 10) if isinstance(early_stop_config, dict) else 10
        early_stop_min_delta = early_stop_config.get('min_delta', 0.001) if isinstance(early_stop_config, dict) else 0.001
        
        self.early_stopping = EarlyStopping(
            patience=early_stop_patience,
            min_delta=early_stop_min_delta,
            mode='max'  # Surveiller F1-macro (maximiser)
        )
        self.early_stopping_enabled = early_stop_config.get('enabled', True) if isinstance(early_stop_config, dict) else True
        
        # 🆕 AMP (Automatic Mixed Precision)
        self.use_amp = torch.cuda.is_available()  # Activé seulement si GPU
        self.scaler = None
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            logger.info("AMP active : entrainement 2-3x plus rapide")
        
        # Métriques
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.worst_val_acc = 100.0  # Pour analyse
        self.history = defaultdict(list)
        
        logger.info("Système d'entraînement initialisé")
    
    async def send_update(self, message: Dict):
        """Envoie une mise à jour via callback"""
        if self.callback:
            try:
                await self.callback(message)
            except Exception as e:
                logger.error(f"Erreur callback: {e}")
    
    async def setup(self) -> bool:
        """Configure tout le système"""
        try:
            await self.send_update({
                'type': 'progress_update',
                'stage': 'setup',
                'progress': 0.1,
                'message': 'Chargement des données...'
            })
            
            # 1. Charger les données
            if not await self._setup_data():
                return False
            
            await self.send_update({
                'type': 'progress_update',
                'stage': 'setup',
                'progress': 0.5,
                'message': 'Construction du modèle...'
            })
            
            # 2. Créer le modèle
            architecture = self.config.get('model', 'architecture', default='efficientnetv2_s')
            num_classes = self.config.get('model', 'num_classes', default=3)
            use_cbam = self.config.get('model', 'use_cbam', default=True)
            dropout = self.config.get('model', 'dropout_rate', default=0.4)
            
            self.model = BreastAIModel(architecture, num_classes, use_cbam, dropout)
            self.model.to(self.device)
            
            # 🆕 Configuration Gradient Accumulation
            self.gradient_accumulation_steps = self.config.get('data', 'gradient_accumulation_steps', default=1)
            batch_size = self.config.get('data', 'batch_size', default=4)
            effective_batch = batch_size * self.gradient_accumulation_steps
            
            if self.gradient_accumulation_steps > 1:
                logger.info(f"🔄 Gradient Accumulation: {self.gradient_accumulation_steps} steps")
                logger.info(f"   Batch physique: {batch_size} | Batch effectif: {effective_batch}")
            
            # 🆕 Configuration EMA
            self.use_ema = self.config.get('training', 'use_ema', default=False)
            self.ema_decay = self.config.get('training', 'ema_decay', default=0.9998)
            
            if self.use_ema:
                import copy
                self.model_ema = copy.deepcopy(self.model)
                self.model_ema.eval()
                for param in self.model_ema.parameters():
                    param.requires_grad = False
                logger.info(f"✅ EMA activé avec decay={self.ema_decay}")
            
            await self.send_update({
                'type': 'progress_update',
                'stage': 'setup',
                'progress': 0.7,
                'message': 'Configuration optimizer...'
            })
            
            # 3. Optimizer & Loss
            lr = self.config.get('training', 'learning_rate', default=0.0003)
            weight_decay = self.config.get('training', 'weight_decay', default=0.0001)
            
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            
            # 🆕 Loss function : Focal Loss OU CrossEntropy
            focal_config = self.config.config.get('training', {}).get('focal_loss', {})
            use_focal = focal_config.get('enabled', False) if isinstance(focal_config, dict) else False
            
            if use_focal:
                alpha = focal_config.get('alpha', [0.25, 0.50, 0.25])  # Priorité malignant
                gamma = focal_config.get('gamma', 2.5)
                self.criterion = FocalLoss(alpha=alpha, gamma=gamma)
                logger.info(f"🎯 Loss: FocalLoss avec alpha={alpha}, gamma={gamma}")
                logger.info(f"   → Focus sur classe malignant (alpha={alpha[1]})")
            else:
                label_smoothing = self.config.get('training', 'label_smoothing', default=0.1)
                self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
                logger.info(f"Loss: CrossEntropyLoss avec label_smoothing={label_smoothing}")
            
            # 4. Scheduler
            epochs = self.config.get('training', 'epochs', default=50)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
            
            await self.send_update({
                'type': 'progress_update',
                'stage': 'setup',
                'progress': 1.0,
                'message': 'Configuration terminée!'
            })
            
            logger.info("Setup terminé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur setup: {e}", exc_info=True)
            await self.send_update({
                'type': 'error',
                'message': f'Erreur setup: {str(e)}'
            })
            return False
    
    async def _setup_data(self) -> bool:
        """Configure les dataloaders"""
        try:
            # Chemins
            data_dir = Path(self.config.get('paths', 'data_dir', default='data'))
            
            # Corriger le chemin si nécessaire
            if data_dir.name in ['train', 'val', 'test']:
                data_dir = data_dir.parent
            
            train_dir = data_dir / 'train'
            val_dir = data_dir / 'val'
            test_dir = data_dir / 'test'
            
            logger.info(f"Data: train={train_dir}, val={val_dir}, test={test_dir}")
            
            if not train_dir.exists():
                raise ValueError(f"Répertoire train introuvable: {train_dir}")
            
            # Transformations
            image_size = self.config.get('data', 'image_size', default=512)
            
            train_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.3),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            val_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # Datasets
            train_dataset = MedicalDataset(train_dir, train_transform, use_clahe=True)
            
            # Class weights pour le sampler
            labels = [label for _, label in train_dataset.samples]
            class_counts = Counter(labels)
            weights = [1.0 / class_counts[label] for label in labels]
            sampler = WeightedRandomSampler(weights, len(weights))
            
            # DataLoaders
            batch_size = self.config.get('data', 'batch_size', default=4)
            num_workers = self.config.get('data', 'num_workers', default=4)
            
            self.train_loader = DataLoader(
                train_dataset, batch_size=batch_size,
                sampler=sampler, num_workers=num_workers,
                pin_memory=False
            )
            
            if val_dir.exists():
                val_dataset = MedicalDataset(val_dir, val_transform, use_clahe=True)
                self.val_loader = DataLoader(
                    val_dataset, batch_size=batch_size,
                    shuffle=False, num_workers=num_workers
                )
            
            if test_dir.exists():
                test_dataset = MedicalDataset(test_dir, val_transform, use_clahe=True)
                self.test_loader = DataLoader(
                    test_dataset, batch_size=batch_size,
                    shuffle=False, num_workers=num_workers
                )
            
            logger.info(f"Données chargées: {len(train_dataset)} train, "
                       f"{len(val_dataset) if val_dir.exists() else 0} val")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur setup data: {e}", exc_info=True)
            return False
    
    def _apply_progressive_unfreezing(self, epoch: int, total_epochs: int):
        """
        🔓 PROGRESSIVE UNFREEZING 4 PHASES - Optimisation CPU
        
        Phase 1 (1-8)   : Backbone gelé → ×3 rapide
        Phase 2 (9-20)  : Dégel 25% → ×2 rapide
        Phase 3 (21-40) : Dégel 50% → ×1.5 rapide
        Phase 4 (41+)   : Dégel 100% → vitesse normale
        """
        if not hasattr(self.model, 'backbone'):
            return  # Pas de backbone à geler
        
        # Récupérer config ou utiliser défauts
        phase1_end = self.config.get('model', 'progressive_unfreezing', {}).get('phase1_epochs', 8)
        phase2_end = self.config.get('model', 'progressive_unfreezing', {}).get('phase2_epochs', 20)
        phase3_end = self.config.get('model', 'progressive_unfreezing', {}).get('phase3_epochs', 40)
        
        # PHASE 1 : Epochs 1-8 → BACKBONE 100% GELÉ
        if epoch <= phase1_end:
            if epoch == 1:
                # Geler tout le backbone
                for param in self.model.backbone.parameters():
                    param.requires_grad = False
                
                # Compter les paramètres
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                logger.info(f"🔒 [Phase 1/4] Backbone GELÉ - Epochs 1-{phase1_end} (×3 plus rapide)")
                logger.info(f"   → Paramètres entraînables: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
            return
        
        # PHASE 2 : Epochs 9-20 → DÉGEL 25%
        elif epoch == phase1_end + 1:
            logger.info(f"🔓 [Phase 2/4] Dégel 25% - Epochs {phase1_end+1}-{phase2_end} (×2 plus rapide)")
            
            if hasattr(self.model.backbone, 'features'):
                total_blocks = len(self.model.backbone.features)
                unfreeze_from = int(total_blocks * 0.75)  # 25% des derniers blocs
                
                for idx, block in enumerate(self.model.backbone.features):
                    if idx >= unfreeze_from:
                        for param in block.parameters():
                            param.requires_grad = True
                
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in self.model.parameters())
                logger.info(f"   → Blocs {unfreeze_from}-{total_blocks} dégelés ({100*trainable/total:.1f}% params)")
            
            # Réduire LR
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5
            logger.info(f"   → Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            return
        
        elif epoch <= phase2_end:
            return  # Phase 2 en cours
        
        # PHASE 3 : Epochs 21-40 → DÉGEL 50%
        elif epoch == phase2_end + 1:
            logger.info(f"🔥 [Phase 3/4] Dégel 50% - Epochs {phase2_end+1}-{phase3_end} (×1.5 plus rapide)")
            
            if hasattr(self.model.backbone, 'features'):
                total_blocks = len(self.model.backbone.features)
                unfreeze_from = int(total_blocks * 0.50)  # 50% des blocs
                
                for idx, block in enumerate(self.model.backbone.features):
                    if idx >= unfreeze_from:
                        for param in block.parameters():
                            param.requires_grad = True
                
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in self.model.parameters())
                logger.info(f"   → Blocs {unfreeze_from}-{total_blocks} dégelés ({100*trainable/total:.1f}% params)")
            
            # Réduire encore LR
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.6
            logger.info(f"   → Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            return
        
        elif epoch <= phase3_end:
            return  # Phase 3 en cours
        
        # PHASE 4 : Epochs 41+ → DÉGEL 100% COMPLET
        elif epoch == phase3_end + 1:
            logger.info(f"💪 [Phase 4/4] Dégel 100% COMPLET - Epochs {phase3_end+1}+ (vitesse normale)")
            
            # Dégeler TOUT le backbone
            for param in self.model.backbone.parameters():
                param.requires_grad = True
            
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            logger.info(f"   → TOUS les paramètres dégelés ({100*trainable/total:.1f}% = 100%)")
            
            # LR finale réduite
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.33
            logger.info(f"   → Learning rate final: {self.optimizer.param_groups[0]['lr']:.2e}")
            return
    
    async def train(self, epochs: Optional[int] = None, start_epoch: int = 1):
        """Lance l'entraînement (supporte reprise depuis checkpoint)"""
        if self.is_training:
            logger.warning("Entraînement déjà en cours")
            return
        
        self.is_training = True
        epochs = epochs or self.config.get('training', 'epochs', default=50)
        
        try:
            await self.send_update({
                'type': 'training_started',
                'total_epochs': epochs,
                'start_epoch': start_epoch,
                'resuming': start_epoch > 1,
                'timestamp': datetime.now().isoformat()
            })
            
            if start_epoch > 1:
                logger.info(f"REPRISE ENTRAÎNEMENT: epochs {start_epoch} à {epochs}")
            else:
                logger.info(f"DÉMARRAGE ENTRAÎNEMENT: {epochs} epochs")
            
            for epoch in range(start_epoch, epochs + 1):
                if not self.is_training:
                    logger.info("Entraînement arrêté par l'utilisateur")
                    break
                
                # 🔓 Progressive Unfreezing (optimisation CPU)
                self._apply_progressive_unfreezing(epoch, epochs)
                
                # Train
                train_metrics = await self._train_epoch(epoch)
                
                # Validation
                val_metrics = await self._validate_epoch(epoch)
                
                # Update détaillé pour l'interface
                await self.send_update({
                    'type': 'training_update',
                    'epoch': epoch,
                    'total_epochs': epochs,
                    'train_loss': float(train_metrics['loss']),
                    'train_accuracy': float(train_metrics['accuracy']),
                    'val_loss': float(val_metrics['loss']),
                    'val_accuracy': float(val_metrics['accuracy']),
                    'val_f1_macro': float(val_metrics.get('f1_macro', 0)),
                    'val_f1_weighted': float(val_metrics.get('f1_weighted', 0)),
                    'val_precision_macro': float(val_metrics.get('precision_macro', 0)),
                    'val_recall_macro': float(val_metrics.get('recall_macro', 0)),
                    'val_precision_weighted': float(val_metrics.get('precision_weighted', 0)),
                    'val_recall_weighted': float(val_metrics.get('recall_weighted', 0)),
                    'learning_rate': float(self.optimizer.param_groups[0]['lr']),
                    'progress': (epoch / epochs) * 100,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Scheduler
                self.scheduler.step()
                
                # 🎯 Sauvegarder best/worst models
                current_f1 = val_metrics.get('f1_macro', 0)
                current_acc = val_metrics['accuracy']
                
                # Best model (basé sur F1-macro)
                if current_f1 > self.best_val_f1:
                    self.best_val_f1 = current_f1
                    self.best_val_acc = current_acc
                    self._save_checkpoint(epoch, 'best.pth')
                    logger.info(f"✅ Nouveau meilleur modèle : F1={current_f1:.4f}, Acc={current_acc:.2f}%")
                
                # Worst model (sauvegardé périodiquement pour analyse, pas à chaque dégradation)
                # Sauvegarder seulement tous les 10 epochs ou à la fin
                if current_acc < self.worst_val_acc:
                    self.worst_val_acc = current_acc
                    # Sauvegarder uniquement tous les 10 epochs ou à la fin
                    if epoch % 10 == 0 or epoch == epochs:
                        self._save_checkpoint(epoch, 'worst.pth')
                        logger.info(f"📉 Pire modèle sauvegardé (analyse) : Acc={current_acc:.2f}% à epoch {epoch}")
                
                # 🛑 Early Stopping avec classe dédiée
                if self.early_stopping_enabled and self.early_stopping(current_f1):
                    await self.send_update({
                        'type': 'log',
                        'message': f'⏹️ Early stopping : pas d\'amélioration depuis {self.early_stopping.patience} epochs',
                        'level': 'warning'
                    })
                    break
                
                # Save periodic
                if epoch % 10 == 0:
                    self._save_checkpoint(epoch, f'epoch_{epoch}.pth')
            
            # Terminé
            await self.send_update({
                'type': 'training_complete',
                'final_metrics': {'best_val_accuracy': self.best_val_acc},
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"ENTRAÎNEMENT TERMINÉ! Best accuracy: {self.best_val_acc:.4f}")
            
        except Exception as e:
            logger.error(f"Erreur entraînement: {e}", exc_info=True)
            await self.send_update({
                'type': 'error',
                'message': f'Erreur entraînement: {str(e)}'
            })
        finally:
            self.is_training = False
    
    async def _train_epoch(self, epoch: int) -> Dict:
        """Une epoch d'entraînement avec Gradient Accumulation & EMA"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        skipped_batches = 0
        
        # Reset accumulation counter au début de l'epoch
        self.optimizer.zero_grad()
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            try:
                # 🛡️ Vérification batch vide
                if images.size(0) == 0:
                    logger.warning(f"Batch {batch_idx} vide - skip")
                    continue
                
                # 🛡️ Vérification NaN
                if torch.isnan(images).any() or torch.isnan(labels.float()).any():
                    logger.error(f"Batch {batch_idx} contient NaN - skip")
                    skipped_batches += 1
                    continue
                
                images, labels = images.to(self.device), labels.to(self.device)
                
                # ⚡ MIXED PRECISION (AMP)
                if self.use_amp:
                    from torch.cuda.amp import autocast
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                        
                        if self.gradient_accumulation_steps > 1:
                            loss = loss / self.gradient_accumulation_steps
                    
                    self.scaler.scale(loss).backward()
                else:
                    # Standard FP32
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                    if self.gradient_accumulation_steps > 1:
                        loss = loss / self.gradient_accumulation_steps
                    
                    loss.backward()
                
                # Accumulation counter
                self.accumulation_counter += 1
                
                # Optimizer step seulement tous les N batches
                if self.accumulation_counter % self.gradient_accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    
                    # ✅ EMA UPDATE (après optimizer step)
                    if self.use_ema and self.model_ema is not None:
                        self._update_ema()
                
                # Métriques (attention: loss déjà divisée si accumulation)
                if self.gradient_accumulation_steps > 1:
                    total_loss += loss.item() * self.gradient_accumulation_steps
                else:
                    total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
            except torch.cuda.OutOfMemoryError as e:
                logger.critical(f"❌ OUT OF MEMORY à batch {batch_idx}!")
                logger.critical(f"💡 Solution : Réduire batch_size ou activer gradient_accumulation")
                await self.send_update({
                    'type': 'error',
                    'message': '❌ OUT OF MEMORY - Réduire batch_size'
                })
                raise RuntimeError("OOM - Réduire batch_size") from e
            
            except (IOError, OSError) as e:
                logger.error(f"❌ Erreur I/O batch {batch_idx}: {e}")
                skipped_batches += 1
                if skipped_batches > len(self.train_loader) * 0.1:  # >10% erreurs
                    raise RuntimeError(f"Trop d'erreurs I/O : {skipped_batches} batches") from e
                await self.send_update({
                    'type': 'log',
                    'message': f'⚠️ Erreur I/O batch {batch_idx}, skip',
                    'level': 'warning'
                })
                continue
            
            except ValueError as e:
                logger.warning(f"⚠️ Donnée invalide batch {batch_idx}: {e}")
                skipped_batches += 1
                continue
            
            except Exception as e:
                logger.error(f"❌ Erreur inattendue batch {batch_idx}: {type(e).__name__}: {e}")
                skipped_batches += 1
                if skipped_batches > len(self.train_loader) * 0.2:  # >20% erreurs
                    raise RuntimeError(f"Trop d'erreurs : {skipped_batches} batches") from e
                await self.send_update({
                    'type': 'log',
                    'message': f'⚠ Erreur batch {batch_idx}, skip',
                    'level': 'warning'
                })
                continue  # Passer à la batch suivante
            
            # ⚡ TEMPS RÉEL : Envoyer à CHAQUE BATCH
            current_acc = 100. * correct / total if total > 0 else 0
            batch_progress = (batch_idx / len(self.train_loader)) * 100
            
            # Log fichier (toutes les 10 batches pour ne pas surcharger)
            if batch_idx % 10 == 0:
                log_msg = f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] Loss: {loss.item():.4f} Acc: {current_acc:.2f}%"
                logger.info(log_msg)
            
            # ⚡ Interface web EN TEMPS RÉEL (chaque batch)
            log_msg = f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] Loss: {loss.item():.4f} Acc: {current_acc:.2f}%"
            await self.send_update({
                'type': 'log',
                'message': log_msg,
                'level': 'info',
                'timestamp': datetime.now().isoformat()
            })
            
            # Progression détaillée EN TEMPS RÉEL
            await self.send_update({
                'type': 'batch_progress',
                'epoch': epoch,
                'batch': batch_idx,
                'total_batches': len(self.train_loader),
                'batch_progress': batch_progress,
                'current_loss': float(loss.item()),
                'current_accuracy': float(current_acc),
                'timestamp': datetime.now().isoformat()
            })
        
        avg_loss = total_loss / (len(self.train_loader) - skipped_batches) if len(self.train_loader) > skipped_batches else 0
        avg_acc = 100. * correct / total if total > 0 else 0
        
        if skipped_batches > 0:
            logger.warning(f"Epoch {epoch}: {skipped_batches} batches skipped due to errors")
        
        logger.info(f"Epoch {epoch} Training: Loss={avg_loss:.4f}, Acc={avg_acc:.2f}%")
        
        return {
            'loss': avg_loss,
            'accuracy': avg_acc
        }
    
    def _update_ema(self):
        """
        ✅ Update EMA model
        EMA: model_ema = decay * model_ema + (1 - decay) * model
        """
        with torch.no_grad():
            for ema_param, model_param in zip(self.model_ema.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)
    
    async def _validate_epoch(self, epoch: int) -> Dict:
        """Validation avec TTA optionnel"""
        if self.val_loader is None:
            return {'loss': 0, 'accuracy': 0, 'f1_macro': 0}
        
        # 🆕 Vérifier si TTA activé
        use_tta = self.config.get('inference', 'tta_enabled', False)
        
        if use_tta:
            return await self._validate_with_tta(epoch)
        else:
            return await self._validate_standard(epoch)
    
    async def _validate_standard(self, epoch: int) -> Dict:
        """Validation standard (sans TTA)"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        await self.send_update({
            'type': 'log',
            'message': f'Validation epoch {epoch} en cours...',
            'level': 'info'
        })
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Métriques macro et weighted
        accuracy = accuracy_score(all_labels, all_preds) * 100
        
        # Macro (moyenne simple)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        # Weighted (moyenne pondérée par support)
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        val_msg = f"Epoch {epoch} Validation: Loss={total_loss/len(self.val_loader):.4f}, Acc={accuracy:.2f}%, F1-Macro={f1_macro:.4f}, F1-Weighted={f1_weighted:.4f}"
        logger.info(val_msg)
        
        # Envoyer à l'interface
        await self.send_update({
            'type': 'log',
            'message': val_msg,
            'level': 'success'
        })
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted
        }
    
    async def _validate_with_tta(self, epoch: int) -> Dict:
        """
        🔄 Validation avec Test-Time Augmentation
        Applique 6 transformations et moyenne les prédictions
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        await self.send_update({
            'type': 'log',
            'message': f'🔄 Validation epoch {epoch} avec TTA (6x augmentations)...',
            'level': 'info'
        })
        
        # Définir les transformations TTA
        tta_transforms = [
            lambda x: x,  # Original
            lambda x: torch.flip(x, dims=[3]),  # Horizontal flip
            lambda x: torch.flip(x, dims=[2]),  # Vertical flip
            lambda x: x,  # Rotation skip (complexe)
            lambda x: torch.clamp(x * 1.05, 0, 1),  # Brightness +5%
            lambda x: torch.clamp((x - x.mean(dim=[2,3], keepdim=True)) * 1.05 + x.mean(dim=[2,3], keepdim=True), 0, 1),  # Contrast
        ]
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Collecter prédictions de toutes les augmentations
                predictions = []
                for transform in tta_transforms:
                    aug_images = transform(images)
                    outputs = self.model(aug_images)
                    probs = F.softmax(outputs, dim=1)
                    predictions.append(probs)
                
                # Moyenne des prédictions
                final_probs = torch.mean(torch.stack(predictions), dim=0)
                _, predicted = final_probs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculer métriques (pas de loss avec TTA)
        accuracy = accuracy_score(all_labels, all_preds) * 100
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        val_msg = f"Validation TTA - Acc: {accuracy:.2f}%, F1-macro: {f1_macro:.3f}, F1-weighted: {f1_weighted:.3f}"
        logger.info(val_msg)
        
        await self.send_update({
            'type': 'log',
            'message': val_msg,
            'level': 'success'
        })
        
        return {
            'loss': 0,  # Pas de loss avec TTA
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted
        }
    
    def _validate_checkpoint_path(self, checkpoint_path: str, check_exists: bool = False) -> Path:
        """
        Valide et sécurise le chemin d'un checkpoint
        
        Args:
            checkpoint_path: Chemin du checkpoint à valider
            check_exists: Si True, vérifie que le fichier existe
            
        Returns:
            Path validé et résolu
            
        Raises:
            ValueError: Si le chemin est invalide ou en dehors du dossier checkpoints
            FileNotFoundError: Si check_exists=True et le fichier n'existe pas
        """
        try:
            path = Path(checkpoint_path).resolve()
            checkpoint_dir = Path(self.config.get('paths', 'checkpoint_dir', default='checkpoints')).resolve()
            
            # 🛡️ Vérifier que le chemin est dans checkpoint_dir (protection path traversal)
            try:
                path.relative_to(checkpoint_dir)
            except ValueError:
                raise ValueError(f"⚠️ Chemin checkpoint invalide (hors de '{checkpoint_dir}'): {checkpoint_path}")
            
            # 🛡️ Vérifier que le fichier existe (pour load_checkpoint)
            if check_exists and not path.exists():
                raise FileNotFoundError(f"❌ Checkpoint introuvable: {path}")
            
            # 🛡️ Vérifier l'extension .pth
            if path.suffix != '.pth':
                raise ValueError(f"⚠️ Format checkpoint invalide (attendu .pth, reçu {path.suffix})")
            
            return path
        
        except (ValueError, FileNotFoundError) as e:
            logger.error(f"❌ Validation checkpoint échouée: {e}")
            raise
        
        except Exception as e:
            logger.error(f"❌ Erreur inattendue lors de la validation: {type(e).__name__}: {e}")
            raise RuntimeError(f"Erreur validation checkpoint: {checkpoint_path}") from e
    
    def _save_checkpoint(self, epoch: int, filename: str):
        """Sauvegarde un checkpoint avec métadonnées complètes et validation du chemin"""
        checkpoint_dir = Path(self.config.get('paths', 'checkpoint_dir', default='checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 🛡️ Validation du chemin (sécurité, pas besoin de vérifier existence pour save)
        try:
            checkpoint_path = self._validate_checkpoint_path(filename, check_exists=False)
        except ValueError as e:
            logger.error(f"❌ Erreur validation chemin : {e}")
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            # Métadonnées
            'architecture': self.config.get('model', 'architecture', default='unknown'),
            'num_classes': self.config.get('model', 'num_classes', default=3),
            'use_cbam': self.config.get('model', 'use_cbam', default=True),
            'image_size': self.config.get('data', 'image_size', default=512),
            'timestamp': datetime.now().isoformat(),
            'config': self.config.config  # Config complète pour reproductibilité
        }
        
        torch.save(checkpoint, checkpoint_dir / filename)
        logger.info(f"Checkpoint sauvegardé: {filename} (epoch {epoch}, acc {self.best_val_acc:.2f}%)")
    
    async def stop(self):
        """Arrête l'entraînement"""
        self.is_training = False
        await self.send_update({
            'type': 'training_stopped',
            'timestamp': datetime.now().isoformat()
        })
        logger.info("Arrêt demandé")
    
    async def export_onnx(self, checkpoint_path: Optional[str] = None) -> bool:
        """Exporte le modèle en ONNX"""
        try:
            await self.send_update({
                'type': 'log',
                'message': 'Début export ONNX...',
                'level': 'info'
            })
            
            # Si checkpoint spécifié, charger le modèle
            if checkpoint_path:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Modèle chargé depuis {checkpoint_path}")
            
            # Créer le dossier exports
            export_dir = Path(self.config.get('paths', 'export_dir', default='exports'))
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Nom du fichier
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            onnx_path = export_dir / f'breastai_{timestamp}.onnx'
            
            # Mettre le modèle en mode eval
            self.model.eval()
            
            # Input dummy
            image_size = self.config.get('data', 'image_size', default=512)
            dummy_input = torch.randn(1, 3, image_size, image_size).to(self.device)
            
            await self.send_update({
                'type': 'log',
                'message': 'Conversion en cours...',
                'level': 'info'
            })
            
            # Export ONNX
            torch.onnx.export(
                self.model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # Vérifier la taille
            file_size = onnx_path.stat().st_size / (1024 * 1024)  # MB
            
            await self.send_update({
                'type': 'export_complete',
                'path': str(onnx_path),
                'size': file_size,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Export ONNX réussi: {onnx_path} ({file_size:.2f} MB)")
            return True
            
        except Exception as e:
            logger.error(f"Erreur export ONNX: {e}", exc_info=True)
            await self.send_update({
                'type': 'error',
                'message': f'Erreur export ONNX: {str(e)}'
            })
            return False
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Charge un checkpoint pour reprendre l'entraînement avec validation du chemin"""
        try:
            # 🛡️ Validation du chemin (sécurité + existence)
            validated_path = self._validate_checkpoint_path(checkpoint_path, check_exists=True)
            
            logger.info(f"Chargement checkpoint: {validated_path}")
            
            checkpoint = torch.load(validated_path, map_location=self.device)
            
            # Restaurer l'état
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
            
            logger.info(f"Checkpoint chargé - Epoch {checkpoint['epoch']}, Best Acc: {self.best_val_acc:.2f}%")
            
            return checkpoint['epoch']
            
        except Exception as e:
            logger.error(f"Erreur chargement checkpoint: {e}", exc_info=True)
            return None

# ==================================================================================
# POINT D'ENTRÉE
# ==================================================================================

async def main():
    """Test du système"""
    config = Config()
    system = TrainingSystem(config)
    
    if await system.setup():
        await system.train(epochs=2)

if __name__ == '__main__':
    # Créer le dossier logs
    Path('logs').mkdir(exist_ok=True)
    
    # Lancer
    asyncio.run(main())
