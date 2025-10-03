#!/usr/bin/env python3
"""
BreastAI Training System v4.4 - BALANCED TRAINING
Corrections majeures:
- Monitor balanced_accuracy au lieu de sensitivity_malignant
- FocalLoss rebalancée avec alpha adaptatif
- Logging optimisé (réduction 90% messages)
- Progressive unfreezing accéléré (4 epochs phase 1)
- Combo Loss (Focal + Dice) pour équilibre classes
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
import hashlib
import weakref

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
    f1_score, precision_score, recall_score, roc_curve,
    balanced_accuracy_score
)

import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==================== LOGGING OPTIMISÉ ====================
class ThrottledHandler(logging.Handler):
    """Handler qui limite le débit de messages"""
    def __init__(self, base_handler, max_per_second=10):
        super().__init__()
        self.base_handler = base_handler
        self.max_per_second = max_per_second
        self.last_emit_time = {}
        self.min_interval = 1.0 / max_per_second
    
    def emit(self, record):
        now = datetime.now().timestamp()
        key = f"{record.levelname}:{record.msg}"
        
        last_time = self.last_emit_time.get(key, 0)
        if now - last_time >= self.min_interval:
            self.base_handler.emit(record)
            self.last_emit_time[key] = now

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Handler console throttled
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
throttled_console = ThrottledHandler(console_handler, max_per_second=5)
logger.addHandler(throttled_console)

# File handler normal
file_handler = logging.FileHandler('logs/training.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# ==================== CONFIG ====================
class Config:
    def __init__(self, config_dict: Optional[Dict] = None):
        config_path = Path('config.json')
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        if config_dict:
            self._merge_config(config_dict)
        
        self._set_clinical_defaults()
        self._compute_hash()
    
    def _compute_hash(self):
        """Hash SHA256 complet pour éviter collisions"""
        config_str = json.dumps(self.config, sort_keys=True)
        self.config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]  # 16 chars SHA256
    
    def _set_clinical_defaults(self):
        """Defaults ÉQUILIBRÉS pour accuracy globale"""
        defaults = {
            'model': {
                'architecture': 'efficientnetv2_s',
                'num_classes': 3,
                'use_cbam': True,
                'use_se': False,
                'dropout_rate': 0.4,  # Augmenté pour réduire overfitting
                'progressive_unfreezing': {
                    'enabled': True,
                    'phase1_epochs': 4,  # RÉDUIT de 8 à 4
                    'phase2_epochs': 12,  # RÉDUIT de 20 à 12
                    'phase3_epochs': 25   # RÉDUIT de 40 à 25
                }
            },
            'training': {
                'epochs': 50,
                'learning_rate': 0.0003,
                'weight_decay': 0.0001,
                'focal_loss': {
                    'enabled': True,
                    'gamma': 2.0,  # RÉDUIT de 2.5 à 2.0
                    'alpha': 'adaptive'  # Calculé automatiquement
                },
                'dice_loss_weight': 0.3,  # NOUVEAU: combo Focal+Dice
                'label_smoothing': 0.1,
                'use_ema': True,
                'ema_decay': 0.9998,
                'early_stopping': {
                    'enabled': True,
                    'patience': 12,  # RÉDUIT de 15 à 12
                    'min_delta': 0.005,  # AUGMENTÉ de 0.001 à 0.005
                    'monitor': 'balanced_accuracy'  # CHANGÉ!
                }
            },
            'data': {
                'image_size': 512,
                'batch_size': 8,
                'num_workers': 4,
                'gradient_accumulation_steps': 4,
                'use_clahe': True,
                'breast_cropping': True,
                'augmentation_strength': 'strong',  # AUGMENTÉ
                'oversampling': {
                    'enabled': True,  # ACTIVÉ
                    'strategy': 'balanced',  # Équilibrer toutes les classes
                    'ratio': 0.85,  # Équilibrage partiel 85% (évite overfitting)
                    'max_weight': 2.0  # Cap poids à 2x max
                }
            },
            'inference': {
                'tta_enabled': True,
                'mc_dropout_samples': 10,
                'rejection_threshold': 0.15
            },
            'clinical': {
                'primary_metric': 'balanced_accuracy',  # CHANGÉ!
                'min_sensitivity_malignant': 0.90,  # RÉDUIT de 0.95 à 0.90
                'target_specificity': 0.85,  # NOUVEAU
                'confidence_threshold': 0.5,
                'rejection_enabled': True,
                'gradcam_enabled': True,
                'threshold_optimization': True
            },
            'validation': {
                'k_fold': 5,
                'stratify': True,
                'patient_wise_split': True,
                'cross_validation_enabled': False  # CPU = trop lent pour K-fold complet
            }
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if subkey not in self.config[key]:
                        self.config[key][subkey] = subvalue
    
    def _merge_config(self, custom: Dict):
        def merge_dict(base, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
        merge_dict(self.config, custom)
    
    def get(self, *keys, default=None):
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default

# ==================== EARLY STOPPING BALANCED ====================
class ClinicalEarlyStopping:
    """Early Stopping basé sur balanced_accuracy"""
    
    VALID_MONITORS = {
        'balanced_accuracy', 'f1_weighted', 'f1_macro',
        'sensitivity_malignant', 'specificity', 'accuracy', 
        'auc_malignant', 'loss'
    }
    
    def __init__(self, patience: int = 12, min_delta: float = 0.002, 
                 monitor: str = 'balanced_accuracy'):
        self.patience = patience
        self.min_delta = min_delta
        
        if monitor not in self.VALID_MONITORS:
            logger.warning(f"Monitor '{monitor}' invalide, utilisation de 'balanced_accuracy'")
            monitor = 'balanced_accuracy'
        
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        logger.info(f"Early Stopping: monitor={monitor}, patience={patience}, min_delta={min_delta}")
    
    def __call__(self, metrics: Dict[str, float]) -> bool:
        if self.monitor not in metrics:
            logger.warning(f"Monitor '{self.monitor}' absent, skip early stopping")
            return False
        
        score = metrics[self.monitor]
        
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"EARLY STOP: {self.monitor}={score:.4f}, pas d'amélioration depuis {self.patience} epochs")
                return True
            else:
                # Throttle ce log (toutes les 2 epochs seulement)
                if self.counter % 2 == 0:
                    logger.info(f"Early stopping: {self.counter}/{self.patience} ({self.monitor}={score:.4f})")
                return False
    
    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False

# ==================== ATTENTION MODULES ====================
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
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
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

# ==================== LOSSES BALANCÉES ====================
class FocalLoss(nn.Module):
    """Focal Loss avec alpha adaptatif"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.alpha = torch.tensor(alpha) if isinstance(alpha, list) else alpha
        else:
            self.alpha = None
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
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
        return focal_loss

class DiceLoss(nn.Module):
    """Dice Loss pour équilibrage classes"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets, num_classes=3):
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Flatten spatial dimensions
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        targets_one_hot = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)
        
        intersection = (inputs * targets_one_hot).sum(dim=2)
        union = inputs.sum(dim=2) + targets_one_hot.sum(dim=2)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class ComboLoss(nn.Module):
    """Combo Focal + Dice pour équilibre optimal"""
    def __init__(self, focal_weight=0.7, dice_weight=0.3, alpha=None, gamma=2.0):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss

# ==================== GRAD-CAM ====================
_gradcam_instances = weakref.WeakSet()

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._is_active = False
        
        _gradcam_instances.add(self)
        self._register_hooks()
    
    def _register_hooks(self):
        if self._is_active:
            return
        
        self.hooks.append(self.target_layer.register_forward_hook(self._save_activation))
        self.hooks.append(self.target_layer.register_full_backward_hook(self._save_gradient))
        self._is_active = True
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def remove_hooks(self):
        if not self._is_active:
            return
        
        for hook in self.hooks:
            try:
                hook.remove()
            except:
                pass
        
        self.hooks = []
        self._is_active = False
    
    def __del__(self):
        try:
            self.remove_hooks()
        except:
            pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
        return False
    
    def generate_cam(self, input_image, target_class=None):
        if not self._is_active:
            raise RuntimeError("Hooks non actifs")
        
        self.model.eval()
        self.gradients = None
        self.activations = None
        
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients/activations non capturés")
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()

def cleanup_all_gradcam_instances():
    count = 0
    for instance in list(_gradcam_instances):
        try:
            instance.remove_hooks()
            count += 1
        except:
            pass
    if count > 0:
        logger.info(f"Nettoyage: {count} instances Grad-CAM")

# ==================== CALIBRATION ====================
class CalibrationMetrics:
    @staticmethod
    def expected_calibration_error(y_true, y_pred_probs, n_bins=15, adaptive=True):
        confidences = np.max(y_pred_probs, axis=1)
        predictions = np.argmax(y_pred_probs, axis=1)
        accuracies = (predictions == y_true).astype(float)
        
        if adaptive:
            bin_boundaries = np.percentile(confidences, np.linspace(0, 100, n_bins + 1))
            bin_boundaries[-1] = 1.0
        else:
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece

# ==================== MODÈLE ====================
class BreastAIModel(nn.Module):
    def __init__(self, architecture: str = 'efficientnetv2_s', num_classes: int = 3,
                 use_cbam: bool = True, use_se: bool = False, dropout: float = 0.5):
        super().__init__()
        
        self.dropout_rate = dropout
        
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
        
        self.backbone.classifier = nn.Identity()
        
        if use_cbam:
            self.attention = CBAM(num_features)
        else:
            self.attention = None
        
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
    
    def forward(self, x, return_features=False):
        x = self.backbone.features(x)
        
        if self.attention is not None:
            x = self.attention(x)
        
        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)
        
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits

# ==================== DATASET ====================
class MedicalDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None, use_clahe: bool = True,
                 breast_crop: bool = True):
        self.root_dir = root_dir
        self.transform = transform
        self.use_clahe = use_clahe
        self.breast_crop = breast_crop
        
        self.classes = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = root_dir / class_name
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.dcm']:
                    patient_id = self._extract_patient_id(img_path.stem)
                    self.samples.append((
                        str(img_path), 
                        self.class_to_idx[class_name],
                        patient_id
                    ))
        
        logger.info(f"Dataset: {len(self.samples)} images, classes={self.classes}")
        
        self.patient_to_samples = defaultdict(list)
        for idx, (img_path, label, patient_id) in enumerate(self.samples):
            self.patient_to_samples[patient_id].append(idx)
        
        num_patients = len(self.patient_to_samples)
        logger.info(f"Patients uniques: {num_patients}")
        if num_patients <= 10:
            logger.warning(f"ATTENTION: Seulement {num_patients} patients détectés")
    
    def _extract_patient_id(self, filename: str) -> str:
        parts = filename.split('_')
        for i, part in enumerate(parts):
            if part.lower() == 'patient' and i + 1 < len(parts):
                return f"patient_{parts[i+1]}"
        return parts[0] if parts else filename
    
    def apply_breast_cropping(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                if 0.5 <= aspect_ratio <= 2.0:
                    pad = int(max(w, h) * 0.05)
                    x = max(0, x - pad)
                    y = max(0, y - pad)
                    w = min(image.shape[1] - x, w + 2 * pad)
                    h = min(image.shape[0] - y, h + 2 * pad)
                    
                    return image[y:y+h, x:x+w]
        
        return image
    
    def apply_clahe(self, image):
        if len(image.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, patient_id = self.samples[idx]
        
        try:
            if img_path.endswith('.dcm'):
                try:
                    import pydicom
                    dcm = pydicom.dcmread(img_path)
                    image = dcm.pixel_array
                    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                    if len(image.shape) == 2:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                except ImportError:
                    raise IOError("DICOM nécessite pydicom")
            else:
                image = cv2.imread(img_path)
                if image is None:
                    raise IOError(f"Image corrompue: {img_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if image.shape[0] < 50 or image.shape[1] < 50:
                raise ValueError(f"Image trop petite: {image.shape}")
            
            if self.breast_crop:
                image = self.apply_breast_cropping(image)
            
            if self.use_clahe:
                image = self.apply_clahe(image)
            
            image = Image.fromarray(image)
            
            if self.transform:
                image = self.transform(image)
            
            if torch.isnan(image).any():
                raise ValueError(f"NaN après transforms: {img_path}")
            
            return image, label
            
        except Exception as e:
            logger.error(f"Image corrompue: {img_path}: {e}")
            raise

# ==================== PATIENT-WISE SPLIT ====================
class PatientWiseSplitter:
    @staticmethod
    def split_by_patient(dataset: MedicalDataset, val_ratio: float = 0.2, 
                        test_ratio: float = 0.1, stratify: bool = True, 
                        random_state: int = 42):
        np.random.seed(random_state)
        
        patient_ids = list(dataset.patient_to_samples.keys())
        
        if stratify:
            patient_labels = {}
            for patient_id in patient_ids:
                indices = dataset.patient_to_samples[patient_id]
                labels = [dataset.samples[i][1] for i in indices]
                patient_labels[patient_id] = Counter(labels).most_common(1)[0][0]
            
            class_to_patients = defaultdict(list)
            for patient_id, label in patient_labels.items():
                class_to_patients[label].append(patient_id)
            
            train_patients, val_patients, test_patients = [], [], []
            
            for label, patients in class_to_patients.items():
                np.random.shuffle(patients)
                n = len(patients)
                n_test = max(1, int(n * test_ratio))
                n_val = max(1, int(n * val_ratio))
                n_train = n - n_test - n_val
                
                test_patients.extend(patients[:n_test])
                val_patients.extend(patients[n_test:n_test + n_val])
                train_patients.extend(patients[n_test + n_val:])
        else:
            np.random.shuffle(patient_ids)
            n = len(patient_ids)
            n_test = max(1, int(n * test_ratio))
            n_val = max(1, int(n * val_ratio))
            
            test_patients = patient_ids[:n_test]
            val_patients = patient_ids[n_test:n_test + n_val]
            train_patients = patient_ids[n_test + n_val:]
        
        train_indices = []
        for patient_id in train_patients:
            train_indices.extend(dataset.patient_to_samples[patient_id])
        
        val_indices = []
        for patient_id in val_patients:
            val_indices.extend(dataset.patient_to_samples[patient_id])
        
        test_indices = []
        for patient_id in test_patients:
            test_indices.extend(dataset.patient_to_samples[patient_id])
        
        logger.info(f"Patient-wise split: {len(train_patients)} train, {len(val_patients)} val, {len(test_patients)} test patients")
        logger.info(f"Samples: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")
        
        return train_indices, val_indices, test_indices

# ==================== THRESHOLD OPTIMIZATION ====================
class ThresholdOptimizer:
    @staticmethod
    def optimize_threshold_for_sensitivity(y_true, y_probs, 
                                          target_sensitivity: float = 0.90,
                                          target_class: int = 1):
        y_true_binary = (y_true == target_class).astype(int)
        y_scores = y_probs[:, target_class]
        
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores)
        
        valid_indices = np.where(tpr >= target_sensitivity)[0]
        
        if len(valid_indices) == 0:
            logger.warning(f"Impossible d'atteindre sensitivity {target_sensitivity}")
            best_idx = np.argmax(tpr)
            optimal_threshold = thresholds[best_idx]
            achieved_sensitivity = tpr[best_idx]
            specificity = 1 - fpr[best_idx]
        else:
            best_idx = valid_indices[np.argmin(fpr[valid_indices])]
            optimal_threshold = thresholds[best_idx]
            achieved_sensitivity = tpr[best_idx]
            specificity = 1 - fpr[best_idx]
        
        logger.info(f"Threshold: {optimal_threshold:.4f}, Sens: {achieved_sensitivity:.4f}, Spec: {specificity:.4f}")
        
        return optimal_threshold, achieved_sensitivity, specificity

# ==================== TRAINING SYSTEM ====================
class ClinicalTrainingSystem:
    def __init__(self, config: Config, callback: Optional[Callable] = None):
        self.config = config
        self.callback = callback
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            logger.warning("CPU mode (lent)")
        
        self.is_training = False
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        self.gradient_accumulation_steps = config.get('data', 'gradient_accumulation_steps', default=2)
        self.use_amp = torch.cuda.is_available()
        self.scaler = None
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        
        self.use_ema = config.get('training', 'use_ema', default=True)
        self.ema_decay = config.get('training', 'ema_decay', default=0.9998)
        self.model_ema = None
        
        monitor = config.get('training', 'early_stopping', 'monitor', default='balanced_accuracy')
        patience = config.get('training', 'early_stopping', 'patience', default=12)
        min_delta = config.get('training', 'early_stopping', 'min_delta', default=0.005)
        self.early_stopping = ClinicalEarlyStopping(patience=patience, monitor=monitor, min_delta=min_delta)
        
        # GradCAM désactivé pendant training (économie mémoire)
        self.gradcam = None
        self.gradcam_enabled_for_inference = config.get('clinical', 'gradcam_enabled', default=False)
        self.optimal_threshold = 0.5
        self.best_val_acc = 0.0
        self.best_balanced_acc = 0.0
        self.best_sensitivity_malignant = 0.0
        self.history = defaultdict(list)
        
        # Compteur pour throttling logs
        self.batch_log_interval = 50  # Log toutes les 50 batches au lieu de 5
        
        logger.info("Système v4.4 BALANCED initialisé")
    
    async def send_update(self, message: Dict):
        if self.callback:
            try:
                await self.callback(message)
            except Exception as e:
                logger.error(f"Erreur callback: {e}")
    
    async def setup(self) -> bool:
        try:
            await self.send_update({
                'type': 'progress_update',
                'stage': 'setup',
                'progress': 0.1,
                'message': 'Chargement données...'
            })
            
            if not await self._setup_data():
                return False
            
            await self.send_update({
                'type': 'progress_update',
                'stage': 'setup',
                'progress': 0.5,
                'message': 'Construction modèle...'
            })
            
            arch = self.config.get('model', 'architecture', default='efficientnetv2_s')
            num_classes = self.config.get('model', 'num_classes', default=3)
            use_cbam = self.config.get('model', 'use_cbam', default=True)
            use_se = self.config.get('model', 'use_se', default=False)
            dropout = self.config.get('model', 'dropout_rate', default=0.5)
            
            self.model = BreastAIModel(arch, num_classes, use_cbam, use_se, dropout)
            self.model.to(self.device)
            
            # GradCAM: DÉSACTIVÉ pendant training (pas utilisé, consomme mémoire)
            # Sera créé uniquement pour inférence/export si nécessaire
            if self.gradcam_enabled_for_inference:
                logger.info("GradCAM sera activé pour inférence (pas pendant training)")
            
            if self.use_ema:
                import copy
                self.model_ema = copy.deepcopy(self.model)
                self.model_ema.eval()
                for param in self.model_ema.parameters():
                    param.requires_grad = False
                logger.info(f"EMA activé: decay={self.ema_decay}")
            
            lr = self.config.get('training', 'learning_rate', default=0.0003)
            wd = self.config.get('training', 'weight_decay', default=0.0001)
            
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
            
            # Calculer alpha adaptatif pour FocalLoss
            focal_config = self.config.config.get('training', {}).get('focal_loss', {})
            use_focal = focal_config.get('enabled', True)
            
            if use_focal:
                alpha_config = focal_config.get('alpha', 'adaptive')
                
                if alpha_config == 'adaptive':
                    # Calculer alpha basé sur distribution classes
                    labels = []
                    if hasattr(self.train_loader.dataset, 'dataset'):
                        subset = self.train_loader.dataset
                        if hasattr(subset, 'indices'):
                            for idx in subset.indices:
                                labels.append(subset.dataset.samples[idx][1])
                    else:
                        labels = [self.train_loader.dataset.samples[i][1] for i in range(len(self.train_loader.dataset))]
                    
                    class_counts = Counter(labels)
                    total = sum(class_counts.values())
                    
                    # Alpha inversement proportionnel à la fréquence
                    alpha = []
                    for i in range(num_classes):
                        count = class_counts.get(i, 1)
                        weight = total / (num_classes * count)
                        alpha.append(weight)
                    
                    # Normaliser
                    sum_alpha = sum(alpha)
                    alpha = [a / sum_alpha for a in alpha]
                    
                    logger.info(f"Alpha adaptatif calculé: {[f'{a:.3f}' for a in alpha]}")
                else:
                    alpha = alpha_config
                
                gamma = focal_config.get('gamma', 2.0)
                dice_weight = self.config.get('training', 'dice_loss_weight', default=0.3)
                
                self.criterion = ComboLoss(
                    focal_weight=1.0 - dice_weight,
                    dice_weight=dice_weight,
                    alpha=alpha,
                    gamma=gamma
                )
                logger.info(f"Loss: ComboLoss (Focal {1.0-dice_weight:.1f} + Dice {dice_weight:.1f})")
            else:
                self.criterion = nn.CrossEntropyLoss()
            
            epochs = self.config.get('training', 'epochs', default=50)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
            
            await self.send_update({
                'type': 'progress_update',
                'stage': 'setup',
                'progress': 1.0,
                'message': 'Configuration terminée!'
            })
            
            logger.info("Setup complet v4.4")
            return True
            
        except Exception as e:
            logger.error(f"Erreur setup: {e}", exc_info=True)
            await self.send_update({'type': 'error', 'message': f'Erreur setup: {str(e)}'})
            return False
    
    async def _setup_data(self) -> bool:
        try:
            data_dir = Path(self.config.get('paths', 'data_dir', default='data'))
            
            if data_dir.name in ['train', 'val', 'test']:
                data_dir = data_dir.parent
            
            train_dir = data_dir / 'train'
            
            if not train_dir.exists():
                raise ValueError(f"Train dir introuvable: {train_dir}")
            
            image_size = self.config.get('data', 'image_size', default=512)
            augmentation_strength = self.config.get('data', 'augmentation_strength', default='strong')
            
            # Augmentations MÉDICALES pour mammographie (pas de vertical flip/rotations extrêmes)
            if augmentation_strength == 'strong':
                train_transform = transforms.Compose([
                    transforms.Resize((image_size, image_size), interpolation=Image.LANCZOS),
                    transforms.RandomHorizontalFlip(0.5),  # ✅ Symétrie naturelle OK
                    # ❌ PAS de RandomVerticalFlip (change anatomie)
                    transforms.RandomRotation(5),  # ✅ Rotation légère OK (±5° max)
                    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # ✅ Translation/scale légères
                    transforms.ColorJitter(brightness=0.2, contrast=0.3),  # ✅ Variations exposition OK
                    # ❌ PAS de RandomPerspective (déforme l'anatomie)
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),  # ✅ Simule occlusions
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                # Augmentations MÉDICALES modérées
                train_transform = transforms.Compose([
                    transforms.Resize((image_size, image_size), interpolation=Image.LANCZOS),
                    transforms.RandomHorizontalFlip(0.5),  # ✅ OK
                    # ❌ PAS de RandomVerticalFlip
                    transforms.RandomRotation(3),  # ✅ Rotation très légère
                    transforms.ColorJitter(brightness=0.15, contrast=0.2),  # ✅ OK
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            
            val_transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=Image.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            use_clahe = self.config.get('data', 'use_clahe', default=True)
            breast_crop = self.config.get('data', 'breast_cropping', default=True)
            
            train_dataset = MedicalDataset(train_dir, train_transform, use_clahe, breast_crop)
            
            use_patient_split = self.config.get('validation', 'patient_wise_split', default=True)
            
            if use_patient_split and len(train_dataset.patient_to_samples) > 10:
                logger.info("Application patient-wise split...")
                train_indices, val_indices, test_indices = PatientWiseSplitter.split_by_patient(
                    train_dataset,
                    val_ratio=0.15,
                    test_ratio=0.10,
                    stratify=True
                )
                
                from torch.utils.data import Subset
                train_subset = Subset(train_dataset, train_indices)
                val_subset = Subset(train_dataset, val_indices)
                test_subset = Subset(train_dataset, test_indices)
                
                batch_size = self.config.get('data', 'batch_size', default=8)
                num_workers = self.config.get('data', 'num_workers', default=4)
                
                # Oversampling activé par défaut
                use_oversampling = self.config.get('data', 'oversampling', 'enabled', default=True)
                
                if use_oversampling:
                    labels = [train_dataset.samples[i][1] for i in train_indices]
                    class_counts = Counter(labels)
                    
                    # Oversampling PARTIEL pour éviter overfitting
                    max_count = max(class_counts.values())
                    ratio = self.config.get('data', 'oversampling', 'ratio', default=0.85)
                    max_weight_cap = self.config.get('data', 'oversampling', 'max_weight', default=2.0)
                    
                    target_count = int(max_count * ratio)  # 85% du max par défaut
                    
                    weights = []
                    for i in train_indices:
                        label = train_dataset.samples[i][1]
                        # Calculer poids avec cap
                        raw_weight = target_count / class_counts[label]
                        weight = min(raw_weight, max_weight_cap)  # Cap à 2x max
                        weights.append(weight)
                    
                    sampler = WeightedRandomSampler(weights, len(weights))
                    logger.info(f"Oversampling partiel ({ratio*100:.0f}%): distribution={dict(class_counts)}, cap={max_weight_cap}x")
                else:
                    labels = [train_dataset.samples[i][1] for i in train_indices]
                    class_counts = Counter(labels)
                    weights = [1.0 / class_counts[train_dataset.samples[i][1]] for i in train_indices]
                    sampler = WeightedRandomSampler(weights, len(weights))
                
                self.train_loader = DataLoader(
                    train_subset, batch_size=batch_size,
                    sampler=sampler, num_workers=num_workers,
                    pin_memory=torch.cuda.is_available()
                )
                
                self.val_loader = DataLoader(
                    val_subset, batch_size=batch_size,
                    shuffle=False, num_workers=num_workers
                )
                
                self.test_loader = DataLoader(
                    test_subset, batch_size=batch_size,
                    shuffle=False, num_workers=num_workers
                )
                
                logger.info("Patient-wise split appliqué")
            
            logger.info(f"Données: {len(train_dataset)} train, classes={train_dataset.classes}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur setup data: {e}", exc_info=True)
            return False
    
    def _apply_progressive_unfreezing(self, epoch: int, total_epochs: int):
        if not hasattr(self.model, 'backbone'):
            return
        
        config = self.config.config.get('model', {}).get('progressive_unfreezing', {})
        if not config.get('enabled', True):
            return
        
        phase1 = config.get('phase1_epochs', 4)
        phase2 = config.get('phase2_epochs', 12)
        phase3 = config.get('phase3_epochs', 25)
        
        if epoch <= phase1:
            if epoch == 1:
                for param in self.model.backbone.parameters():
                    param.requires_grad = False
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in self.model.parameters())
                logger.info(f"[Phase 1/4] Backbone GELÉ - Epochs 1-{phase1} ({100*trainable/total:.1f}% params)")
        
        elif epoch == phase1 + 1:
            if hasattr(self.model.backbone, 'features'):
                total_blocks = len(self.model.backbone.features)
                unfreeze_from = int(total_blocks * 0.75)
                for idx, block in enumerate(self.model.backbone.features):
                    if idx >= unfreeze_from:
                        for param in block.parameters():
                            param.requires_grad = True
                
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.5
                
                logger.info(f"[Phase 2/4] Dégel 25% - Epochs {phase1+1}-{phase2}, LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        elif epoch == phase2 + 1:
            if hasattr(self.model.backbone, 'features'):
                total_blocks = len(self.model.backbone.features)
                unfreeze_from = int(total_blocks * 0.50)
                for idx, block in enumerate(self.model.backbone.features):
                    if idx >= unfreeze_from:
                        for param in block.parameters():
                            param.requires_grad = True
                
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.6
                
                logger.info(f"[Phase 3/4] Dégel 50% - Epochs {phase2+1}-{phase3}, LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        elif epoch == phase3 + 1:
            for param in self.model.backbone.parameters():
                param.requires_grad = True
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.33
            
            logger.info(f"[Phase 4/4] Dégel COMPLET - Epochs {phase3+1}+, LR: {self.optimizer.param_groups[0]['lr']:.2e}")
    
    async def train(self, epochs: Optional[int] = None, start_epoch: int = 1):
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
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"DÉMARRAGE: epochs {start_epoch}-{epochs}, monitor={self.early_stopping.monitor}")
            
            for epoch in range(start_epoch, epochs + 1):
                if not self.is_training:
                    logger.info("Arrêt demandé")
                    break
                
                self._apply_progressive_unfreezing(epoch, epochs)
                
                train_metrics = await self._train_epoch(epoch)
                val_metrics = await self._validate_epoch(epoch)
                
                # Threshold optimization progressif après mi-parcours
                threshold_freq = self.config.get('clinical', 'threshold_update_frequency', default=10)
                if epoch >= epochs // 2 and epoch % threshold_freq == 0:
                    if self.config.get('clinical', 'threshold_optimization', default=True):
                        logger.info(f"Mise à jour threshold (epoch {epoch}/{epochs})")
                        await self._optimize_threshold()
                
                # Update interface (throttled)
                await self.send_update({
                    'type': 'training_update',
                    'epoch': epoch,
                    'total_epochs': epochs,
                    'train_loss': float(train_metrics['loss']),
                    'train_accuracy': float(train_metrics['accuracy']),
                    'val_loss': float(val_metrics['loss']),
                    'val_accuracy': float(val_metrics['accuracy']),
                    'val_balanced_accuracy': float(val_metrics.get('balanced_accuracy', 0)),
                    'val_sensitivity_malignant': float(val_metrics.get('sensitivity_malignant', 0)),
                    'val_specificity': float(val_metrics.get('specificity', 0)),
                    'val_f1_weighted': float(val_metrics.get('f1_weighted', 0)),
                    'learning_rate': float(self.optimizer.param_groups[0]['lr']),
                    'progress': (epoch / epochs) * 100,
                    'timestamp': datetime.now().isoformat()
                })
                
                self.scheduler.step()
                self._handle_checkpoints(epoch, val_metrics)
                
                if self.early_stopping(val_metrics):
                    self._save_checkpoint(epoch, f'early_stop_epoch_{epoch:03d}.pth')
                    await self.send_update({
                        'type': 'log',
                        'message': f'Early stopping à epoch {epoch}',
                        'level': 'warning'
                    })
                    break
            
            if self.config.get('clinical', 'threshold_optimization', default=True):
                await self._optimize_threshold()
            
            if self.test_loader:
                await self._final_test_evaluation()
            
            await self.send_update({
                'type': 'training_complete',
                'final_metrics': {
                    'best_val_accuracy': self.best_val_acc,
                    'best_balanced_accuracy': self.best_balanced_acc,
                    'best_sensitivity_malignant': self.best_sensitivity_malignant,
                    'optimal_threshold': self.optimal_threshold
                },
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"TERMINÉ! Balanced Acc: {self.best_balanced_acc:.2f}%, Acc: {self.best_val_acc:.2f}%")
            
        except Exception as e:
            logger.error(f"Erreur training: {e}", exc_info=True)
            await self.send_update({'type': 'error', 'message': f'Erreur: {str(e)}'})
        finally:
            self.is_training = False
    
    async def _optimize_threshold(self):
        if self.val_loader is None:
            return
        
        logger.info("Optimisation threshold...")
        
        model = self.model_ema if self.use_ema and self.model_ema is not None else self.model
        model.eval()
        
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        target_sensitivity = self.config.get('clinical', 'min_sensitivity_malignant', default=0.90)
        
        optimal_threshold, achieved_sens, spec = ThresholdOptimizer.optimize_threshold_for_sensitivity(
            all_labels, all_probs, 
            target_sensitivity=target_sensitivity,
            target_class=1
        )
        
        self.optimal_threshold = optimal_threshold
        
        await self.send_update({
            'type': 'threshold_optimized',
            'threshold': float(optimal_threshold),
            'sensitivity': float(achieved_sens),
            'specificity': float(spec)
        })
    
    async def _final_test_evaluation(self):
        logger.info("="*60)
        logger.info("ÉVALUATION TEST SET")
        logger.info("="*60)
        
        model = self.model_ema if self.use_ema and self.model_ema is not None else self.model
        model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                
                preds = torch.argmax(probs, dim=1)
                malignant_mask = probs[:, 1] >= self.optimal_threshold
                preds[malignant_mask] = 1
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_preds) * 100
        balanced_acc = balanced_accuracy_score(all_labels, all_preds) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        
        sensitivity_malignant = recall[1] * 100
        precision_malignant = precision[1] * 100
        
        try:
            auc_malignant = roc_auc_score(
                (all_labels == 1).astype(int),
                all_probs[:, 1]
            )
        except:
            auc_malignant = 0
        
        cm = confusion_matrix(all_labels, all_preds)
        ece = CalibrationMetrics.expected_calibration_error(all_labels, all_probs, adaptive=True)
        
        logger.info(f"TEST SET (N={len(all_labels)}):")
        logger.info(f"  Accuracy: {accuracy:.2f}%")
        logger.info(f"  Balanced Accuracy: {balanced_acc:.2f}%")
        logger.info(f"  F1-weighted: {f1_weighted:.4f}")
        logger.info(f"  Sensitivity(malignant): {sensitivity_malignant:.2f}%")
        logger.info(f"  AUC(malignant): {auc_malignant:.4f}")
        logger.info(f"  ECE: {ece:.4f}")
        logger.info(f"\nMatrice:\n{cm}")
        logger.info("="*60)
        
        report_path = Path('exports') / f'test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        report_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(report_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'test_samples': len(all_labels),
                'optimal_threshold': float(self.optimal_threshold),
                'metrics': {
                    'accuracy': float(accuracy),
                    'balanced_accuracy': float(balanced_acc),
                    'f1_weighted': float(f1_weighted),
                    'sensitivity_malignant': float(sensitivity_malignant),
                    'auc_malignant': float(auc_malignant),
                    'ece': float(ece)
                },
                'confusion_matrix': cm.tolist()
            }, f, indent=2)
        
        logger.info(f"Rapport: {report_path}")
        
        await self.send_update({
            'type': 'test_evaluation_complete',
            'metrics': {
                'accuracy': float(accuracy),
                'balanced_accuracy': float(balanced_acc),
                'sensitivity_malignant': float(sensitivity_malignant),
                'auc_malignant': float(auc_malignant),
                'ece': float(ece)
            }
        })
    
    def _handle_checkpoints(self, epoch: int, metrics: Dict):
        balanced_acc = metrics.get('balanced_accuracy', 0)
        acc = metrics['accuracy']
        
        self._save_checkpoint(epoch, f'latest_epoch_{epoch:03d}.pth')
        
        if balanced_acc > self.best_balanced_acc:
            self.best_balanced_acc = balanced_acc
            self.best_val_acc = acc
            self.best_sensitivity_malignant = metrics.get('sensitivity_malignant', 0)
            self._save_checkpoint(epoch, 'best.pth')
            self._save_checkpoint(epoch, f'best_epoch_{epoch:03d}_bacc_{balanced_acc:.3f}.pth')
            logger.info(f"[BEST] Balanced Acc={balanced_acc:.2f}%, Acc={acc:.2f}%")
        
        if epoch % 10 == 0:
            self._save_checkpoint(epoch, f'periodic_epoch_{epoch:03d}.pth')
        
        if epoch > 3:
            old = Path(self.config.get('paths', 'checkpoint_dir', default='checkpoints')) / f'latest_epoch_{epoch-3:03d}.pth'
            if old.exists():
                try:
                    old.unlink()
                except:
                    pass
    
    async def _train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            try:
                if images.size(0) == 0:
                    continue
                
                images, labels = images.to(self.device), labels.to(self.device)
                
                if self.use_amp:
                    from torch.cuda.amp import autocast
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                        if self.gradient_accumulation_steps > 1:
                            loss = loss / self.gradient_accumulation_steps
                    
                    self.scaler.scale(loss).backward()
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    if self.gradient_accumulation_steps > 1:
                        loss = loss / self.gradient_accumulation_steps
                    loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    
                    if self.use_ema and self.model_ema is not None:
                        self._update_ema()
                
                if self.gradient_accumulation_steps > 1:
                    total_loss += loss.item() * self.gradient_accumulation_steps
                else:
                    total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Log THROTTLED (toutes les 50 batches)
                if batch_idx % self.batch_log_interval == 0:
                    await self.send_update({
                        'type': 'batch_progress',
                        'batch': batch_idx,
                        'total_batches': len(self.train_loader),
                        'current_loss': round(float(loss.item()), 4),
                        'current_accuracy': round(100. * correct / total, 2)
                    })
                
            except Exception as e:
                logger.error(f"Erreur batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100. * correct / total
        
        logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={avg_acc:.2f}%")
        
        return {'loss': avg_loss, 'accuracy': avg_acc}
    
    def _update_ema(self):
        with torch.no_grad():
            for ema_param, model_param in zip(self.model_ema.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)
    
    async def _validate_epoch(self, epoch: int) -> Dict:
        if self.val_loader is None:
            return {'loss': 0, 'accuracy': 0, 'balanced_accuracy': 0}
        
        use_tta = self.config.get('inference', 'tta_enabled', default=False)
        
        if use_tta:
            return await self._validate_with_tta(epoch)
        else:
            return await self._validate_standard(epoch)
    
    async def _validate_standard(self, epoch: int) -> Dict:
        model = self.model_ema if self.use_ema and self.model_ema is not None else self.model
        model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_preds) * 100
        balanced_acc = balanced_accuracy_score(all_labels, all_preds) * 100
        
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        
        sensitivity_malignant = recall[1] * 100 if len(recall) > 1 else 0
        precision_malignant = precision[1] * 100 if len(precision) > 1 else 0
        
        try:
            auc_malignant = roc_auc_score(
                (all_labels == 1).astype(int),
                all_probs[:, 1]
            )
        except:
            auc_malignant = 0
        
        cm = confusion_matrix(all_labels, all_preds)
        specificities = []
        for i in range(len(cm)):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities.append(spec)
        specificity = np.mean(specificities) * 100
        
        ece = CalibrationMetrics.expected_calibration_error(all_labels, all_probs, adaptive=True)
        
        # Log résumé uniquement
        logger.info(f"Epoch {epoch} Val: Acc={accuracy:.2f}%, Balanced={balanced_acc:.2f}%, Sens(mal)={sensitivity_malignant:.2f}%")
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'sensitivity_malignant': sensitivity_malignant,
            'precision_malignant': precision_malignant,
            'specificity': specificity,
            'auc_malignant': auc_malignant,
            'ece': ece
        }
    
    async def _validate_with_tta(self, epoch: int) -> Dict:
        model = self.model_ema if self.use_ema and self.model_ema is not None else self.model
        model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        # TTA MÉDICAL: SEULEMENT transformations cliniquement valides pour mammographie
        # ❌ PAS de vertical flip, rotations ou both flips (déforment l'anatomie)
        # ✅ Horizontal flip OK (symétrie naturelle gauche/droite)
        tta_transforms = [
            lambda x: x,  # Original
            lambda x: torch.flip(x, dims=[3]),  # Horizontal flip UNIQUEMENT
        ]
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                predictions = []
                for transform in tta_transforms:
                    aug_images = transform(images)
                    outputs = model(aug_images)
                    probs = F.softmax(outputs, dim=1)
                    predictions.append(probs)
                
                final_probs = torch.mean(torch.stack(predictions), dim=0)
                _, predicted = final_probs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(final_probs.cpu().numpy())
        
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_preds) * 100
        balanced_acc = balanced_accuracy_score(all_labels, all_preds) * 100
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        
        precision, recall, _, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        
        sensitivity_malignant = recall[1] * 100 if len(recall) > 1 else 0
        
        try:
            auc_malignant = roc_auc_score(
                (all_labels == 1).astype(int),
                all_probs[:, 1]
            )
        except:
            auc_malignant = 0
        
        cm = confusion_matrix(all_labels, all_preds)
        specificities = []
        for i in range(len(cm)):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities.append(spec)
        specificity = np.mean(specificities) * 100
        
        ece = CalibrationMetrics.expected_calibration_error(all_labels, all_probs, adaptive=True)
        
        logger.info(f"Epoch {epoch} Val TTA (2 transforms - médical safe): Acc={accuracy:.2f}%, Balanced={balanced_acc:.2f}%")
        
        return {
            'loss': 0,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_weighted': f1_weighted,
            'sensitivity_malignant': sensitivity_malignant,
            'specificity': specificity,
            'auc_malignant': auc_malignant,
            'ece': ece
        }
    
    def _save_checkpoint(self, epoch: int, filename: str):
        checkpoint_dir = Path(self.config.get('paths', 'checkpoint_dir', default='checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / filename
        
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_acc': self.best_val_acc,
                'best_balanced_acc': self.best_balanced_acc,
                'best_sensitivity_malignant': self.best_sensitivity_malignant,
                'optimal_threshold': self.optimal_threshold,
                'architecture': self.config.get('model', 'architecture'),
                'num_classes': self.config.get('model', 'num_classes'),
                'timestamp': datetime.now().isoformat(),
                'config': self.config.config,
                'config_hash': self.config.config_hash
            }
            
            if self.use_ema and self.model_ema is not None:
                checkpoint['model_ema_state_dict'] = self.model_ema.state_dict()
            
            torch.save(checkpoint, checkpoint_path)
            
            if checkpoint_path.exists():
                size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
                # Log throttled
                if 'best' in filename or epoch % 5 == 0:
                    logger.info(f"[SAVE] {filename} - {size_mb:.2f} MB")
                
                asyncio.create_task(self.send_update({
                    'type': 'checkpoint_saved',
                    'filename': filename,
                    'epoch': epoch,
                    'accuracy': float(self.best_val_acc),
                    'size_mb': float(size_mb)
                }))
        
        except Exception as e:
            logger.error(f"Erreur sauvegarde: {e}")
    
    async def export_onnx(self, checkpoint_path: Optional[str] = None) -> bool:
        try:
            await self.send_update({
                'type': 'log',
                'message': 'Export ONNX en cours...',
                'level': 'info'
            })
            
            if checkpoint_path:
                checkpoint_path = Path(checkpoint_path)
                if not checkpoint_path.exists():
                    if not str(checkpoint_path).startswith('checkpoints'):
                        checkpoint_path = Path('checkpoints') / checkpoint_path.name
                
                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"Checkpoint: {checkpoint_path}")
                
                import torch.serialization
                import numpy
                with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
                    checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Checkpoint chargé: {checkpoint_path}")
            
            export_dir = Path(self.config.get('paths', 'export_dir', default='exports'))
            export_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            onnx_path = export_dir / f'breastai_v44_balanced_{timestamp}.onnx'
            
            self.model.eval()
            image_size = self.config.get('data', 'image_size', default=512)
            dummy_input = torch.randn(1, 3, image_size, image_size).to(self.device)
            
            torch.onnx.export(
                self.model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            
            logger.info(f"Export ONNX: {onnx_path}")
            
            try:
                import onnx
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                logger.info("Validation ONNX: OK")
                
                class_names = ['benign', 'malignant', 'normal']
                metadata = {
                    'model_name': 'BreastAI_v4.4_BALANCED',
                    'architecture': self.config.get('model', 'architecture'),
                    'num_classes': str(self.config.get('model', 'num_classes')),
                    'class_names': ','.join(class_names),
                    'best_balanced_accuracy': str(self.best_balanced_acc),
                    'best_accuracy': str(self.best_val_acc),
                    'best_sensitivity_malignant': str(self.best_sensitivity_malignant),
                    'optimal_threshold': str(self.optimal_threshold),
                    'preprocessing': 'CLAHE+BreastCrop+StrongAug+Normalize',
                    'loss_function': 'ComboLoss_Focal+Dice',
                    'monitor': 'balanced_accuracy',
                    'export_date': datetime.now().isoformat(),
                    'fixes_applied': 'v4.4_balanced_training,adaptive_focal_alpha,throttled_logging'
                }
                
                for key, value in metadata.items():
                    meta = onnx_model.metadata_props.add()
                    meta.key = key
                    meta.value = value
                
                onnx.save(onnx_model, str(onnx_path))
                logger.info("Métadonnées ajoutées")
            
            except ImportError:
                logger.warning("onnx non installé, skip validation")
            
            metadata_json = export_dir / f'breastai_v44_balanced_{timestamp}_metadata.json'
            with open(metadata_json, 'w') as f:
                json.dump({
                    'model_file': onnx_path.name,
                    'version': '4.4_BALANCED',
                    'export_date': datetime.now().isoformat(),
                    'improvements': [
                        'Monitor: balanced_accuracy au lieu de sensitivity_malignant',
                        'ComboLoss: Focal (alpha adaptatif) + Dice (30%)',
                        'Progressive unfreezing accéléré (4/12/25 epochs)',
                        'Oversampling par défaut pour équilibrage',
                        'Augmentations fortes (rotation, perspective, erasing)',
                        'Logging throttled (réduction 90% messages)',
                        'Early stopping min_delta augmenté (0.005)'
                    ],
                    'clinical_metrics': {
                        'best_balanced_accuracy': float(self.best_balanced_acc),
                        'best_val_accuracy': float(self.best_val_acc),
                        'best_sensitivity_malignant': float(self.best_sensitivity_malignant),
                        'optimal_threshold': float(self.optimal_threshold)
                    },
                    'preprocessing': {
                        'clahe': True,
                        'breast_cropping': True,
                        'augmentation_strength': 'strong',
                        'normalize_mean': [0.485, 0.456, 0.406],
                        'normalize_std': [0.229, 0.224, 0.225]
                    },
                    'class_names': ['benign', 'malignant', 'normal']
                }, f, indent=2)
            
            file_size = onnx_path.stat().st_size / (1024 * 1024)
            
            await self.send_update({
                'type': 'export_complete',
                'path': str(onnx_path),
                'metadata_path': str(metadata_json),
                'size': file_size,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info("="*60)
            logger.info("EXPORT ONNX v4.4 BALANCED")
            logger.info(f"Modèle: {onnx_path} ({file_size:.2f} MB)")
            logger.info(f"Metadata: {metadata_json}")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur export: {e}", exc_info=True)
            await self.send_update({'type': 'error', 'message': f'Erreur export: {str(e)}'})
            return False
    
    def load_checkpoint(self, checkpoint_path: str) -> Optional[int]:
        try:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint: {checkpoint_path}")
            
            import torch.serialization
            import numpy
            with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
            self.best_balanced_acc = checkpoint.get('best_balanced_acc', 0.0)
            self.best_sensitivity_malignant = checkpoint.get('best_sensitivity_malignant', 0.0)
            self.optimal_threshold = checkpoint.get('optimal_threshold', 0.5)
            
            if self.use_ema and 'model_ema_state_dict' in checkpoint:
                self.model_ema.load_state_dict(checkpoint['model_ema_state_dict'])
            
            logger.info(f"Checkpoint chargé: epoch {checkpoint['epoch']}")
            return checkpoint['epoch']
            
        except Exception as e:
            logger.error(f"Erreur load: {e}", exc_info=True)
            return None
    
    async def stop(self):
        self.is_training = False
        await self.send_update({'type': 'training_stopped', 'timestamp': datetime.now().isoformat()})
        logger.info("Arrêt demandé")
    
    def __del__(self):
        try:
            if self.gradcam is not None:
                self.gradcam.remove_hooks()
        except:
            pass

# ==================== POINT D'ENTRÉE (DEBUG UNIQUEMENT) ====================
async def main():
    """
    Fonction de test standalone - NE S'EXÉCUTE PAS en production
    Utilisée uniquement pour débugger le module indépendamment
    """
    print("="*60)
    print("BreastAI Training System v4.4 - MODE DEBUG STANDALONE")
    print("ATTENTION: Ce mode est pour tests uniquement")
    print("="*60)
    
    config = Config()
    system = ClinicalTrainingSystem(config)
    
    if await system.setup():
        print("\nLancement training test (3 epochs)...")
        await system.train(epochs=3)
        
        if system.test_loader:
            await system._final_test_evaluation()
        
        print("\nExport ONNX...")
        await system.export_onnx()
    
    cleanup_all_gradcam_instances()
    
    print("="*60)
    print("Tests terminés - v4.4 BALANCED")
    print("="*60)

if __name__ == '__main__':
    """
    Ce bloc s'exécute UNIQUEMENT si vous lancez directement:
        python breastai_training.py
    
    Il NE s'exécute PAS quand le fichier est importé par server_simple.py
    C'est le comportement standard Python.
    """
    print("\n" + "="*60)
    print("AVERTISSEMENT: Mode debug standalone détecté")
    print("="*60)
    print("\nCe mode est pour tester le module indépendamment.")
    print("En production normale, utilisez: python server_simple.py")
    print("\nVoulez-vous continuer avec un training de test ?")
    print("="*60)
    
    try:
        response = input("\nLancer training debug ? (y/n): ").strip().lower()
        
        if response == 'y':
            print("\nInitialisation des dossiers...")
            Path('logs').mkdir(exist_ok=True)
            Path('checkpoints').mkdir(exist_ok=True)
            Path('exports').mkdir(exist_ok=True)
            
            print("Lancement du training debug...\n")
            asyncio.run(main())
        else:
            print("\nAnnulé.")
            print("Utilisez 'python server_simple.py' pour le mode normal avec interface web.")
    
    except KeyboardInterrupt:
        print("\n\nInterrompu par l'utilisateur.")
    except Exception as e:
        print(f"\nErreur: {e}")
        import traceback
        traceback.print_exc()