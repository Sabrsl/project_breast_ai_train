# 🧠 BreastAI Production System v3.2.0

## Système de Classification Médicale de Grade Clinique pour la Détection du Cancer du Sein

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table des Matières

- [Caractéristiques Principales](#-caractéristiques-principales)
- [Architecture du Système](#-architecture-du-système)
- [Installation](#-installation)
- [Structure des Données](#-structure-des-données)
- [Utilisation](#-utilisation)
- [Configuration](#-configuration)
- [Modèles Supportés](#-modèles-supportés)
- [Interface Web](#-interface-web)
- [API WebSocket](#-api-websocket)
- [Performance et Optimisation](#-performance-et-optimisation)
- [Troubleshooting](#-troubleshooting)

---

## 🌟 Caractéristiques Principales

### Architecture de Deep Learning Avancée
- **EfficientNetV2** (S, M, L) - Recommandé pour usage clinique
- **EfficientNet** (B0-B7) - Maximum précision avec B7
- **ResNet, DenseNet, ConvNeXt** - Architectures alternatives
- **CBAM** (Convolutional Block Attention Module) intégré
- **Attention Pooling** pour améliorer la précision

### Prétraitement Médical Professionnel
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
- Augmentation de données médicales adaptée
- Normalisation ImageNet standard
- Support images 512x512 pour précision clinique maximale
- Détection automatique des classes (insensible à la casse)

### Entraînement Production-Ready
- **CPU optimisé** : < 16 GB RAM requis
- **Gestion automatique du déséquilibre des classes**
- **WeightedRandomSampler** pour échantillonnage équilibré
- **CosineAnnealingWarmRestarts** scheduler
- **Gradient clipping** et **Label smoothing**
- **Early stopping** configurable
- **Checkpoints automatiques** avec validation d'intégrité (SHA256)

### Interface Web en Temps Réel
- Dashboard moderne et réactif
- Communication **WebSocket** temps réel
- Éditeur de code intégré (ACE Editor)
- Visualisations **Chart.js** des métriques
- Gestion complète des checkpoints
- Export ONNX/PyTorch intégré
- Système de backup automatique

### Métriques Cliniques Avancées
- Accuracy, Precision, Recall, F1-Score (macro & weighted)
- **Sensitivity** et **Specificity** par classe
- **Cohen's Kappa** pour l'accord inter-observateurs
- **Matrice de confusion** détaillée
- **AUC-ROC** pour évaluation de la discrimination
- Métriques par classe pour analyse détaillée

---

## 🏗️ Architecture du Système

```
project_breast_ai/
│
├── breastai_complete.py      # Module d'entraînement principal
├── server_aligned.py          # Serveur WebSocket production
├── config.json                # Configuration centralisée
│
├── frontend/
│   └── index.html             # Interface web complète
│
├── data/
│   ├── train/                 # Données d'entraînement
│   │   ├── benign/           # Images bénignes
│   │   ├── malignant/        # Images malignes
│   │   └── normal/           # Images normales (optionnel)
│   ├── val/                   # Données de validation (optionnel)
│   └── test/                  # Données de test (optionnel)
│
├── checkpoints/               # Modèles sauvegardés
├── exports/
│   └── onnx/                  # Exports ONNX
├── logs/                      # Fichiers de log
└── backups/                   # Sauvegardes automatiques
```

---

## 📦 Installation

### 1. Prérequis

```bash
Python 3.9+
Torch 2.0+
16 GB RAM minimum (CPU)
Windows/Linux/MacOS
```

### 2. Installation des Dépendances

```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Installer les dépendances
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas scikit-learn matplotlib seaborn
pip install pillow opencv-python tqdm psutil
pip install websockets asyncio
```

### 3. Vérification de l'Installation

```bash
python breastai_complete.py
```

---

## 📁 Structure des Données

### Format Requis

Le système détecte **automatiquement** les classes à partir de la structure des dossiers :

```
data/
├── train/
│   ├── benign/       ou   begin/   ou   Benign/
│   │   ├── img001.jpg
│   │   ├── img002.png
│   │   └── ...
│   ├── malignant/    ou   Malignant/
│   │   ├── img001.jpg
│   │   └── ...
│   └── normal/       (optionnel)
│       └── ...
│
├── val/              (optionnel - sinon split automatique 80/20)
│   ├── benign/
│   └── malignant/
│
└── test/             (optionnel)
    ├── benign/
    └── malignant/
```

### Formats d'Images Supportés
- `.jpg`, `.jpeg` (recommandé)
- `.png`
- `.bmp`, `.tiff`, `.tif`

### Recommandations
- **Minimum**: 100 images par classe pour l'entraînement
- **Recommandé**: 500+ images par classe
- **Optimal**: 1000+ images par classe
- **Résolution**: Le système redimensionne automatiquement en 512x512
- **Équilibrage**: Le système gère automatiquement le déséquilibre des classes

---

## 🚀 Utilisation

### Option 1 : Interface Web (Recommandé)

1. **Démarrer le serveur WebSocket** :
```bash
python server_aligned.py
```

2. **Ouvrir l'interface web** :
```bash
# Ouvrir frontend/index.html dans un navigateur
```

3. **Workflow dans l'interface** :
   - Cliquer sur "Se connecter"
   - Configurer les paramètres dans l'onglet "Configuration Avancée"
   - Cliquer sur "Démarrer Entraînement"
   - Surveiller en temps réel dans le "Dashboard"
   - Exporter le modèle au format ONNX

### Option 2 : Ligne de Commande

```python
import asyncio
from breastai_complete import (
    MedicalProductionConfig,
    BreastAITrainingSystem,
    TrainingBroadcaster,
    TrainingStateManager
)

async def main():
    # Configuration
    config = MedicalProductionConfig("config.json")
    
    # Système d'entraînement
    broadcaster = TrainingBroadcaster()
    await broadcaster.connect()
    
    state_manager = TrainingStateManager(broadcaster)
    training_system = BreastAITrainingSystem(config, state_manager)
    
    # Configurer et entraîner
    await training_system.setup_training()
    await training_system.start_training(epochs=50)
    
    # Exporter
    await training_system.export_model("onnx")
    
    await broadcaster.close()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## ⚙️ Configuration

### Paramètres Principaux (config.json)

```json
{
  "data": {
    "image_size": 512,           // Taille images (512x512 recommandé)
    "batch_size": 4,             // 4-8 pour CPU
    "num_workers": 4,            // Threads de chargement
    "val_split": 0.2             // Split validation si pas de dossier val/
  },
  
  "model": {
    "architecture": "efficientnetv2_s",  // Modèle à utiliser
    "num_classes": 3,                     // Nombre de classes
    "dropout_rate": 0.4,                  // Dropout pour régularisation
    "use_cbam": true,                     // Activer CBAM
    "cbam_reduction": 16                  // Ratio de réduction CBAM
  },
  
  "training": {
    "epochs": 50,                         // Nombre d'epochs
    "learning_rate": 0.0003,              // Taux d'apprentissage
    "weight_decay": 0.001,                // Régularisation L2
    "optimizer": "adamw",                 // Optimiseur
    "scheduler": "cosine",                // Scheduler LR
    "gradient_clip": 1.0,                 // Gradient clipping
    "label_smoothing": 0.1,               // Label smoothing
    
    "early_stopping": {
      "patience": 10,                     // Patience early stopping
      "min_delta": 0.001,                 // Delta minimum
      "monitor": "val_f1_weighted"        // Métrique à surveiller
    }
  },
  
  "system": {
    "num_threads": 8,                     // Threads PyTorch
    "seed": 42                            // Seed pour reproductibilité
  }
}
```

---

## 🤖 Modèles Supportés

### Recommandations par Cas d'Usage

| Cas d'Usage | Modèle Recommandé | Précision | Vitesse | RAM |
|-------------|-------------------|-----------|---------|-----|
| **Prototype rapide** | efficientnet_b0 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 4GB |
| **Production équilibrée** | efficientnetv2_s | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 8GB |
| **Haute précision** | efficientnetv2_m | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 12GB |
| **Maximum précision** | efficientnet_b7 | ⭐⭐⭐⭐⭐ | ⭐⭐ | 16GB |
| **Recherche clinique** | efficientnetv2_l | ⭐⭐⭐⭐⭐ | ⭐⭐ | 16GB |

### Liste Complète

**EfficientNetV2** (Recommandé) :
- `efficientnetv2_s` ⭐ (Rapide et précis)
- `efficientnetv2_m` ⭐⭐ (Très performant)
- `efficientnetv2_l` ⭐⭐⭐ (Ultra précis)

**EfficientNet B0-B7** :
- `efficientnet_b0` à `efficientnet_b7`
- B4 : Équilibré
- B5-B7 : Maximum précision

**Autres Architectures** :
- `resnet50`, `resnet101` (Classiques)
- `densenet121`, `densenet169` (Connexions denses)
- `convnext_tiny`, `convnext_small`, `convnext_base` (Modernes)

---

## 🌐 Interface Web

### Fonctionnalités

1. **Connexion Serveur**
   - Connexion/déconnexion WebSocket
   - Indicateur de statut temps réel

2. **Contrôle d'Entraînement**
   - Démarrer/Arrêter entraînement
   - Barre de progression en temps réel
   - Métriques live (Loss, Accuracy, F1-Score)

3. **Configuration Avancée**
   - Paramètres modèle
   - Paramètres optimiseur
   - Configuration CBAM
   - Augmentation de données

4. **Dashboard Temps Réel**
   - Graphiques Chart.js des métriques
   - Visualisation de l'évolution de l'entraînement

5. **Gestion Checkpoints**
   - Liste des checkpoints
   - Chargement/Suppression
   - Reprise d'entraînement

6. **Fonctions Avancées**
   - Export ONNX/PyTorch
   - Rapport clinique
   - Diagnostics système
   - Sauvegarde projet complète

---

## 📡 API WebSocket

### Messages Supportés

#### Client → Serveur

```javascript
// Démarrer l'entraînement
{
  "type": "start_training",
  "config": {...},
  "epochs": 50,
  "session_id": "default"
}

// Arrêter l'entraînement
{
  "type": "stop_training",
  "session_id": "default"
}

// Export modèle
{
  "type": "export_model",
  "format": "onnx",
  "session_id": "default"
}

// Diagnostics système
{
  "type": "system_diagnostics"
}

// Lister checkpoints
{
  "type": "list_checkpoints"
}

// Créer backup
{
  "type": "create_backup"
}
```

#### Serveur → Client

```javascript
// Mise à jour entraînement
{
  "type": "training_update",
  "epoch": 10,
  "train_loss": 0.45,
  "train_accuracy": 0.89,
  "val_loss": 0.48,
  "val_accuracy": 0.87,
  "val_f1_weighted": 0.86,
  "learning_rate": 0.0002
}

// Entraînement terminé
{
  "type": "training_complete",
  "final_metrics": {...},
  "history": {...}
}

// Export terminé
{
  "type": "export_complete",
  "format": "onnx",
  "path": "exports/onnx/model.onnx",
  "size_mb": 45.2
}

// Erreur
{
  "type": "error",
  "message": "Error description"
}
```

---

## ⚡ Performance et Optimisation

### Optimisations CPU

Le système est **optimisé pour CPU** avec :
- `torch.set_num_threads()` configuré automatiquement
- `num_workers` ajusté pour éviter la surcharge
- `pin_memory=False` pour CPU
- Batch size réduit (4-8 recommandé)
- Gradient accumulation possible

### Gestion Mémoire

- **Images 512x512** : ~4GB RAM minimum
- **Batch size 4** : ~8GB RAM requis
- **Batch size 8** : ~12GB RAM requis
- **Nettoyage automatique** : `gc.collect()` périodique
- **Checkpoints compressés** : Économie d'espace disque

### Accélération Possible

```json
{
  "data": {
    "num_workers": 8,         // Plus de workers si CPU puissant
    "pin_memory": false       // Garder false pour CPU
  },
  "training": {
    "use_amp": false,         // AMP non recommandé pour CPU
    "gradient_clip": 1.0      // Peut être augmenté si stable
  }
}
```

---

## 🔧 Troubleshooting

### Problème : "No class directories found"

**Solution** : Vérifiez la structure de vos dossiers de données. Le dossier `data/train/` doit contenir des sous-dossiers pour chaque classe.

```bash
data/train/
├── benign/
└── malignant/
```

### Problème : "Out of Memory"

**Solutions** :
1. Réduire le `batch_size` dans config.json (essayer 2 ou 1)
2. Réduire `num_workers` à 2 ou 0
3. Choisir un modèle plus léger (efficientnet_b0)
4. Réduire `image_size` à 384 ou 256

### Problème : "WebSocket connection failed"

**Solutions** :
1. Vérifier que `server_aligned.py` est bien lancé
2. Vérifier le port 8765 n'est pas utilisé
3. Désactiver le pare-feu/antivirus temporairement
4. Utiliser `localhost` au lieu de `127.0.0.1` ou vice-versa

### Problème : Entraînement très lent

**Solutions** :
1. Réduire `image_size` à 384 ou 256
2. Utiliser un modèle plus léger
3. Augmenter `num_threads` dans config.json
4. Vérifier que rien d'autre n'utilise le CPU intensivement

### Problème : Métriques ne s'affichent pas

**Solutions** :
1. Vérifier la connexion WebSocket (indicateur vert)
2. Ouvrir la console du navigateur (F12) pour voir les erreurs
3. Rafraîchir la page web
4. Redémarrer le serveur

### Problème : Classes mal détectées

**Solution** : Le système normalise les noms en minuscules. "Benign", "benign", "BENIGN" sont tous reconnus comme "benign". Vérifiez qu'il n'y a pas de dossiers cachés ou de fichiers système dans les dossiers de classes.

---

## 📊 Exemple de Résultats Attendus

### Avec 1000 images par classe (benign, malignant)

| Métrique | Valeur Attendue |
|----------|----------------|
| **Accuracy** | 92-96% |
| **F1-Score (weighted)** | 0.91-0.95 |
| **Sensitivity** | 90-95% |
| **Specificity** | 92-97% |
| **AUC-ROC** | 0.95-0.98 |

### Temps d'Entraînement (50 epochs)

| Configuration | Temps Estimé |
|--------------|--------------|
| CPU i5, 8GB RAM, batch=4 | ~6-8 heures |
| CPU i7, 16GB RAM, batch=8 | ~4-5 heures |
| CPU Ryzen 9, 32GB RAM, batch=16 | ~2-3 heures |

---

## 📝 Licence

MIT License - Voir LICENSE pour plus de détails

---

## 🤝 Support

Pour toute question ou problème :
1. Vérifier ce README
2. Consulter les logs dans `logs/`
3. Vérifier la configuration dans `config.json`

---

## 🔬 Citations

Si vous utilisez ce système dans vos recherches, veuillez citer :

```
BreastAI Production System v3.2.0
Système de Classification Médicale pour la Détection du Cancer du Sein
2024
```

---

## 🎯 Roadmap

- [ ] Support GPU automatique
- [ ] Quantization INT8 pour déploiement edge
- [ ] Interface Gradio/Streamlit alternative
- [ ] API REST en plus du WebSocket
- [ ] Support multi-GPU distribué
- [ ] Fine-tuning automatique des hyperparamètres
- [ ] Explainability avec GradCAM
- [ ] Validation croisée K-fold intégrée

---

**Développé avec ❤️ pour la recherche médicale et le diagnostic clinique**

Version 3.2.0 - Dernière mise à jour : Septembre 2024
