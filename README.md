# 🧠 BreastAI Studio v3.3

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Production](https://img.shields.io/badge/Status-Production-green.svg)]()

> 🏥 **Système d'intelligence artificielle de pointe pour le diagnostic du cancer du sein**  
> Architecture: **EfficientNetV2 + CBAM** | Interface Web Moderne | Entraînement en Temps Réel

---

## ✨ Fonctionnalités Principales

### 🎯 Capacités IA
- **Architecture avancée** : EfficientNetV2 (S/M/L) avec module d'attention CBAM
- **Classification 3 classes** : Normal, Bénin, Malin
- **Entraînement optimisé** : Mixed Precision, Gradient Accumulation, Label Smoothing
- **Augmentation données** : Rotations, flips, ajustements colorimétriques
- **Schedulers avancés** : Cosine Annealing, ReduceLROnPlateau, OneCycle

### 🖥️ Interface Web Moderne
- ⚡ **Temps réel** : WebSocket pour suivi live de l'entraînement
- 📊 **Dashboard interactif** : Métriques, graphiques, progression par batch
- 💻 **Console logs** : Affichage complet de tous les événements
- 💾 **Gestion checkpoints** : Reprise, export ONNX, suppression
- ⚙️ **Configuration flexible** : Tous les hyperparamètres ajustables

### 🚀 Production Ready
- 📦 **Export ONNX** : Déploiement multi-plateforme
- 🔄 **Resume training** : Reprise depuis n'importe quel checkpoint
- 📝 **Logging complet** : Logs fichiers + console + interface
- 🛡️ **Gestion erreurs** : Skip des batchs corrompus automatique
- 🔓 **Progressive Unfreezing** : Optimisation CPU (×3-4 plus rapide au début)

---

## 📋 Prérequis

```bash
Python 3.10+
PyTorch 2.0+
CUDA 11.8+ (recommandé pour GPU)
8GB RAM minimum (16GB recommandé)
```

---

## 🚀 Installation Rapide

### 1️⃣ Cloner le projet
```bash
git clone https://github.com/votre-username/project_breast_ai.git
cd project_breast_ai
```

### 2️⃣ Créer l'environnement virtuel
```bash
python -m venv venv_breastai
```

**Windows:**
```bash
venv_breastai\Scripts\activate
```

**Linux/Mac:**
```bash
source venv_breastai/bin/activate
```

### 3️⃣ Installer les dépendances
```bash
pip install -r requirements.txt
```

---

## 📊 Structure des Données

Le projet attend une structure de données spécifique :

```
data/
├── train/
│   ├── normal/      # Images normales
│   ├── begin/       # Tumeurs bénignes
│   └── malignant/   # Tumeurs malignes
├── val/
│   ├── benign/
│   └── malignant/
└── test/
    ├── Benign/
    └── Malignant/
```

> ⚠️ **Note**: Les données ne sont PAS incluses dans ce repo (trop volumineuses).  
> Placez vos datasets dans le dossier `data/` selon la structure ci-dessus.

---

## 🎮 Utilisation

### Mode Interface Web (Recommandé)

#### 1️⃣ Démarrer le serveur WebSocket
```bash
python server_simple.py
```

#### 2️⃣ Ouvrir l'interface
Ouvrez `frontend/app.html` dans votre navigateur

#### 3️⃣ Utiliser l'interface
1. **Connecter** au serveur
2. **Configurer** les hyperparamètres
3. **Démarrer** l'entraînement
4. **Suivre** en temps réel dans le dashboard et la console

### Mode CLI (Avancé)

```python
from breastai_training import TrainingSystem, Config

# Configuration
config = Config({
    'model': {
        'architecture': 'efficientnetv2_s',
        'num_classes': 3,
        'use_cbam': True
    },
    'training': {
        'epochs': 50,
        'learning_rate': 0.0003,
        'optimizer': 'adamw'
    }
})

# Lancer l'entraînement
system = TrainingSystem(config)
await system.setup()
await system.train(epochs=50)
```

---

## 📈 Fonctionnalités Détaillées

### 🎛️ Hyperparamètres Configurables

| Paramètre | Options | Description |
|-----------|---------|-------------|
| **Modèle** | efficientnetv2_s/m/l, efficientnet_b4 | Architecture du réseau |
| **Epochs** | 1-200 | Nombre d'époques |
| **Batch Size** | 1-32 | Taille des batchs |
| **Learning Rate** | 0.00001-0.01 | Taux d'apprentissage |
| **Optimizer** | AdamW, Adam, SGD | Optimiseur |
| **Scheduler** | Cosine, ReduceLR, OneCycle | Scheduler LR |
| **CBAM** | Activé/Désactivé | Module d'attention |
| **Dropout** | 0.0-0.8 | Taux de dropout |

### 📊 Métriques Suivies

- **Loss** : Train & Validation
- **Accuracy** : Train & Validation  
- **F1-Score** : Macro & Weighted
- **Learning Rate** : Évolution en temps réel
- **Batch Progress** : Détails par batch

### 💾 Checkpoints

- **Automatique** : Sauvegarde du meilleur modèle
- **Périodique** : Tous les 10 epochs
- **Métadonnées** : Epoch, accuracy, config complète
- **Reprise** : Continue depuis n'importe quel checkpoint

### 📦 Export ONNX

```python
# Via l'interface : Bouton "Export ONNX"
# Ou en CLI :
await system.export_onnx('checkpoints/best_model.pth')
```

---

## 🏗️ Architecture Technique

```
┌─────────────────────────────────────────────────────────┐
│                   Frontend (app.html)                    │
│              Interface Web Moderne + WebSocket           │
└───────────────────────────┬─────────────────────────────┘
                            │ ws://localhost:8765
┌───────────────────────────▼─────────────────────────────┐
│              Server WebSocket (server_simple.py)         │
│          Gestion connexions + Broadcast messages         │
└───────────────────────────┬─────────────────────────────┘
                            │ async/await
┌───────────────────────────▼─────────────────────────────┐
│         Training System (breastai_training.py)           │
│   ┌─────────────────────────────────────────────────┐   │
│   │  EfficientNetV2 + CBAM                          │   │
│   │  ├── Feature Extraction (Pretrained)            │   │
│   │  ├── CBAM Attention Module                      │   │
│   │  └── Classification Head (3 classes)            │   │
│   └─────────────────────────────────────────────────┘   │
│                                                           │
│   • DataLoaders (Train/Val/Test)                        │
│   • Optimizer + Scheduler                                │
│   • Mixed Precision Training                             │
│   • Checkpoint Management                                │
│   • ONNX Export                                          │
└───────────────────────────────────────────────────────────┘
```

---

## 📁 Structure du Projet

```
project_breast_ai/
├── 🧠 CORE
│   ├── breastai_training.py    # Système d'entraînement
│   ├── server_simple.py        # Serveur WebSocket
│   └── inference_onnx.py       # Inférence ONNX
│
├── 🎨 FRONTEND
│   ├── app.html               # Interface web v3.3
│   └── requirements.txt
│
├── 📁 DATA (non inclus)
│   ├── train/
│   ├── val/
│   └── test/
│
├── 💾 OUTPUTS (généré)
│   ├── checkpoints/           # Modèles sauvegardés
│   ├── exports/onnx/          # Exports ONNX
│   └── logs/                  # Logs d'entraînement
│
├── 📚 DOCUMENTATION
│   ├── QUICK_START_FR.md      # Démarrage rapide
│   ├── GUIDE_INTERFACE_FR.md  # Guide interface
│   └── CHANGELOG_FR.md        # Historique
│
└── ⚙️ CONFIG
    ├── config.json            # Configuration par défaut
    ├── requirements.txt       # Dépendances Python
    └── .gitignore            # Exclusions Git
```

---

## ⚡ Progressive Unfreezing (Optimisation CPU)

**Nouvelle fonctionnalité v3.3.1** : Accélération massive de l'entraînement sur CPU !

### 📊 Comment ça marche ?

Le **Progressive Unfreezing** gèle progressivement les couches du backbone pré-entraîné pour accélérer drastiquement l'entraînement, surtout sur CPU.

```
🔒 Phase 1 (Epochs 1-5)   : Backbone gelé → Classifier seul
                            ✅ ×3-4 plus rapide
                            📉 ~5% des paramètres entraînés

🔓 Phase 2 (Epochs 6-15)  : Dégel partiel → 3 derniers blocs
                            ✅ ×2 plus rapide
                            📉 ~30% des paramètres entraînés

🔥 Phase 3 (Epochs 16+)   : Dégel complet → Tous les paramètres
                            ⚙️ Vitesse normale
                            📈 100% des paramètres entraînés
```

### 💡 Gain estimé

| Configuration | Sans PU | Avec PU | Gain |
|---------------|---------|---------|------|
| **CPU (53k images)** | ~220 jours | ~70-90 jours | **×2.5-3** |
| **GPU (53k images)** | ~2-3 jours | ~1.5-2 jours | **×1.3-1.5** |

> **Note** : Le Progressive Unfreezing est activé **automatiquement** ! Aucune configuration nécessaire.

---

## 🎯 Performance

### Configuration Recommandée

| Composant | Recommandé | Minimum |
|-----------|-----------|---------|
| **GPU** | NVIDIA RTX 3060+ (12GB) | GTX 1660 (6GB) |
| **RAM** | 16GB | 8GB |
| **CPU** | 8 cores | 4 cores |
| **Storage** | SSD 50GB | HDD 50GB |

### Benchmarks (Exemple)

- **EfficientNetV2-S** : ~95% Val Accuracy, ~30min/epoch (RTX 3060)
- **EfficientNetV2-M** : ~96% Val Accuracy, ~45min/epoch (RTX 3060)
- **EfficientNetV2-L** : ~97% Val Accuracy, ~60min/epoch (RTX 3060)

> 📊 Résultats dépendent fortement de votre dataset

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Fork le projet
2. Créez une branche (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

---

## 📝 Changelog

### v3.3.1 (Actuel - 2024-09-30)
- 🔓 **Progressive Unfreezing** : Accélération massive pour CPU (×2.5-3)
- 🎮 **Auto-détection GPU** : Bascule automatique GPU/CPU
- ⚡ **Logs temps réel** : Affichage à chaque batch
- 📊 **Statistiques détaillées** : Comptage paramètres entraînables
- 🐛 **Fix** : Correction device CPU forcé

### v3.3.0 (2024-09-29)
- 🎨 **Interface moderne** : Dashboard amélioré
- 🔧 **Optimisations** : Performance et stabilité
- 🐛 **Corrections** : Gestion erreurs améliorée

Voir [CHANGELOG_FR.md](CHANGELOG_FR.md) pour l'historique complet.

---

## 📄 License

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 🙏 Remerciements

- **PyTorch** pour le framework deep learning
- **EfficientNet** pour l'architecture de base
- **CBAM** pour le module d'attention
- La communauté open-source

---

## 📧 Contact & Support

Pour toute question ou support :
- 🐛 **Issues** : [Sabrsl](https://github.com/votre-username/project_breast_ai/issues)

---

<div align="center">

**Développé avec ❤️ pour améliorer le diagnostic du cancer du sein**

⭐ Si ce projet vous est utile, donnez-lui une étoile !

</div>
