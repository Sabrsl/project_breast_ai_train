# 🧠 BreastAI Training Studio

Application complète pour l'entraînement de modèles d'IA de détection du cancer du sein.

## 🚀 Installation

### 1. Installer Python 3.8+
```bash
python --version
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Préparer les données
Organisez vos données dans cette structure :
```
data/breast_cancer/
├── train/
│   ├── benign/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── malignant/
│       ├── image3.jpg
│       └── image4.jpg
└── val/
    ├── benign/
    │   ├── image5.jpg
    │   └── image6.jpg
    └── malignant/
        ├── image7.jpg
        └── image8.jpg
```

## 🎯 Utilisation

### 1. Démarrer le serveur Python
```bash
python breastai_server.py
```

### 2. Ouvrir l'interface web
Ouvrez `breastai_training_studio.html` dans votre navigateur.

### 3. Configurer l'entraînement
- **Dataset Path** : Chemin vers vos données
- **Epochs** : Nombre d'époques (défaut: 100)
- **Batch Size** : Taille des lots (défaut: 32)
- **Learning Rate** : Taux d'apprentissage (défaut: 0.001)
- **Model Type** : Type de modèle (ResNet18, ResNet50, DenseNet121, EfficientNet)

### 4. Démarrer l'entraînement
- Cliquez sur **🚀 Démarrer Entraînement**
- Suivez les métriques en temps réel
- Visualisez les graphiques de progression

## 📊 Fonctionnalités

### Interface Moderne
- Design dark professionnel
- Navigation par onglets fluide
- Interface responsive (mobile/tablette)

### Éditeur de Code Intégré
- Éditeur Ace avec coloration syntaxique Python
- Template de code complet fourni
- Raccourcis clavier (Ctrl+S, F5)
- Sauvegarde automatique

### Dashboard Temps Réel
- 4 graphiques Chart.js (Loss, Accuracy, Learning Rate, Memory)
- Métriques temps réel pendant l'entraînement
- Console de logs avec couleurs
- Barres de progression animées

### Configuration Complète
- Tous les hyperparamètres modifiables
- Gestion de dataset (scan, préparation)
- Augmentations configurables
- Export ONNX intégré

### Fonctionnalités Avancées
- Communication WebSocket temps réel
- Gestion d'erreurs robuste
- Sauvegarde intelligente
- Export de modèles
- Support GPU/CPU automatique

## 🔧 Modèles Supportés

- **ResNet18** : Rapide, bon pour débuter
- **ResNet50** : Plus précis, plus lent
- **DenseNet121** : Efficace en mémoire
- **EfficientNet** : Optimal performance/coût

## 📁 Structure du Projet

```
breastai_training_studio/
├── breastai_training_studio.html  # Interface web
├── breastai_server.py             # Serveur Python
├── requirements.txt               # Dépendances
├── README.md                      # Documentation
└── data/breast_cancer/            # Données d'entraînement
    ├── train/
    └── val/
```

## 🎮 Raccourcis Clavier

- **Ctrl+S** : Sauvegarder le code
- **F5** : Démarrer l'entraînement
- **Ctrl+O** : Charger un fichier

## 🚨 Dépannage

### Erreur de connexion WebSocket
- Vérifiez que le serveur Python est démarré
- Vérifiez le port 5000 (pas de conflit)

### Erreur de données
- Vérifiez la structure des dossiers
- Vérifiez les formats d'images (JPG, PNG)

### Erreur GPU
- Vérifiez l'installation de CUDA
- Vérifiez les drivers GPU

## 📈 Métriques Suivies

- **Training Loss** : Perte d'entraînement
- **Validation Loss** : Perte de validation
- **Training Accuracy** : Précision d'entraînement
- **Validation Accuracy** : Précision de validation
- **Learning Rate** : Taux d'apprentissage
- **GPU Memory** : Utilisation mémoire GPU

## 🔄 Workflow

1. **Préparation** : Organisez vos données
2. **Configuration** : Ajustez les hyperparamètres
3. **Entraînement** : Lancez l'entraînement
4. **Monitoring** : Suivez les métriques
5. **Export** : Sauvegardez le modèle
6. **Déploiement** : Utilisez le modèle entraîné

## 🎯 Résultats

L'application génère :
- **Modèle PyTorch** : `breastai_model.pth`
- **Modèle ONNX** : `breastai_model.onnx`
- **Métriques** : Graphiques et logs détaillés
- **Code** : Script d'entraînement sauvegardé

## 🆘 Support

Pour toute question ou problème :
1. Vérifiez les logs dans la console
2. Consultez la documentation PyTorch
3. Vérifiez la compatibilité des versions

---

**BreastAI Training Studio** - Solution complète pour l'entraînement de modèles d'IA de détection du cancer du sein.
