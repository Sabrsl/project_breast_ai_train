# 🚀 Guide de Démarrage Rapide - BreastAI v3.2.0

## En 5 minutes chrono ! ⏱️

---

## 📋 Étape 1 : Installation (2 min)

### Windows

```bash
# Ouvrir PowerShell ou CMD

# Créer environnement virtuel
python -m venv venv

# Activer l'environnement
venv\Scripts\activate

# Installer les dépendances
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Linux/Mac

```bash
# Ouvrir Terminal

# Créer environnement virtuel
python3 -m venv venv

# Activer l'environnement
source venv/bin/activate

# Installer les dépendances
pip install torch torchvision
pip install -r requirements.txt
```

---

## 📁 Étape 2 : Préparer vos Données (1 min)

Organisez vos images comme ceci :

```
data/
└── train/
    ├── benign/          # Images bénignes
    │   ├── img001.jpg
    │   ├── img002.jpg
    │   └── ...
    └── malignant/       # Images malignes
        ├── img001.jpg
        ├── img002.jpg
        └── ...
```

**💡 Notes importantes :**
- Minimum 100 images par classe (recommandé 500+)
- Formats supportés : `.jpg`, `.png`, `.bmp`
- Le système redimensionne automatiquement en 512x512

---

## ✅ Étape 3 : Validation (30 sec)

Vérifiez que tout est correctement installé :

```bash
python validate_system.py
```

Vous devriez voir des ✅ partout !

---

## 🚀 Étape 4 : Lancer le Système (30 sec)

### Option A : Interface Web (Recommandé) 🌐

1. **Démarrer le serveur** :
```bash
python server_aligned.py
```

Vous verrez :
```
==============================================================================
BREASTAI PRODUCTION WEBSOCKET SERVER v3.2.0 - READY
Listening on ws://localhost:8765
==============================================================================
```

2. **Ouvrir l'interface** :
   - Double-cliquez sur `frontend/index.html`
   - OU ouvrez-le avec votre navigateur (Chrome, Firefox, Edge)

3. **Connecter et démarrer** :
   - Cliquez sur "🔌 Se connecter"
   - L'indicateur devient vert ✅
   - Cliquez sur "🚀 Démarrer Entraînement"

### Option B : Ligne de Commande 💻

```python
# Créer un fichier run_training.py

import asyncio
from breastai_complete import (
    MedicalProductionConfig,
    BreastAITrainingSystem,
    TrainingBroadcaster,
    TrainingStateManager
)

async def main():
    config = MedicalProductionConfig("config.json")
    broadcaster = TrainingBroadcaster()
    await broadcaster.connect()
    
    state_manager = TrainingStateManager(broadcaster)
    training_system = BreastAITrainingSystem(config, state_manager)
    
    await training_system.setup_training()
    await training_system.start_training(epochs=50)
    await training_system.export_model("onnx")
    
    await broadcaster.close()

asyncio.run(main())
```

Puis lancez :
```bash
python run_training.py
```

---

## 📊 Étape 5 : Surveiller l'Entraînement (en temps réel)

### Dans l'Interface Web

1. **Onglet "Dashboard Temps Réel"** :
   - Graphiques de Loss et Accuracy
   - Évolution des métriques par epoch

2. **Section "Métriques en Temps Réel"** (barre latérale) :
   - Epoch actuel
   - Loss en cours
   - Accuracy
   - F1-Score

3. **Onglet "Console de Logs"** :
   - Logs détaillés de l'entraînement
   - Messages d'erreur si problème

---

## 💾 Étape 6 : Exporter votre Modèle

### Via l'Interface Web

1. Attendez la fin de l'entraînement
2. Cliquez sur "📤 Export ONNX"
3. Le fichier sera dans `exports/onnx/`

### Via la Ligne de Commande

```bash
# Le modèle ONNX est déjà exporté automatiquement
# Trouvez-le dans : exports/onnx/breastai_model_YYYYMMDD_HHMMSS.onnx
```

---

## 🎯 Configuration Rapide

### Pour Tests Rapides (Prototype)

Éditez `config.json` :

```json
{
  "model": {
    "architecture": "efficientnet_b0",  // Modèle léger
    "num_classes": 2                     // 2 classes
  },
  "data": {
    "image_size": 224,                   // Taille réduite
    "batch_size": 8                      // Batch plus grand
  },
  "training": {
    "epochs": 10,                        // Peu d'epochs pour test
    "learning_rate": 0.001
  }
}
```

### Pour Production Clinique (Recommandé)

```json
{
  "model": {
    "architecture": "efficientnetv2_s",  // Modèle performant
    "num_classes": 3                     // benign, malignant, normal
  },
  "data": {
    "image_size": 512,                   // Haute résolution
    "batch_size": 4                      // Optimisé CPU
  },
  "training": {
    "epochs": 50,                        // Entraînement complet
    "learning_rate": 0.0003
  }
}
```

### Pour Maximum Précision

```json
{
  "model": {
    "architecture": "efficientnet_b7",   // Maximum précision
    "num_classes": 3
  },
  "data": {
    "image_size": 512,
    "batch_size": 2                      // Batch réduit (gros modèle)
  },
  "training": {
    "epochs": 100,                       // Beaucoup d'epochs
    "learning_rate": 0.0001              // LR plus faible
  }
}
```

---

## ⚡ Astuces pour Accélérer

1. **Réduire la taille des images** :
```json
"image_size": 384  // Au lieu de 512
```

2. **Augmenter le batch size** si vous avez de la RAM :
```json
"batch_size": 8    // Au lieu de 4
```

3. **Choisir un modèle plus léger** :
```json
"architecture": "efficientnet_b0"  // Au lieu de efficientnetv2_s
```

4. **Réduire les workers** si CPU limité :
```json
"num_workers": 2   // Au lieu de 4
```

---

## 🔍 Vérification des Résultats

### Résultats Typiques (1000 images/classe)

Après 50 epochs, vous devriez obtenir :

| Métrique | Valeur Attendue |
|----------|----------------|
| **Accuracy** | 92-96% |
| **F1-Score** | 0.91-0.95 |
| **Sensitivity** | 90-95% |
| **Specificity** | 92-97% |

### Où Trouver les Résultats ?

1. **Checkpoints** : `checkpoints/`
   - `best_model_epoch_XXX_f1_0.XXXX.pth` (meilleur modèle)
   - `checkpoint_epoch_XXX.pth` (sauvegardes périodiques)

2. **Exports ONNX** : `exports/onnx/`
   - `breastai_model_YYYYMMDD_HHMMSS.onnx`

3. **Logs** : `logs/`
   - `breastai_YYYYMMDD.log` (logs détaillés)
   - `training_report_YYYYMMDD_HHMMSS.json` (rapport final)

4. **Backups** : `backups/`
   - `breastai_backup_YYYYMMDD_HHMMSS.zip` (sauvegarde complète)

---

## 🆘 Problèmes Courants

### "Out of Memory"

```json
// Dans config.json, réduire :
{
  "data": {
    "batch_size": 2,      // Réduire à 2 ou 1
    "num_workers": 0      // Désactiver workers
  }
}
```

### "No class directories found"

```
Vérifier la structure :
data/train/benign/       ✅
data/train/malignant/    ✅
```

### "WebSocket connection failed"

```bash
# 1. Vérifier que le serveur est lancé
python server_aligned.py

# 2. Vérifier qu'il écoute sur 8765
# 3. Rafraîchir la page web
```

### Entraînement très lent

```json
// Réduire la taille des images :
{
  "data": {
    "image_size": 256     // Au lieu de 512
  }
}
```

---

## 📚 Aller Plus Loin

1. **README complet** : `README_FR.md`
   - Documentation détaillée
   - Tous les paramètres
   - API WebSocket

2. **Validation système** : `python validate_system.py`
   - Vérifier l'installation
   - Diagnostics complets

3. **Configuration avancée** : `config.json`
   - CBAM (attention mechanisms)
   - Augmentation de données
   - Early stopping
   - Schedulers personnalisés

---

## 🎓 Workflow Complet Recommandé

```
1. Installation       → pip install -r requirements.txt
2. Validation         → python validate_system.py
3. Préparer données   → Organiser dans data/train/
4. Configurer         → Éditer config.json si nécessaire
5. Lancer serveur     → python server_aligned.py
6. Ouvrir interface   → frontend/index.html
7. Connecter          → Clic "Se connecter"
8. Entraîner          → Clic "Démarrer Entraînement"
9. Surveiller         → Onglet Dashboard
10. Exporter          → Clic "Export ONNX"
11. Utiliser modèle   → exports/onnx/model.onnx
```

---

## 🏆 Vous êtes Prêt !

Maintenant vous pouvez :
- ✅ Entraîner un modèle de classification médical professionnel
- ✅ Surveiller l'entraînement en temps réel
- ✅ Exporter au format ONNX pour déploiement
- ✅ Obtenir des métriques cliniques détaillées
- ✅ Gérer vos checkpoints et backups

**Bon entraînement ! 🚀**

---

**Version 3.2.0** | **Support** : Consultez README_FR.md | **Licence** : MIT
