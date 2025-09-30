# 🏥 Configuration Clinique Optimale - BreastAI v3.3.1

## 📊 Configuration Recommandée pour Qualité Médicale Maximale

Cette configuration est optimisée pour obtenir les meilleures performances cliniques sur CPU (AMD Ryzen + Radeon).

---

## 🎯 Objectifs Cliniques

- **Sensibilité élevée** : Détection maximale des cas malins (minimiser faux négatifs)
- **Spécificité élevée** : Réduction des faux positifs
- **AUC-ROC** : >0.95 (standard clinique)
- **Reproductibilité** : Résultats stables et fiables

---

## ⚙️ Configuration Complète

### 🧠 Architecture du Modèle

```json
{
  "architecture": "efficientnetv2_l",
  "pretrained": true,
  "use_cbam": true,
  "dropout_rate": 0.4,
  "num_classes": 3
}
```

**Justification :**
- **EfficientNetV2-L** : Meilleure précision (vs S/M), architecture state-of-the-art
- **Pretrained** : Transfer learning sur ImageNet → convergence plus rapide
- **CBAM** : Module d'attention pour focus sur zones importantes
- **Dropout 0.4** : Régularisation forte pour éviter l'overfitting

---

### 📐 Données d'Entrée

```json
{
  "image_size": 512,
  "batch_size": 4,
  "num_workers": 2,
  "pin_memory": false
}
```

**Justification :**
- **512×512** : Résolution clinique standard (détails fins visibles)
- **Batch 4** : Maximum pour CPU sans OOM (Out of Memory)
- **num_workers 2** : Parallélisation CPU optimale

---

### 🎓 Entraînement

```json
{
  "epochs": 50,
  "learning_rate": 0.0003,
  "weight_decay": 0.0001,
  "optimizer": "adamw",
  "scheduler": "cosine",
  "label_smoothing": 0.1,
  "gradient_clip": 1.0
}
```

**Justification :**
- **50 epochs** : Convergence complète (médical ≠ natural images)
- **LR 3e-4** : Optimal pour EfficientNetV2 + transfer learning
- **Weight decay 1e-4** : Régularisation L2 modérée
- **AdamW** : Meilleur que Adam pour transfer learning
- **Cosine** : Décroissance douce du LR
- **Label smoothing** : Réduit la sur-confiance
- **Gradient clip** : Stabilité d'entraînement

---

### 🔓 Progressive Unfreezing (ACTIVÉ AUTOMATIQUEMENT)

```python
Phase 1 (Epochs 1-5)   : Backbone gelé
                         → Entraînement classifier seul
                         → ×3-4 plus rapide

Phase 2 (Epochs 6-15)  : Dégel partiel (3 derniers blocs)
                         → Fine-tuning progressif
                         → ×2 plus rapide

Phase 3 (Epochs 16-50) : Dégel complet + LR réduit (×0.1)
                         → Fine-tuning complet
                         → Vitesse normale
```

**Avantages :**
- Accélération massive sur CPU (×2.5-3 global)
- Convergence plus stable
- Meilleure généralisation

---

### 🎨 Augmentation de Données (Médicale)

```json
{
  "horizontal_flip": 0.5,
  "vertical_flip": 0.3,
  "rotation": 15,
  "brightness": 0.2,
  "contrast": 0.2,
  "use_clahe": true
}
```

**Justification :**
- **Flips** : Valides médicalement (pas d'orientation fixe)
- **Rotation ±15°** : Variations d'acquisition réalistes
- **Brightness/Contrast** : Variabilité équipement médical
- **CLAHE** : Normalisation histogramme (améliore contraste)

**❌ Pas d'augmentation agressive** :
- Pas de CutMix/Mixup (dégrade structures médicales)
- Pas de Gaussian Noise excessif
- Pas de Random Erasing agressif

---

### 📊 Métriques Cliniques

```json
{
  "primary_metric": "f1_weighted",
  "track_metrics": [
    "accuracy",
    "auc_roc",
    "auc_pr",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1_macro",
    "f1_weighted"
  ]
}
```

**Métriques critiques pour usage clinique :**

| Métrique | Importance | Objectif |
|----------|------------|----------|
| **Sensitivity (Recall)** | ⭐⭐⭐⭐⭐ | >95% (cancer = faux négatif grave) |
| **Specificity** | ⭐⭐⭐⭐ | >90% (limiter faux positifs) |
| **AUC-ROC** | ⭐⭐⭐⭐⭐ | >0.95 (standard clinique) |
| **PPV** | ⭐⭐⭐⭐ | >85% (confiance diagnostic +) |
| **NPV** | ⭐⭐⭐⭐⭐ | >98% (confiance diagnostic -) |
| **F1-Score** | ⭐⭐⭐ | Balance précision/rappel |

---

## ⏱️ Temps d'Entraînement Estimé

### CPU (AMD Ryzen) - 53k images

| Configuration | Sans Progressive Unfreezing | Avec Progressive Unfreezing |
|---------------|------------------------------|------------------------------|
| **Epochs 1-5** | ~55 jours | ~14-18 jours (×3-4) ✅ |
| **Epochs 6-15** | ~55 jours | ~27-33 jours (×2) ✅ |
| **Epochs 16-50** | ~110 jours | ~110 jours (normal) |
| **TOTAL (50 epochs)** | ~220 jours | **~70-90 jours** 🎯 |

### GPU (NVIDIA RTX 3060) - Comparaison

| Configuration | Temps Estimé |
|---------------|--------------|
| **EfficientNetV2-L** | ~3-4 jours |
| **EfficientNetV2-M** | ~2-3 jours |
| **EfficientNetV2-S** | ~1.5-2 jours |

---

## 🚀 Démarrage Rapide

### 1. Configuration Automatique

La configuration optimale est déjà définie dans `config.json` et l'interface web :

```bash
# Démarrer le serveur
python server_simple.py
```

### 2. Lancer via Interface Web

1. Ouvrir `frontend/app.html`
2. Cliquer sur **"Se connecter"**
3. **La configuration est déjà optimale par défaut** :
   - Modèle : EfficientNetV2-L ⭐
   - Epochs : 50
   - Batch Size : 4
   - Learning Rate : 0.0003
   - Weight Decay : 0.0001
   - Optimizer : AdamW
   - CBAM : Activé
4. Cliquer sur **"Démarrer"**

### 3. Progressive Unfreezing Automatique

✅ **Aucune configuration nécessaire** - Le système applique automatiquement :
- Epochs 1-5 : Backbone gelé
- Epochs 6-15 : Dégel partiel
- Epochs 16+ : Dégel complet

Les logs afficheront :
```
🔒 [Phase 1/3] Backbone GELÉ - Entraînement classifier seul (×3-4 plus rapide)
   → Paramètres entraînables: 524,803 / 118,515,843 (0.4%)
```

---

## 📈 Résultats Attendus

### Performance Cible (50 epochs)

```
Validation Metrics:
├─ Accuracy        : 94-96%
├─ AUC-ROC         : 0.96-0.98
├─ Sensitivity     : 95-97% (détection malignant)
├─ Specificity     : 92-95%
├─ F1-Score (macro): 0.93-0.95
└─ F1-Score (weighted): 0.94-0.96

Per-Class Performance:
├─ Normal    : F1 ~0.92-0.94
├─ Benign    : F1 ~0.91-0.93
└─ Malignant : F1 ~0.94-0.96 (prioritaire)
```

---

## 🔍 Validation Clinique

### Test sur Dataset Indépendant

Après entraînement, **valider impérativement sur** :

1. **Test set séparé** (jamais vu en train/val)
2. **Cross-validation** (si dataset petit)
3. **Dataset externe** (autre hôpital, autre équipement)

### Métriques Critiques

```python
# Sensibilité > 95% OBLIGATOIRE
# Faux négatifs = danger patient

# Spécificité > 90% RECOMMANDÉE
# Faux positifs = examens inutiles

# Confiance calibrée
# Probabilités = risque réel
```

---

## ⚠️ Limitations & Considérations

### Limitations CPU

- **Entraînement très long** (~70-90 jours pour 50 epochs)
- **Progressive Unfreezing obligatoire** pour temps raisonnable
- **Pas de Mixed Precision** (AMP incompatible CPU)

### Alternatives Recommandées

1. **Google Colab (GPU gratuit)** :
   - Tesla T4 gratuit
   - 50 epochs en ~6-8 heures
   - Notebook fourni dans le projet

2. **Kaggle Notebooks** :
   - GPU gratuit 30h/semaine
   - Compatible avec le code

3. **Cloud payant** :
   - AWS p3.2xlarge : ~$3/heure → ~$20 pour 50 epochs
   - GCP V100 : ~$2.5/heure → ~$15 pour 50 epochs

---

## 🎯 Checklist Pré-Entraînement

Avant de lancer un entraînement long :

- [ ] Dataset vérifié (pas d'images corrompues)
- [ ] Classes équilibrées (ou class_weights activé)
- [ ] Configuration sauvegardée
- [ ] Espace disque suffisant (checkpoints ~500MB chacun)
- [ ] Logs activés
- [ ] PC stable (pas de mise en veille)

---

## 📞 Support

Pour questions sur cette configuration :
- GitHub Issues : [project_breast_ai_train](https://github.com/Sabrsl/project_breast_ai_train/issues)
- Voir aussi : `QUICK_START_FR.md`, `GUIDE_INTERFACE_FR.md`

---

**Développé pour usage clinique - BreastAI Team © 2024**
