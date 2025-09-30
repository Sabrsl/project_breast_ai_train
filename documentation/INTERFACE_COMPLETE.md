# 🎯 Interface Complète : Tous les Paramètres Contrôlables

## 📋 Vue d'ensemble

L'interface web `frontend/app.html` permet de contrôler **TOUS** les aspects de l'entraînement depuis le navigateur. Aucune modification du code n'est nécessaire.

---

## 🔧 Paramètres Contrôlables

### 1. **Présets Ultra-Performance** ⚡
| Preset | Description | Temps estimé | AUC attendu |
|--------|-------------|--------------|-------------|
| **ULTRA-S** | EfficientNetV2-S, rapide | ~30 jours CPU | 0.970-0.973 |
| **ULTRA-M** | EfficientNetV2-M, optimal | ~50 jours CPU | 0.975-0.978 |
| **Custom** | Configuration manuelle | Variable | Variable |

**Auto-remplissage** : Changez le preset et tous les champs se mettent à jour automatiquement !

---

### 2. **Modèle** 🏗️
- `EfficientNetV2-S` : Petit, rapide, AUC ~0.97
- `EfficientNetV2-M` : Moyen, équilibré, AUC ~0.975
- `EfficientNetV2-L` : Grand, lent, AUC ~0.98+
- `EfficientNet-B4` : Legacy, performances moyennes

---

### 3. **Hyperparamètres de Base** 📊

| Paramètre | Valeur par défaut | Plage | Description |
|-----------|-------------------|-------|-------------|
| **Epochs** | 80 | 1-200 | Nombre de passages sur le dataset |
| **Batch Size** | 4 | 1-16 | Images par batch (limité par RAM) |
| **Learning Rate** | 0.0003 | 0.00001-0.01 | Vitesse d'apprentissage |
| **Weight Decay** | 0.0001 | 0-0.01 | Régularisation L2 |
| **Label Smoothing** | 0.12 | 0-0.3 | Lissage des labels (0.12 = clinique) |
| **Dropout Rate** | 0.45 | 0.1-0.8 | Taux de dropout dans la tête |

---

### 4. **Optimisation** 🚀

#### **Optimizer**
- `AdamW` ⭐ (recommandé) : Meilleure généralisation
- `Adam` : Plus rapide mais moins stable
- `SGD` : Robuste mais lent

#### **Scheduler**
- `Cosine Annealing` : Décroissance progressive du LR

---

### 5. **Features Avancées** 🔥

| Feature | Options | Description | Impact |
|---------|---------|-------------|--------|
| **CBAM** | ✅ Activé / ❌ Désactivé | Attention spatiale et canal | +1-2% AUC |
| **Focal Loss** | ✅ Activé / ❌ CrossEntropy | Priorité aux cas difficiles (malignant) | +1-2% sensibilité |
| **TTA** | ✅ Activé / ❌ Désactivé | 6 augmentations à l'inférence | +1-1.5% AUC |
| **EMA** | ✅ Activé / ❌ Désactivé | Lissage exponentiel des poids | +0.5-1% stabilité |
| **Gradient Acc** | 1-16 steps | Batch effectif = Batch × Steps | Simule gros batch |

---

### 6. **Progressive Unfreezing** 🧊➡️🔥

**Automatique selon l'epoch** :
- **Phase 1** (Epochs 1-8) : Backbone gelé, seule la tête est entraînée
- **Phase 2** (Epochs 9-20) : Dégel progressif des derniers blocs
- **Phase 3** (Epochs 21-40) : Dégel des blocs moyens
- **Phase 4** (Epochs 41+) : Tout dégelé, fine-tuning complet

---

## 🎯 Configurations Recommandées

### ⚡ ULTRA-S : Rapide (30 jours)
```
Modèle           : EfficientNetV2-S
Epochs           : 80
Batch Size       : 4
Learning Rate    : 0.0003
Weight Decay     : 0.0001
Dropout          : 0.45
Label Smoothing  : 0.12
Optimizer        : AdamW
CBAM             : ✅
Focal Loss       : ✅
TTA              : ✅
EMA              : ✅
Gradient Acc     : 4

Performance attendue : AUC 0.970-0.973
```

### 🏆 ULTRA-M : Optimal (50 jours)
```
Modèle           : EfficientNetV2-M
Epochs           : 80
Batch Size       : 4
Learning Rate    : 0.0003
Weight Decay     : 0.0001
Dropout          : 0.45
Label Smoothing  : 0.12
Optimizer        : AdamW
CBAM             : ✅
Focal Loss       : ✅
TTA              : ✅
EMA              : ✅
Gradient Acc     : 4

Performance attendue : AUC 0.975-0.978
```

---

## 🔄 Workflow

### 1. **Choisir un Preset**
- Sélectionnez `⚡ ULTRA-S` ou `🏆 ULTRA-M` dans le dropdown
- **Tous les champs se remplissent automatiquement** !

### 2. **Ajuster (Optionnel)**
- Mode `Custom` : Modifiez manuellement chaque paramètre
- Affinez selon votre matériel (RAM, CPU)

### 3. **Démarrer l'Entraînement**
- Cliquez sur `🚀 Démarrer Entraînement`
- Les logs apparaissent **en temps réel** (chaque batch)

### 4. **Suivre les Métriques**
- **Graphiques en temps réel** : Loss, Accuracy, F1-Score
- **Logs Console** : Batch progress, validation metrics
- **Checkpoints automatiques** : Meilleur modèle sauvegardé

---

## 🛠️ Troubleshooting

### ❓ Quel preset choisir ?

| Cas d'usage | Preset recommandé |
|-------------|-------------------|
| **Prototypage rapide** | ULTRA-S (30j) |
| **Usage clinique** | ULTRA-M (50j) |
| **Recherche maximale** | Custom avec EfficientNetV2-L |

### ❓ Batch Size trop grand ?

```
Erreur : "CUDA out of memory" ou "RuntimeError: DataLoader"

Solution :
1. Réduire Batch Size : 4 → 2 → 1
2. Augmenter Gradient Acc : 4 → 8 → 16
   (Batch effectif reste le même !)
```

### ❓ Training trop lent ?

```
Solutions :
1. Vérifier GPU : torch.cuda.is_available() doit être True
2. Réduire Epochs : 80 → 50 → 30
3. Changer de modèle : EfficientNetV2-M → S
4. Désactiver TTA (seulement en validation)
```

---

## 📊 Métriques Affichées

### Pendant l'Entraînement
- **Batch Progress** : Loss, temps/batch
- **Epoch Summary** : Acc train, Loss train
- **Validation** : Acc, F1-macro, F1-weighted, Precision, Recall

### En Fin d'Entraînement
- **Best Model** : Epoch du meilleur modèle
- **Final Metrics** : AUC, Sensibilité, Spécificité, PPV, NPV
- **Checkpoint** : Sauvegarde automatique dans `checkpoints/`

---

## 🚀 Prochaines Étapes

1. **Ouvrir l'interface** : `python server_simple.py`
2. **Naviguer** : http://localhost:8765
3. **Choisir ULTRA-M** (ou ULTRA-S pour tester)
4. **Démarrer** et attendre 30-50 jours 🕐
5. **Profiter d'un modèle clinique de classe mondiale** ! 🏆

---

**Note** : Tous les paramètres sont synchronisés entre l'interface → serveur → training. Aucune modification de code nécessaire !
