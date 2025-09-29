# 🖥️ Guide de l'Interface Web BreastAI v3.2.0

## Vue d'Ensemble

L'interface BreastAI est divisée en **4 zones principales** :
1. **Barre Latérale** (gauche) - Contrôles et métriques
2. **En-tête** - Statut et titre
3. **Onglets** - Différentes vues
4. **Zone de Contenu** - Éditeur, configuration, dashboard, logs

---

## 📍 Barre Latérale (Gauche)

### 🔌 Section "Connexion Serveur"
```
┌────────────────────────────────┐
│ Connexion Serveur              │
├────────────────────────────────┤
│ [🔌 Se connecter]              │
│ ⚫ Déconnecté                  │
└────────────────────────────────┘
```

**Actions** :
- Cliquer sur "Se connecter" → Connexion au serveur WebSocket
- L'indicateur devient vert ✅ quand connecté
- Texte passe à "Connecté"

---

### 🚀 Section "Actions Principales"
```
┌────────────────────────────────┐
│ Actions Principales            │
├────────────────────────────────┤
│ [🚀 Démarrer Entraînement]    │  → Lance l'entraînement
│ [⏹️ Arrêter]                   │  → Stoppe l'entraînement
│ [💾 Sauvegarder Code]          │  → Sauvegarde l'éditeur
│ [📁 Charger Code]              │  → Charge un fichier
│ [📤 Export ONNX]               │  → Exporte le modèle
└────────────────────────────────┘
```

**Notes** :
- Boutons grisés si non connecté
- "Démarrer" devient actif après connexion
- "Arrêter" s'active pendant l'entraînement

---

### 📋 Section "Gestion Checkpoints"
```
┌────────────────────────────────┐
│ Gestion Checkpoints            │
├────────────────────────────────┤
│ [📋 Lister Checkpoints]        │
│ [🔄 Reprendre Entraînement]    │
│                                │
│ Liste des checkpoints:         │
│ ┌────────────────────────────┐ │
│ │ best_model_epoch_45_...    │ │
│ │ Epoch 45 | F1: 0.9421     │ │
│ └────────────────────────────┘ │
│ ┌────────────────────────────┐ │
│ │ checkpoint_epoch_40        │ │
│ │ Epoch 40 | F1: 0.9315     │ │
│ └────────────────────────────┘ │
│                                │
│ [🔄 Reprendre depuis sélect.]  │
└────────────────────────────────┘
```

**Utilisation** :
1. Cliquer "Lister Checkpoints"
2. La liste apparaît en dessous
3. Cliquer sur un checkpoint pour le sélectionner (devient bleu)
4. Cliquer "Reprendre depuis sélectionné"

---

### ⚙️ Section "Fonctions Avancées"
```
┌────────────────────────────────┐
│ Fonctions Avancées             │
├────────────────────────────────┤
│ [📋 Rapport Clinique]          │  → Génère rapport détaillé
│ [⚙️ Diagnostics Système]       │  → CPU/RAM/Disque
│ [💾 Sauvegarde Projet]         │  → Backup complet
│ [📈 Visualisations Avancées]   │  → Graphiques supplémentaires
└────────────────────────────────┘
```

---

### 📊 Section "Progression Entraînement"
```
┌────────────────────────────────┐
│ Progression Entraînement       │
├────────────────────────────────┤
│ ████████████░░░░░░░░ 60%       │
│ Epoch 30/50 - En cours...      │
└────────────────────────────────┘
```

**Affichage** :
- Barre de progression animée
- Pourcentage et epoch actuel
- Message de statut

---

### 📈 Section "Métriques en Temps Réel"
```
┌──────────────────────────────┐
│ Métriques en Temps Réel      │
├──────────────────────────────┤
│ ┌──────┐ ┌──────┐ ┌──────┐  │
│ │  30  │ │ 0.45 │ │ 89%  │  │
│ │Epoch │ │ Loss │ │ Acc  │  │
│ └──────┘ └──────┘ └──────┘  │
│ ┌──────┐                     │
│ │ 0.87 │                     │
│ │  F1  │                     │
│ └──────┘                     │
└──────────────────────────────┘
```

**Mise à jour** :
- Automatique chaque epoch
- Valeurs en couleur (bleu cyan)
- Animation lors des changements

---

## 📑 Onglets Principaux

### 📝 Onglet "Éditeur de Code"
```
┌─────────────────────────────────────────────────┐
│ 📝 Éditeur de Code                              │
├─────────────────────────────────────────────────┤
│  1 | import torch                                │
│  2 | import torchvision                          │
│  3 |                                             │
│  4 | # Configuration                             │
│  5 | config = {                                  │
│  6 |     "epochs": 50,                          │
│  7 |     "batch_size": 4,                       │
│  8 | }                                           │
│  9 |                                             │
│ ...│                                             │
└─────────────────────────────────────────────────┘
```

**Fonctionnalités** :
- Éditeur ACE avec coloration syntaxique Python
- Numéros de ligne
- Auto-indentation
- Rechercher/Remplacer (Ctrl+F)
- Sauvegarde avec bouton "💾 Sauvegarder Code"

---

### ⚙️ Onglet "Configuration Avancée"
```
┌─────────────────────────────────────────────────────┐
│ ⚙️ Configuration Avancée                            │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ╔══════════════════════╗  ╔══════════════════════╗ │
│  ║ Configuration de Base║  ║  Optimiseur & Sched. ║ │
│  ╠══════════════════════╣  ╠══════════════════════╣ │
│  ║ Dataset Path         ║  ║ Optimizer: AdamW     ║ │
│  ║ ./data/train         ║  ║ Weight Decay: 0.001  ║ │
│  ║                      ║  ║ Scheduler: Cosine    ║ │
│  ║ Epochs: 50           ║  ║ T_0: 10             ║ │
│  ║ Batch Size: 4        ║  ║ T_mult: 2           ║ │
│  ║ Learning Rate: 0.0003║  ║ [x] AMP             ║ │
│  ║ Model: efficientnetv2║  ║                     ║ │
│  ║ Image Size: 512      ║  ║                     ║ │
│  ║ Num Classes: 3       ║  ║                     ║ │
│  ╚══════════════════════╝  ╚══════════════════════╝ │
│                                                      │
│  ╔══════════════════════╗  ╔══════════════════════╗ │
│  ║ Modules d'Attention  ║  ║ Augmentation Données ║ │
│  ╠══════════════════════╣  ╠══════════════════════╣ │
│  ║ [x] Activer CBAM     ║  ║ [x] Augmentation     ║ │
│  ║ Reduction: 16        ║  ║ [x] Horizontal Flip  ║ │
│  ║ [x] Spatial CBAM     ║  ║ [x] Vertical Flip    ║ │
│  ║ [x] Channel CBAM     ║  ║ Rotation: 15°        ║ │
│  ║ Position: Sequential ║  ║ Brightness: 0.2      ║ │
│  ╚══════════════════════╝  ╚══════════════════════╝ │
│                                                      │
└─────────────────────────────────────────────────────┘
```

**Sections** :
1. **Configuration de Base** : Paramètres essentiels
2. **Optimiseur & Scheduler** : Paramètres d'optimisation
3. **Modules d'Attention CBAM** : Configuration CBAM
4. **Augmentation de Données** : Transformations d'images
5. **Early Stopping** : Critères d'arrêt
6. **Ressources Système** : CPU threads, num workers

**Modification** :
- Changer les valeurs
- Les valeurs sont automatiquement envoyées au serveur
- Validation en temps réel

---

### 📊 Onglet "Dashboard Temps Réel"
```
┌───────────────────────────────────────────────────┐
│ 📊 Dashboard Temps Réel                           │
├───────────────────────────────────────────────────┤
│                                                    │
│ ┌────────────────────┐  ┌────────────────────┐   │
│ │  Training Loss     │  │  Validation Loss   │   │
│ │                    │  │                    │   │
│ │      📉            │  │      📉            │   │
│ │   Loss vs Epoch    │  │   Loss vs Epoch    │   │
│ └────────────────────┘  └────────────────────┘   │
│                                                    │
│ ┌────────────────────┐  ┌────────────────────┐   │
│ │  Train Accuracy    │  │  Validation Metrics│   │
│ │                    │  │                    │   │
│ │      📈            │  │      📊            │   │
│ │   Acc vs Epoch     │  │  F1/Prec/Recall    │   │
│ └────────────────────┘  └────────────────────┘   │
│                                                    │
└───────────────────────────────────────────────────┘
```

**Graphiques** :
1. **Training Loss** : Évolution de la loss d'entraînement
2. **Validation Loss** : Évolution de la loss de validation
3. **Train Accuracy** : Précision sur le train
4. **Validation Metrics** : F1, Precision, Recall

**Interactivité** :
- Zoom avec molette de souris
- Survol pour valeurs exactes
- Légende cliquable pour masquer/afficher

---

### 🖥️ Onglet "Console de Logs"
```
┌───────────────────────────────────────────────────┐
│ 🖥️ Console de Logs                                │
├───────────────────────────────────────────────────┤
│                                                    │
│ [INFO] 2024-09-29 10:23:45                       │
│ │ BreastAI Production System v3.2.0               │
│ │ Starting training session...                    │
│                                                    │
│ [SUCCESS] 2024-09-29 10:23:50                    │
│ │ Data loaded successfully                        │
│ │ Classes detected: ['benign', 'malignant']       │
│ │ Total samples: 2000                             │
│                                                    │
│ [INFO] 2024-09-29 10:24:00                       │
│ │ Epoch 1/50                                      │
│ │ Training... [=========>........] 45%            │
│                                                    │
│ [WARNING] 2024-09-29 10:25:00                    │
│ │ High memory usage detected: 12.3 GB             │
│                                                    │
│ [ERROR] 2024-09-29 10:26:00                      │
│ │ Failed to load image: corrupted.jpg             │
│ │ Using dummy image instead                       │
│                                                    │
└───────────────────────────────────────────────────┘
```

**Types de Messages** :
- 🔵 **[INFO]** : Informations générales (bleu)
- ✅ **[SUCCESS]** : Opérations réussies (vert)
- ⚠️ **[WARNING]** : Avertissements (orange)
- ❌ **[ERROR]** : Erreurs (rouge)

**Fonctionnalités** :
- Auto-scroll vers le bas
- Timestamps précis
- Coloration syntaxique
- Filtrage par type (futur)

---

## 🎨 Indicateurs Visuels

### Statut de Connexion
```
⚫ Déconnecté   → Pas de connexion au serveur
🟢 Connecté     → Connexion WebSocket active
🔴 Erreur       → Problème de connexion
```

### État des Boutons
```
[Bouton Actif]      → Cliquable, couleur vive
[Bouton Désactivé]  → Grisé, curseur interdit
[Bouton En Cours]   → Animation spinner
```

### Barre de Progression
```
████████░░░░░░░░ 40%  → En cours
████████████████ 100% → Terminé
```

---

## 🔔 Notifications Toast

Les notifications apparaissent en haut à droite :

```
┌─────────────────────────────┐
│ ✅ Connexion établie        │  → Succès (vert)
└─────────────────────────────┘

┌─────────────────────────────┐
│ ⚠️ Checkpoint non trouvé    │  → Avertissement (orange)
└─────────────────────────────┘

┌─────────────────────────────┐
│ ❌ Erreur de connexion      │  → Erreur (rouge)
└─────────────────────────────┘

┌─────────────────────────────┐
│ ℹ️ Export ONNX en cours...  │  → Info (bleu)
└─────────────────────────────┘
```

**Durée** : 3-5 secondes, puis disparaissent automatiquement

---

## 🎯 Workflows Typiques

### Workflow 1 : Premier Entraînement
```
1. Cliquer "🔌 Se connecter"
   → Attendre indicateur vert

2. Onglet "⚙️ Configuration Avancée"
   → Vérifier/Modifier les paramètres

3. Revenir à l'onglet principal
   → Cliquer "🚀 Démarrer Entraînement"

4. Onglet "📊 Dashboard"
   → Surveiller les graphiques

5. Onglet "🖥️ Console"
   → Suivre les logs détaillés

6. Attendre fin de l'entraînement
   → Notification de fin

7. Cliquer "📤 Export ONNX"
   → Modèle exporté dans exports/onnx/
```

### Workflow 2 : Reprendre un Entraînement
```
1. Cliquer "🔌 Se connecter"

2. Cliquer "📋 Lister Checkpoints"
   → Liste des checkpoints apparaît

3. Cliquer sur un checkpoint
   → Il devient bleu (sélectionné)

4. Cliquer "🔄 Reprendre depuis sélectionné"
   → Chargement du checkpoint

5. Cliquer "🚀 Démarrer Entraînement"
   → Continue depuis l'epoch sauvegardée
```

### Workflow 3 : Diagnostic Système
```
1. Cliquer "🔌 Se connecter"

2. Cliquer "⚙️ Diagnostics Système"

3. Onglet "🖥️ Console"
   → Voir les diagnostics détaillés:
     - CPU : nombre de threads, usage
     - RAM : total, utilisé, disponible
     - Disque : espace libre
     - PyTorch : version, CUDA
```

---

## ⌨️ Raccourcis Clavier

Dans l'éditeur de code :
- `Ctrl + S` : Sauvegarder le code
- `Ctrl + F` : Rechercher
- `Ctrl + H` : Remplacer
- `Ctrl + /` : Commenter/Décommenter
- `Tab` : Indentation
- `Shift + Tab` : Dé-indentation

---

## 📱 Responsive Design

L'interface s'adapte à différentes tailles d'écran :

### Desktop (> 1024px)
- Barre latérale visible à gauche
- Dashboard en grille 2x2
- Tout l'espace utilisé

### Tablet (768-1024px)
- Barre latérale repliable
- Dashboard en grille 1x4
- Colonnes simplifiées

### Mobile (< 768px)
- Barre latérale en haut
- Dashboard vertical
- Configuration sur une colonne

---

## 🎨 Thème Visuel

### Couleurs Principales
- **Fond** : Gradient bleu foncé (style cyberpunk)
- **Cartes** : Verre dépoli avec backdrop-filter
- **Accent** : Cyan (#00d4ff)
- **Succès** : Vert (#44ff44)
- **Erreur** : Rouge (#ff4444)
- **Warning** : Orange (#ffaa00)

### Animations
- Transitions fluides (0.3s)
- Effets de survol (hover)
- Shimmer sur les barres de progression
- Pulse sur les indicateurs de statut

---

## 🔧 Troubleshooting Interface

### Problème : Boutons grisés
**Cause** : Pas de connexion au serveur  
**Solution** : Cliquer "🔌 Se connecter"

### Problème : Graphiques vides
**Cause** : Pas d'entraînement en cours  
**Solution** : Démarrer un entraînement

### Problème : Interface figée
**Cause** : Connexion WebSocket perdue  
**Solution** : Rafraîchir la page (F5)

### Problème : Métriques ne s'affichent pas
**Cause** : Serveur non lancé  
**Solution** : `python server_aligned.py`

---

**Version** : 3.2.0  
**Dernière mise à jour** : Septembre 2024
