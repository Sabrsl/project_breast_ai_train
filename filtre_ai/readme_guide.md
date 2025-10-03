# 🚀 Guide d'installation - Téléchargeur d'images multi-sources

## 📋 Prérequis

```bash
pip install requests tqdm pillow
```

## 🔑 Obtenir vos clés API (GRATUITES)

### 1. **Pixabay** (Recommandé - 5000 requêtes/heure)
- Créer un compte sur https://pixabay.com
- Aller sur https://pixabay.com/api/docs/
- Copier votre clé API (gratuit, instantané)
- **Limite**: 5000 requêtes/heure

### 2. **Pexels** (200 requêtes/heure)
- Créer un compte sur https://www.pexels.com
- Aller sur https://www.pexels.com/api/
- Créer une nouvelle application
- Copier votre clé API
- **Limite**: 200 requêtes/heure

### 3. **Unsplash** (50 requêtes/heure)
- Créer un compte sur https://unsplash.com/developers
- Créer une nouvelle application
- Copier votre "Access Key"
- **Limite**: 50 requêtes/heure (peut nécessiter approbation pour plus)

## ⚙️ Configuration

Ouvrir le fichier `image_downloader.py` et modifier la section `Config`:

```python
class Config:
    # Remplacer par vos vraies clés API
    PIXABAY_API_KEY = "123456789abcdef"  # Votre clé Pixabay
    UNSPLASH_ACCESS_KEY = "abc123xyz"     # Votre clé Unsplash
    PEXELS_API_KEY = "def456uvw"          # Votre clé Pexels
    
    # Modifier le chemin de sortie si nécessaire
    OUTPUT_DIR = r"C:\Users\badza\Desktop\project_breast_ai\filtre_ai\downloaded_images"
    
    # Ajuster les limites selon vos besoins
    MAX_IMAGES_PER_SOURCE = 1000  # Nombre max d'images par catégorie
```

## 🎯 Personnaliser les catégories

Vous pouvez modifier les mots-clés de recherche dans `SEARCH_QUERIES`:

```python
SEARCH_QUERIES = {
    'non_medical': [
        'landscape', 'nature', 'city', 'food', 'animal', 'car',
        # Ajoutez vos propres mots-clés ici
    ],
    'medical_other': [
        'hospital', 'doctor', 'xray', 'surgery',
        # Ajoutez vos propres mots-clés ici
    ],
    'breast': [
        'mammography', 'breast ultrasound', 'mammogram',
        # Ajoutez vos propres mots-clés ici
    ]
}
```

## ▶️ Lancer le téléchargement

```bash
python image_downloader.py
```

## 📊 Structure de sortie

```
downloaded_images/
├── non_medical/
│   ├── pixabay_non_medical_1234567890_1234.jpg
│   ├── unsplash_non_medical_1234567891_5678.jpg
│   └── pexels_non_medical_1234567892_9012.jpg
├── medical_other/
│   └── ...
└── breast/
    └── ...
```

## ⚠️ Conseils anti-blocage

Le script intègre déjà ces protections:

1. **Délai entre requêtes**: 1.5 secondes (modifiable)
2. **Rotation de User-Agent**: Change automatiquement
3. **Gestion des erreurs**: Retry automatique avec exponential backoff
4. **Détection de doublons**: Hash MD5 pour éviter les images identiques
5. **Timeout**: 30 secondes par requête

## 🔥 Limitations des API gratuites

| Source | Requêtes/heure | Images/page | Remarques |
|--------|---------------|-------------|-----------|
| **Pixabay** | 5000 | 50 | Meilleure option |
| **Pexels** | 200 | 80 | Bonne qualité |
| **Unsplash** | 50 | 30 | Haute qualité mais limité |

## 🎓 Estimation du temps

Pour télécharger **1000 images par catégorie** (3000 total):

- Avec délai de 1.5s: ~75 minutes
- Avec délai de 1.0s: ~50 minutes (risque de blocage)
- Avec délai de 2.0s: ~100 minutes (plus sûr)

## 🐛 Dépannage

### Erreur "Invalid API key"
- Vérifiez que vous avez bien copié la clé complète
- Vérifiez qu'il n'y a pas d'espaces avant/après
- Pour Unsplash, attendez quelques minutes après création

### Erreur "Rate limit exceeded"
- Augmentez `DELAY_BETWEEN_REQUESTS` à 2 ou 3 secondes
- Réduisez `MAX_IMAGES_PER_SOURCE`
- Attendez 1 heure avant de relancer

### Images non téléchargées
- Vérifiez votre connexion internet
- Vérifiez les permissions du dossier de sortie
- Certains mots-clés peuvent donner 0 résultat

### Script bloqué
- Appuyez sur Ctrl+C pour arrêter
- Le script reprendra là où il s'est arrêté

## 💡 Optimisations

### Télécharger plus vite (attention au rate limit)
```python
DELAY_BETWEEN_REQUESTS = 0.5  # Plus rapide mais risqué
```

### Télécharger en qualité maximale
```python
# Dans chaque downloader, modifier:
'url': hit['largeImageURL']  → 'url': hit['fullHDURL']  # Pixabay
'url': result['urls']['regular']  → 'url': result['urls']['full']  # Unsplash
```

### Filtrer par taille minimale
```python
MIN_IMAGE_WIDTH = 1920   # Full HD
MIN_IMAGE_HEIGHT = 1080
```

## 🔒 Sécurité

- ✅ Toutes les API sont officielles et légales
- ✅ Images gratuites et libres de droits
- ✅ Respecte les rate limits des API
- ✅ Pas de scraping agressif

## 📞 Support

Si vous rencontrez des problèmes:
1. Vérifiez que vos clés API sont valides
2. Vérifiez les logs d'erreur dans le terminal
3. Testez avec une seule source d'abord
4. Réduisez `MAX_IMAGES_PER_SOURCE` à 10 pour tester