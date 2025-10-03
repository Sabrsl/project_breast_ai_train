"""
Script de téléchargement d'images PIXABAY UNIQUEMENT - OPTIMISÉ ANTI-BLOCAGE
Avec rate limiting intelligent, retry exponential, et rotation avancée
"""

import os
import time
import requests
import json
from pathlib import Path
from urllib.parse import urlparse
import random
from typing import List, Dict, Optional
from tqdm import tqdm
import hashlib
from datetime import datetime, timedelta
import logging
from collections import defaultdict

# ======================== CONFIGURATION ========================
class Config:
    # CLÉ API PIXABAY (gratuite - 5000 req/heure)
    PIXABAY_API_KEY = "22900112-27711f944a261ba0b3a11a3e1"
    
    # Dossier de téléchargement
    OUTPUT_DIR = r"C:\Users\badza\Desktop\project_breast_ai\filtre_ai\downloaded_images"
    
    # Limites par source (respectent les quotas API)
    MAX_IMAGES_PER_SOURCE = 500  # Réduit pour sécurité
    MIN_IMAGE_WIDTH = 640
    MIN_IMAGE_HEIGHT = 480
    
    # ⚡ ANTI-BLOCAGE OPTIMISÉ
    DELAYS = {
        'pixabay': (2.0, 4.0),      # 2-4s entre requêtes (bien sous 5000/h)
        'download': (1.0, 2.5)      # 1-2.5s entre téléchargements
    }
    
    # Retry et timeout
    MAX_RETRIES = 5
    INITIAL_RETRY_DELAY = 2  # Secondes
    MAX_RETRY_DELAY = 60
    TIMEOUT = 45
    
    # Rate limiting pour Pixabay
    RATE_LIMITS = {
        'pixabay': {'requests_per_hour': 4500, 'requests_per_minute': 75}
    }
    
    # User agents diversifiés
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0'
    ]
    
    # Requêtes par catégorie - NOUVELLES RECHERCHES (pas de doublons)
    SEARCH_QUERIES = {
    'sports_people': [
        # Sports avec des personnes
        'soccer player', 'basketball player', 'tennis player',
        'volleyball team', 'boxing match', 'runner athlete',
        'swimmer', 'gymnast', 'martial artist',
        'cyclist', 'skateboarder', 'surfer'
    ],

    'education_people': [
        # Éducation & étudiants/professeurs
        'teacher in classroom', 'students studying',
        'graduation ceremony students', 'scientist in lab',
        'professor teaching', 'children learning',
        'university lecture hall', 'researchers working',
        'students in library', 'kids with textbooks'
    ],

    'transport_people': [
        # Transport avec humains
        'bus passengers', 'people riding bikes',
        'family in car', 'commuters in metro',
        'pilot in cockpit', 'sailors on ship',
        'people boarding train', 'driver taxi',
        'tourists on ferry'
    ],

    'events_people': [
        # Événements & célébrations humaines
        'wedding couple', 'birthday party kids',
        'crowd at festival', 'parade with people',
        'people watching fireworks', 'christmas family',
        'halloween costumes people', 'new year celebration crowd',
        'family anniversary party'
    ],

    'creativity_people': [
        # Créativité et artisanat avec humains
        'artist painting', 'sculptor at work',
        'child drawing', 'woman knitting',
        'man woodworking', 'people doing origami',
        'jewelry maker', 'calligraphy artist',
        'fashion designer sewing'
    ],

    'seasons_people': [
        # Saisons avec présence humaine
        'people walking in rain', 'kids playing in snow',
        'family at beach summer', 'friends watching sunset',
        'couple in autumn park', 'hikers in fog',
        'children flying kite spring'
    ],

    'home_people': [
        # Intérieur de maison avec humains
        'family in living room', 'mother cooking kitchen',
        'kids in bedroom', 'people gardening backyard',
        'friends having dinner table', 'child studying desk',
        'people decorating home'
    ],

    'technology_people': [
        # Innovation et tech avec humains
        'engineer with robot', 'person flying drone',
        'gamer with virtual reality headset',
        'technician fixing server',
        'programmer coding computer',
        'scientist with satellite model',
        'team working on blockchain project'
    ]
}


# ======================== LOGGING ========================
def setup_logging():
    """Configure le système de logging"""
    log_dir = os.path.join(Config.OUTPUT_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'download_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ======================== RATE LIMITER ========================
class RateLimiter:
    """Gestionnaire de rate limiting avec historique des requêtes"""
    
    def __init__(self, source_name: str):
        self.source_name = source_name
        self.request_history = []
        self.limits = Config.RATE_LIMITS.get(source_name, {
            'requests_per_hour': 100,
            'requests_per_minute': 10
        })
    
    def can_make_request(self) -> bool:
        """Vérifie si une requête peut être faite"""
        now = datetime.now()
        
        # Nettoyer l'historique (garder dernière heure)
        self.request_history = [
            ts for ts in self.request_history 
            if now - ts < timedelta(hours=1)
        ]
        
        # Vérifier limite par minute
        last_minute = [ts for ts in self.request_history if now - ts < timedelta(minutes=1)]
        if len(last_minute) >= self.limits['requests_per_minute']:
            return False
        
        # Vérifier limite par heure
        if len(self.request_history) >= self.limits['requests_per_hour']:
            return False
        
        return True
    
    def wait_if_needed(self):
        """Attend si nécessaire avant de faire une requête"""
        while not self.can_make_request():
            logging.warning(f"⏳ Rate limit atteint pour {self.source_name}, attente 10s...")
            time.sleep(10)
    
    def record_request(self):
        """Enregistre une requête"""
        self.request_history.append(datetime.now())

# ======================== UTILITAIRES ========================
def get_random_user_agent():
    """Retourne un user agent aléatoire"""
    return random.choice(Config.USER_AGENTS)

def create_session():
    """Crée une session requests avec configuration optimale"""
    session = requests.Session()
    
    # Headers de base
    session.headers.update({
        'User-Agent': get_random_user_agent(),
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9,fr;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    })
    
    # Configuration de retry au niveau TCP
    adapter = requests.adapters.HTTPAdapter(
        max_retries=requests.adapters.Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    return session

def calculate_hash(content: bytes) -> str:
    """Calcule le hash MD5 d'une image"""
    return hashlib.md5(content).hexdigest()

def safe_filename(url: str, prefix: str = 'img') -> str:
    """Génère un nom de fichier sûr"""
    timestamp = int(time.time() * 1000)
    random_suffix = random.randint(1000, 9999)
    ext = os.path.splitext(urlparse(url).path)[1] or '.jpg'
    return f"{prefix}_{timestamp}_{random_suffix}{ext}"

def smart_delay(delay_range: tuple):
    """Applique un délai aléatoire dans une plage"""
    delay = random.uniform(*delay_range)
    time.sleep(delay)

def exponential_backoff(attempt: int) -> float:
    """Calcule le délai avec backoff exponentiel"""
    delay = min(
        Config.INITIAL_RETRY_DELAY * (2 ** attempt),
        Config.MAX_RETRY_DELAY
    )
    # Ajouter jitter (±20%)
    jitter = delay * random.uniform(-0.2, 0.2)
    return delay + jitter

def download_image(url: str, output_path: str, session: requests.Session) -> tuple:
    """Télécharge une image avec retry exponentiel"""
    for attempt in range(Config.MAX_RETRIES):
        try:
            # Rotation user agent
            session.headers['User-Agent'] = get_random_user_agent()
            
            # Requête avec stream
            response = session.get(
                url, 
                timeout=Config.TIMEOUT, 
                stream=True,
                allow_redirects=True
            )
            
            # Vérifier le statut
            if response.status_code == 429:
                wait_time = exponential_backoff(attempt)
                logging.warning(f"  ⏳ Rate limit (429), attente {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            
            # Vérifier le content-type
            content_type = response.headers.get('content-type', '').lower()
            if 'image' not in content_type:
                return False, f"Not an image: {content_type}"
            
            # Télécharger avec limite de taille (50MB max)
            content = b''
            max_size = 50 * 1024 * 1024  # 50MB
            
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > max_size:
                    return False, "Image too large (>50MB)"
            
            # Vérifier taille minimale (1KB)
            if len(content) < 1024:
                return False, "Image too small (<1KB)"
            
            # Sauvegarder
            with open(output_path, 'wb') as f:
                f.write(content)
            
            return True, "Success"
            
        except requests.exceptions.Timeout:
            if attempt < Config.MAX_RETRIES - 1:
                wait_time = exponential_backoff(attempt)
                logging.warning(f"  ⏳ Timeout, retry {attempt+1}/{Config.MAX_RETRIES} dans {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                return False, "Timeout after retries"
                
        except requests.exceptions.RequestException as e:
            if attempt < Config.MAX_RETRIES - 1:
                wait_time = exponential_backoff(attempt)
                logging.warning(f"  ⏳ Erreur réseau, retry dans {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                return False, f"Network error: {str(e)}"
                
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
    
    return False, "Max retries exceeded"

# ======================== PIXABAY ========================
class PixabayDownloader:
    """Téléchargeur Pixabay avec rate limiting"""
    
    BASE_URL = "https://pixabay.com/api/"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = create_session()
        self.rate_limiter = RateLimiter('pixabay')
    
    def search_images(self, query: str, page: int = 1, per_page: int = 50) -> List[Dict]:
        """Recherche d'images sur Pixabay"""
        if not self.api_key or self.api_key == "VOTRE_CLE_PIXABAY":
            return []
        
        # Rate limiting
        self.rate_limiter.wait_if_needed()
        
        params = {
            'key': self.api_key,
            'q': query,
            'image_type': 'photo',
            'min_width': Config.MIN_IMAGE_WIDTH,
            'min_height': Config.MIN_IMAGE_HEIGHT,
            'page': page,
            'per_page': per_page,
            'safesearch': 'true',
            'editors_choice': 'false'
        }
        
        try:
            # Delay avant requête
            smart_delay(Config.DELAYS['pixabay'])
            
            response = self.session.get(
                self.BASE_URL, 
                params=params, 
                timeout=Config.TIMEOUT
            )
            
            self.rate_limiter.record_request()
            
            if response.status_code == 429:
                logging.warning("  ⚠️ Pixabay rate limit atteint, pause...")
                time.sleep(60)
                return []
            
            response.raise_for_status()
            data = response.json()
            
            images = []
            for hit in data.get('hits', []):
                images.append({
                    'url': hit.get('largeImageURL', hit.get('webformatURL')),
                    'id': hit['id'],
                    'source': 'pixabay'
                })
            
            return images
            
        except Exception as e:
            logging.error(f"  ❌ Erreur Pixabay: {e}")
            return []

# ======================== GESTIONNAIRE PRINCIPAL ========================
class ImageDownloadManager:
    """Gestionnaire principal optimisé anti-blocage"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.downloader = PixabayDownloader(Config.PIXABAY_API_KEY)
        self.session = create_session()
        self.downloaded_hashes = set()
        self.stats = defaultdict(lambda: defaultdict(int))
    
    def download_for_category(self, category: str, queries: List[str], max_images: int) -> int:
        """Télécharge des images pour une catégorie"""
        output_dir = os.path.join(Config.OUTPUT_DIR, category)
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"📥 CATÉGORIE: {category.upper()}")
        self.logger.info(f"{'='*70}")
        
        total_downloaded = 0
        images_per_query = max_images // len(queries)
        
        for query in queries:
            if total_downloaded >= max_images:
                break
            
            self.logger.info(f"\n🔍 Recherche: '{query}' (objectif: {images_per_query} images)")
            query_downloaded = 0
            
            self.logger.info(f"  🌐 Source: PIXABAY")
            
            page = 1
            consecutive_failures = 0
            max_pages = 5  # Limiter les pages
            
            while query_downloaded < images_per_query and page <= max_pages:
                # Rechercher
                images = self.downloader.search_images(query, page=page)
                
                if not images:
                    consecutive_failures += 1
                    if consecutive_failures >= 2:
                        self.logger.info(f"    ⚠️ Pas de résultats, fin pour cette requête")
                        break
                    continue
                
                consecutive_failures = 0
                self.logger.info(f"    📄 Page {page}: {len(images)} images trouvées")
                
                # Télécharger
                for img_data in images:
                    if query_downloaded >= images_per_query:
                        break
                    
                    # Delay anti-blocage entre téléchargements
                    smart_delay(Config.DELAYS['download'])
                    
                    filename = safe_filename(
                        img_data['url'],
                        prefix=f"pixabay_{category}"
                    )
                    output_path = os.path.join(output_dir, filename)
                    
                    # Télécharger
                    success, message = download_image(
                        img_data['url'],
                        output_path,
                        self.session
                    )
                    
                    if success:
                        # Vérifier doublons
                        try:
                            with open(output_path, 'rb') as f:
                                img_hash = calculate_hash(f.read())
                            
                            if img_hash in self.downloaded_hashes:
                                os.remove(output_path)
                                self.stats[category]['duplicates'] += 1
                                continue
                            
                            self.downloaded_hashes.add(img_hash)
                            query_downloaded += 1
                            total_downloaded += 1
                            self.stats[category]['success'] += 1
                            
                            self.logger.info(
                                f"    ✅ [{total_downloaded}/{max_images}] {filename}"
                            )
                            
                        except Exception as e:
                            self.logger.error(f"    ❌ Erreur hash: {e}")
                            if os.path.exists(output_path):
                                os.remove(output_path)
                            self.stats[category]['errors'] += 1
                    else:
                        self.stats[category]['failed'] += 1
                        if "429" in message or "rate limit" in message.lower():
                            self.logger.warning(f"    ⏳ Rate limit détecté, pause...")
                            time.sleep(30)
                
                page += 1
            
            self.logger.info(
                f"  ✓ {query_downloaded} images de '{query}' depuis Pixabay"
            )
            
            # Pause entre queries
            if total_downloaded < max_images:
                smart_delay((3.0, 6.0))
        
        self.logger.info(f"\n✅ TOTAL {category}: {total_downloaded} images")
        return total_downloaded
    
    def download_all(self):
        """Télécharge toutes les catégories"""
        self.logger.info("="*70)
        self.logger.info("🚀 TÉLÉCHARGEMENT PIXABAY OPTIMISÉ ANTI-BLOCAGE")
        self.logger.info("="*70)
        self.logger.info(f"📁 Sortie: {Config.OUTPUT_DIR}")
        self.logger.info(f"🎯 Limite: {Config.MAX_IMAGES_PER_SOURCE} images/catégorie")
        
        # Vérifier clé API
        if not Config.PIXABAY_API_KEY or Config.PIXABAY_API_KEY == "VOTRE_CLE_PIXABAY":
            self.logger.error("\n❌ CLÉ API PIXABAY NON CONFIGURÉE!")
            self.logger.info("Ajoutez votre clé API gratuite:")
            self.logger.info("  • Pixabay: https://pixabay.com/api/docs/")
            return
        
        self.logger.info(f"✅ Source active: Pixabay (5000 req/heure)")
        self.logger.info(f"⏱️  Délais optimisés pour éviter tout blocage")
        
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        
        # Télécharger
        total_stats = {}
        start_time = time.time()
        
        for category, queries in Config.SEARCH_QUERIES.items():
            downloaded = self.download_for_category(
                category,
                queries,
                Config.MAX_IMAGES_PER_SOURCE
            )
            total_stats[category] = downloaded
        
        # Résumé
        elapsed = time.time() - start_time
        
        self.logger.info("\n" + "="*70)
        self.logger.info("✅ TÉLÉCHARGEMENT TERMINÉ")
        self.logger.info("="*70)
        self.logger.info(f"⏱️  Durée: {elapsed/60:.1f} minutes ({elapsed/3600:.2f}h)")
        self.logger.info(f"\n📊 STATISTIQUES PAR CATÉGORIE:")
        
        for category, count in total_stats.items():
            stats = self.stats[category]
            self.logger.info(
                f"  • {category:15s}: {count:4d} images "
                f"(succès: {stats['success']}, échecs: {stats['failed']}, "
                f"doublons: {stats['duplicates']})"
            )
        
        total = sum(total_stats.values())
        self.logger.info(f"\n🎯 TOTAL GÉNÉRAL: {total} images téléchargées")
        self.logger.info(f"📁 Dossier: {Config.OUTPUT_DIR}")
        self.logger.info(f"⚡ Vitesse moyenne: {total/(elapsed/60):.1f} images/minute")
        self.logger.info("="*70)

# ======================== MAIN ========================
def main():
    """Point d'entrée principal"""
    try:
        manager = ImageDownloadManager()
        manager.download_all()
    except KeyboardInterrupt:
        logging.info("\n\n⚠️ Interruption par l'utilisateur")
    except Exception as e:
        logging.error(f"\n\n❌ ERREUR FATALE: {e}", exc_info=True)

if __name__ == '__main__':
    main()