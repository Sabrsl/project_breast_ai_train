#!/usr/bin/env python3
"""
Script de nettoyage clinique OBLIGATOIRE
Doit être exécuté avant chaque entraînement pour garantir l'intégrité des données
"""

import os
import cv2
import numpy as np
from pathlib import Path
import logging
import sys

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/clinical_cleanup.log')
    ]
)
logger = logging.getLogger(__name__)

class ClinicalDatasetCleaner:
    """Nettoyeur clinique strict - ZERO tolérance pour images corrompues"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.corrupted_files = []
        self.stats = {
            'total_scanned': 0,
            'corrupted_found': 0,
            'deleted': 0,
            'errors': 0
        }
        
        # Créer le dossier logs
        Path('logs').mkdir(exist_ok=True)
    
    def validate_image_clinical(self, img_path: Path) -> tuple:
        """
        Validation clinique STRICTE d'une image
        Returns: (is_valid, reason)
        """
        try:
            # Test 1: Existence
            if not img_path.exists():
                return False, "Fichier inexistant"
            
            # Test 2: Taille minimale
            if img_path.stat().st_size < 1024:
                return False, f"Fichier trop petit ({img_path.stat().st_size} bytes)"
            
            # Test 3: Lecture OpenCV
            image = cv2.imread(str(img_path))
            if image is None:
                return False, "Lecture impossible"
            
            # Test 4: Dimensions minimales
            if image.shape[0] < 50 or image.shape[1] < 50:
                return False, f"Dimensions trop petites ({image.shape[0]}x{image.shape[1]})"
            
            # Test 5: Image noire (sauf masques)
            if np.all(image == 0) and "_mask" not in img_path.name.lower():
                return False, "Image entièrement noire"
            
            # Test 6: Variance minimale
            if image.var() < 10 and "_mask" not in img_path.name.lower():
                return False, f"Variance trop faible ({image.var():.1f})"
            
            # Test 7: Format correct
            if len(image.shape) != 3 or image.shape[2] != 3:
                return False, f"Format invalide (shape: {image.shape})"
            
            # Test 8: Pas de NaN/Inf
            if np.isnan(image).any():
                return False, "Contient des valeurs NaN"
            if np.isinf(image).any():
                return False, "Contient des valeurs infinies"
            
            # Test 9: Pas de fichiers de test
            if any(keyword in img_path.name.lower() for keyword in ['test', 'placeholder', 'dummy', 'example']):
                return False, "Fichier de test/placeholder"
            
            return True, "OK"
            
        except Exception as e:
            return False, f"Erreur validation: {str(e)}"
    
    def clean_split(self, split_dir: Path) -> int:
        """Nettoie un split du dataset"""
        corrupted_count = 0
        
        if not split_dir.exists():
            logger.warning(f"Répertoire introuvable: {split_dir}")
            return corrupted_count
        
        logger.info(f"Validation clinique du split: {split_dir}")
        
        for img_path in split_dir.rglob('*'):
            if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                self.stats['total_scanned'] += 1
                
                is_valid, reason = self.validate_image_clinical(img_path)
                
                if not is_valid:
                    self.stats['corrupted_found'] += 1
                    corrupted_count += 1
                    
                    logger.critical(f"IMAGE CORROMPUE DETECTEE: {img_path.name} - {reason}")
                    
                    # Supprimer immédiatement
                    try:
                        img_path.unlink()
                        self.stats['deleted'] += 1
                        logger.info(f"SUPPRIME: {img_path.name}")
                    except Exception as e:
                        self.stats['errors'] += 1
                        logger.error(f"Erreur suppression {img_path.name}: {e}")
        
        return corrupted_count
    
    def clean_dataset(self) -> bool:
        """Nettoie tout le dataset - OBLIGATOIRE avant entraînement"""
        logger.critical("="*80)
        logger.critical("NETTOYAGE CLINIQUE OBLIGATOIRE")
        logger.critical("="*80)
        logger.critical("ZERO TOLERANCE pour images corrompues en contexte medical")
        
        if not self.data_dir.exists():
            logger.error(f"Dataset introuvable: {self.data_dir}")
            return False
        
        total_corrupted = 0
        
        # Nettoyer tous les splits
        for split_name in ['train', 'val', 'test']:
            split_path = self.data_dir / split_name
            if split_path.exists():
                corrupted = self.clean_split(split_path)
                total_corrupted += corrupted
                logger.info(f"Split {split_name}: {corrupted} images corrompues supprimees")
        
        # Rapport final
        logger.critical("="*80)
        logger.critical("RAPPORT DE NETTOYAGE CLINIQUE")
        logger.critical("="*80)
        logger.critical(f"Images scannees: {self.stats['total_scanned']}")
        logger.critical(f"Images corrompues trouvees: {self.stats['corrupted_found']}")
        logger.critical(f"Images supprimees: {self.stats['deleted']}")
        logger.critical(f"Erreurs: {self.stats['errors']}")
        
        if self.stats['corrupted_found'] > 0:
            corruption_rate = (self.stats['corrupted_found'] / self.stats['total_scanned']) * 100
            logger.critical(f"Taux de corruption: {corruption_rate:.2f}%")
            
            if corruption_rate > 5:
                logger.critical(f"ATTENTION: Taux de corruption eleve ({corruption_rate:.2f}%)")
                logger.critical("Verifiez la qualite de votre dataset avant l'entrainement")
            else:
                logger.info(f"Taux de corruption acceptable ({corruption_rate:.2f}%)")
        else:
            logger.info("AUCUNE IMAGE CORROMPUE - Dataset clinique valide")
        
        logger.critical("="*80)
        
        # Retourner True seulement si aucune corruption
        return self.stats['corrupted_found'] == 0

def main():
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "data"
    
    logger.critical(f"NETTOYAGE CLINIQUE OBLIGATOIRE du dataset: {data_dir}")
    
    cleaner = ClinicalDatasetCleaner(data_dir)
    success = cleaner.clean_dataset()
    
    if success:
        logger.critical("[OK] DATASET CLINIQUE VALIDE - Entrainement autorise")
        sys.exit(0)
    else:
        logger.critical("[ERREUR] DATASET CORROMPU - Entrainement INTERDIT")
        logger.critical("Corrigez les images corrompues avant de relancer l'entrainement")
        sys.exit(1)

if __name__ == '__main__':
    main()
