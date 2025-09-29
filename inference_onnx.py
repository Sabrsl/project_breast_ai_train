#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'Inférence avec Modèle ONNX BreastAI
Utilise un modèle ONNX exporté pour faire des prédictions
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import cv2

# Vérifier la disponibilité d'ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    print("❌ ONNX Runtime n'est pas installé!")
    print("   Installez avec: pip install onnxruntime")
    ONNX_AVAILABLE = False
    sys.exit(1)


class BreastAIInference:
    """Classe d'inférence pour modèles BreastAI ONNX"""
    
    def __init__(self, model_path: str, config_path: str = "config.json"):
        """
        Initialise le système d'inférence
        
        Args:
            model_path: Chemin vers le modèle ONNX
            config_path: Chemin vers config.json
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modèle ONNX introuvable: {model_path}")
        
        # Charger la configuration
        self.config = self._load_config()
        
        # Charger le modèle ONNX
        self.session = self._load_model()
        
        # Paramètres
        self.image_size = self.config.get('data', {}).get('image_size', 512)
        self.class_names = self.config.get('data', {}).get('class_names', ['benign', 'malignant'])
        self.normalization_mean = self.config.get('data', {}).get('preprocessing', {}).get('normalization', {}).get('mean', [0.485, 0.456, 0.406])
        self.normalization_std = self.config.get('data', {}).get('preprocessing', {}).get('normalization', {}).get('std', [0.229, 0.224, 0.225])
        self.use_clahe = self.config.get('data', {}).get('preprocessing', {}).get('use_clahe', True)
        
        print(f"✅ Modèle chargé: {self.model_path.name}")
        print(f"   Classes: {self.class_names}")
        print(f"   Taille image: {self.image_size}x{self.image_size}")
    
    def _load_config(self) -> Dict:
        """Charge la configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"⚠️  Config non trouvée, utilisation des valeurs par défaut")
            return {}
    
    def _load_model(self):
        """Charge le modèle ONNX"""
        # Configuration de la session ONNX
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Créer la session
        providers = ['CPUExecutionProvider']  # CPU uniquement
        session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers
        )
        
        return session
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Applique CLAHE pour améliorer le contraste"""
        try:
            # Convertir en LAB
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # CLAHE sur canal L
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Reconvertir en RGB
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return image
        except Exception as e:
            print(f"⚠️  CLAHE échoué: {e}, image originale utilisée")
            return image
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Prétraite une image pour l'inférence
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Tensor numpy prêt pour l'inférence [1, 3, H, W]
        """
        # Charger l'image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Appliquer CLAHE si activé
        if self.use_clahe:
            image_np = self.apply_clahe(image_np)
        
        # Redimensionner
        image_resized = cv2.resize(
            image_np,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normaliser [0, 255] -> [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Normaliser avec mean et std ImageNet
        mean = np.array(self.normalization_mean, dtype=np.float32)
        std = np.array(self.normalization_std, dtype=np.float32)
        image_normalized = (image_normalized - mean) / std
        
        # Transposer [H, W, C] -> [C, H, W]
        image_transposed = np.transpose(image_normalized, (2, 0, 1))
        
        # Ajouter dimension batch [C, H, W] -> [1, C, H, W]
        image_batch = np.expand_dims(image_transposed, axis=0)
        
        return image_batch
    
    def predict(self, image_path: str, return_probabilities: bool = True) -> Dict:
        """
        Fait une prédiction sur une image
        
        Args:
            image_path: Chemin vers l'image
            return_probabilities: Si True, retourne les probabilités
            
        Returns:
            Dictionnaire avec prédiction et probabilités
        """
        # Prétraiter l'image
        input_tensor = self.preprocess_image(image_path)
        
        # Obtenir le nom de l'input du modèle
        input_name = self.session.get_inputs()[0].name
        
        # Inférence
        outputs = self.session.run(None, {input_name: input_tensor})
        logits = outputs[0][0]  # [num_classes]
        
        # Softmax pour obtenir les probabilités
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Prédiction (classe avec probabilité max)
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.class_names[predicted_class_idx] if predicted_class_idx < len(self.class_names) else f"class_{predicted_class_idx}"
        confidence = float(probabilities[predicted_class_idx])
        
        result = {
            'image': str(image_path),
            'predicted_class': predicted_class,
            'predicted_class_index': int(predicted_class_idx),
            'confidence': confidence
        }
        
        if return_probabilities:
            result['probabilities'] = {
                class_name: float(probabilities[i])
                for i, class_name in enumerate(self.class_names)
            }
        
        return result
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Fait des prédictions sur un batch d'images
        
        Args:
            image_paths: Liste de chemins vers les images
            
        Returns:
            Liste de dictionnaires avec prédictions
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                print(f"❌ Erreur pour {image_path}: {e}")
                results.append({
                    'image': str(image_path),
                    'error': str(e)
                })
        
        return results
    
    def predict_directory(self, directory: str, extensions: List[str] = None) -> List[Dict]:
        """
        Fait des prédictions sur toutes les images d'un répertoire
        
        Args:
            directory: Chemin vers le répertoire
            extensions: Extensions d'images à traiter (par défaut: jpg, png)
            
        Returns:
            Liste de dictionnaires avec prédictions
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Répertoire introuvable: {directory}")
        
        # Trouver toutes les images
        image_paths = []
        for ext in extensions:
            image_paths.extend(dir_path.glob(f"*{ext}"))
            image_paths.extend(dir_path.glob(f"*{ext.upper()}"))
        
        print(f"📁 Traitement de {len(image_paths)} images depuis {directory}")
        
        return self.predict_batch([str(p) for p in image_paths])
    
    def print_prediction(self, result: Dict):
        """Affiche une prédiction de manière formatée"""
        if 'error' in result:
            print(f"\n❌ {result['image']}")
            print(f"   Erreur: {result['error']}")
            return
        
        print(f"\n📸 Image: {result['image']}")
        print(f"   Prédiction: {result['predicted_class']}")
        print(f"   Confiance: {result['confidence']:.2%}")
        
        if 'probabilities' in result:
            print("   Probabilités détaillées:")
            for class_name, prob in result['probabilities'].items():
                bar_length = int(prob * 30)
                bar = "█" * bar_length + "░" * (30 - bar_length)
                print(f"      {class_name:15s} {bar} {prob:.2%}")


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="Inférence avec modèle ONNX BreastAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  # Prédiction sur une seule image
  python inference_onnx.py --model exports/onnx/model.onnx --image test.jpg
  
  # Prédiction sur un répertoire
  python inference_onnx.py --model exports/onnx/model.onnx --directory data/test/
  
  # Sauvegarder les résultats en JSON
  python inference_onnx.py --model exports/onnx/model.onnx --directory data/test/ --output results.json
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Chemin vers le modèle ONNX'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Chemin vers config.json (défaut: config.json)'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Chemin vers une image à classifier'
    )
    
    parser.add_argument(
        '--directory',
        type=str,
        help='Chemin vers un répertoire d\'images'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Chemin pour sauvegarder les résultats en JSON'
    )
    
    args = parser.parse_args()
    
    # Vérifier qu'au moins une source est fournie
    if not args.image and not args.directory:
        parser.error("Vous devez fournir --image ou --directory")
    
    # Initialiser l'inférence
    print("\n" + "=" * 80)
    print("  BREASTAI INFERENCE SYSTEM - ONNX Runtime")
    print("=" * 80)
    
    try:
        inference = BreastAIInference(args.model, args.config)
    except Exception as e:
        print(f"\n❌ Erreur d'initialisation: {e}")
        return 1
    
    # Faire les prédictions
    results = []
    
    try:
        if args.image:
            # Prédiction sur une seule image
            print(f"\n🔍 Analyse de l'image...")
            result = inference.predict(args.image)
            results = [result]
            inference.print_prediction(result)
        
        elif args.directory:
            # Prédiction sur un répertoire
            results = inference.predict_directory(args.directory)
            
            # Afficher les résultats
            for result in results:
                inference.print_prediction(result)
            
            # Statistiques
            successful = [r for r in results if 'error' not in r]
            failed = [r for r in results if 'error' in r]
            
            print("\n" + "=" * 80)
            print(f"  RÉSUMÉ")
            print("=" * 80)
            print(f"   Total: {len(results)} images")
            print(f"   ✅ Succès: {len(successful)}")
            print(f"   ❌ Erreurs: {len(failed)}")
            
            if successful:
                # Statistiques par classe
                class_counts = {}
                for result in successful:
                    predicted_class = result['predicted_class']
                    class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
                
                print(f"\n   Répartition des prédictions:")
                for class_name, count in class_counts.items():
                    percentage = (count / len(successful)) * 100
                    print(f"      {class_name:15s}: {count:4d} ({percentage:.1f}%)")
                
                # Confiance moyenne
                avg_confidence = np.mean([r['confidence'] for r in successful])
                print(f"\n   Confiance moyenne: {avg_confidence:.2%}")
        
        # Sauvegarder les résultats si demandé
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Résultats sauvegardés: {output_path}")
        
        print("\n" + "=" * 80)
        print("  ✅ INFÉRENCE TERMINÉE")
        print("=" * 80 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Erreur durant l'inférence: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    if not ONNX_AVAILABLE:
        sys.exit(1)
    
    sys.exit(main())
