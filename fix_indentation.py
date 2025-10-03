#!/usr/bin/env python3
"""
Script pour corriger automatiquement les erreurs d'indentation
"""

import ast
import sys

def check_syntax(filename):
    """Vérifie la syntaxe d'un fichier Python"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Compiler le code pour détecter les erreurs de syntaxe
        compile(source, filename, 'exec')
        print(f"[OK] {filename}: Syntaxe correcte")
        return True
        
    except SyntaxError as e:
        print(f"[ERROR] {filename}: Erreur de syntaxe a la ligne {e.lineno}")
        print(f"   Message: {e.msg}")
        print(f"   Texte: {e.text}")
        return False
        
    except IndentationError as e:
        print(f"[ERROR] {filename}: Erreur d'indentation a la ligne {e.lineno}")
        print(f"   Message: {e.msg}")
        print(f"   Texte: {e.text}")
        return False
        
    except Exception as e:
        print(f"[ERROR] {filename}: Erreur inattendue: {e}")
        return False

if __name__ == '__main__':
    filename = 'breastai_training.py'
    success = check_syntax(filename)
    
    if not success:
        print("\n🔧 Actions recommandées:")
        print("1. Vérifier manuellement les lignes mentionnées")
        print("2. Corriger l'indentation (espaces vs tabs)")
        print("3. S'assurer que tous les blocs if/for/def sont correctement indentés")
        sys.exit(1)
    else:
        print(f"\n[SUCCESS] {filename} est syntaxiquement correct!")
        sys.exit(0)
