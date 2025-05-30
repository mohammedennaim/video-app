#!/usr/bin/env python3
"""
Script de validation simple des améliorations du projet de restauration vidéo.
"""

import sys
import os
import numpy as np

# Ajouter le répertoire src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test des imports basiques."""
    print("🧪 Test d'importation des modules...")
    
    try:
        # Test classique
        from src.classical.video_restoration import ClassicalRestoration
        print("✅ Module classique OK")
        
        # Test GAN
        from src.ai.gan_colorization import GANColorization
        print("✅ Module GAN OK")
        
        # Test DeOldify
        from src.ai.deoldify_enhanced import EnhancedDeOldifyColorizer
        print("✅ Module DeOldify OK")
        
        # Test Zhang
        from src.ai.zhang_restoration import ZhangVideoRestoration
        print("✅ Module Zhang OK")
        
        # Test utilitaires
        from src.utils.gpu_acceleration import AccelerationManager
        from src.utils.performance import OptimizedVideoProcessor
        print("✅ Modules utilitaires OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur d'import: {e}")
        return False

def test_initialization():
    """Test d'initialisation des classes."""
    print("\n🔧 Test d'initialisation...")
    
    try:
        from src.classical.video_restoration import ClassicalRestoration
        from src.ai.gan_colorization import GANColorization
        from src.ai.deoldify_enhanced import EnhancedDeOldifyColorizer
        from src.ai.zhang_restoration import ZhangVideoRestoration
        
        # Initialiser les classes
        classical = ClassicalRestoration()
        gan = GANColorization()
        deoldify = EnhancedDeOldifyColorizer()
        zhang = ZhangVideoRestoration()
        
        print("✅ Toutes les classes initialisées")
        return True
        
    except Exception as e:
        print(f"❌ Erreur d'initialisation: {e}")
        return False

def test_simple_colorization():
    """Test de colorisation simple."""
    print("\n🎨 Test de colorisation...")
    
    try:
        from src.ai.deoldify_enhanced import EnhancedDeOldifyColorizer
        
        # Créer une image de test simple
        test_image = np.ones((32, 32), dtype=np.uint8) * 128
        
        # Initialiser le coloriseur
        colorizer = EnhancedDeOldifyColorizer()
        
        # Test de colorisation
        result = colorizer.colorize_frame(test_image)
        
        # Vérifier le résultat
        if result.shape == (32, 32, 3):
            print("✅ Colorisation réussie")
            return True
        else:
            print(f"❌ Forme incorrecte: {result.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur de colorisation: {e}")
        return False

def main():
    """Fonction principale."""
    print("🚀 Validation des améliorations - Version Simplifiée")
    print("=" * 50)
    
    # Tests
    tests = [
        ("Imports", test_basic_imports),
        ("Initialisation", test_initialization),
        ("Colorisation", test_simple_colorization)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # Résumé
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSÉ" if passed else "❌ ÉCHEC"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 VALIDATION RÉUSSIE!")
        print("✨ Fonctionnalités disponibles:")
        print("   - Restauration classique (Levin et al.)")
        print("   - Colorisation GAN améliorée")
        print("   - DeOldify avec optimisations")
        print("   - Techniques Zhang et al. (CVPR 2020)")
    else:
        print("⚠️ CERTAINS TESTS ONT ÉCHOUÉ")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
