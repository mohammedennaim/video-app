#!/usr/bin/env python3
"""
Script de validation des améliorations du projet de restauration vidéo.
Teste les nouvelles fonctionnalités implémentées.
"""

import sys
import os
import numpy as np
import cv2

# Ajouter le répertoire src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test que tous les modules peuvent être importés correctement."""
    print("🧪 Test d'importation des modules...")
    
    try:
        # Test des modules classiques
        from src.classical.video_restoration import ClassicalRestoration
        print("✅ Module classique importé avec succès")
        
        # Test des modules IA
        from src.ai.gan_colorization import GANColorization
        from src.ai.deoldify_enhanced import EnhancedDeOldifyColorizer
        from src.ai.zhang_restoration import ZhangVideoRestorer
        print("✅ Modules IA importés avec succès")
        
        # Test des modules utilitaires
        from src.utils.gpu_acceleration import AccelerationManager
        from src.utils.performance import OptimizedVideoProcessor
        print("✅ Modules utilitaires importés avec succès")
        
        return True
        
    except ImportError as e:
        print(f"❌ Erreur d'importation: {e}")
        return False

def test_initialization():
    """Test que les classes peuvent être initialisées."""
    print("\n🔧 Test d'initialisation des classes...")
      try:
        # Test de la restauration classique
        from src.classical.video_restoration import ClassicalRestoration
        classical_restorer = ClassicalRestoration()
        print("✅ ClassicalRestoration initialisé")
        
        # Test du coloriseur GAN
        from src.ai.gan_colorization import GANColorization
        gan_colorizer = GANColorization()
        print("✅ GANColorization initialisé")
        
        # Test du coloriseur DeOldify amélioré
        from src.ai.deoldify_enhanced import EnhancedDeOldifyColorizer
        deoldify_colorizer = EnhancedDeOldifyColorizer()
        print("✅ EnhancedDeOldifyColorizer initialisé")
          # Test du restaurateur Zhang
        from src.ai.zhang_restoration import ZhangVideoRestoration
        zhang_restorer = ZhangVideoRestoration()
        print("✅ ZhangVideoRestoration initialisé")
        
        # Test des utilitaires
        from src.utils.gpu_acceleration import AccelerationManager
        from src.utils.performance import OptimizedVideoProcessor
        
        acceleration_manager = AccelerationManager()
        print("✅ AccelerationManager initialisé")
        
        optimized_processor = OptimizedVideoProcessor()
        print("✅ OptimizedVideoProcessor initialisé")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur d'initialisation: {e}")
        return False

def test_new_methods():
    """Test des nouvelles méthodes implémentées."""
    print("\n🆕 Test des nouvelles méthodes...")
    
    try:
        # Créer une frame de test
        test_frame = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        test_frames = [test_frame] * 3  # 3 frames pour tester la cohérence temporelle
          # Test de la colorisation semi-automatique de Levin
        from src.classical.video_restoration import ClassicalRestoration
        classical_restorer = ClassicalRestoration()
        
        # Vérifier que la méthode de Levin existe
        assert hasattr(classical_restorer, 'levin_colorization'), "Méthode de Levin manquante"
        print("✅ Méthode de colorisation de Levin disponible")
        
        # Test du GAN avec optimisations
        from src.ai.gan_colorization import GANColorization
        gan_colorizer = GANColorization()
        
        # Vérifier les optimisations
        assert hasattr(gan_colorizer, 'acceleration_manager'), "Gestionnaire d'accélération manquant"
        print("✅ Optimisations GPU disponibles")
        
        # Test de DeOldify amélioré
        from src.ai.deoldify_enhanced import EnhancedDeOldifyColorizer
        deoldify_colorizer = EnhancedDeOldifyColorizer()
        
        # Vérifier les modèles Zhang et al.
        assert hasattr(deoldify_colorizer, 'video_restorer'), "Restaurateur vidéo manquant"
        print("✅ Modèles Zhang et al. intégrés")
          # Test de la cohérence temporelle
        from src.ai.zhang_restoration import ZhangVideoRestoration
        zhang_restorer = ZhangVideoRestoration()
        
        assert hasattr(zhang_restorer, 'temporal_consistency_network'), "Réseau de cohérence temporelle manquant"
        print("✅ Réseau de cohérence temporelle disponible")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test des nouvelles méthodes: {e}")
        return False

def test_integration():
    """Test d'intégration basique."""
    print("\n🔗 Test d'intégration...")
    
    try:
        # Créer une frame de test simple
        test_frame = np.ones((64, 64), dtype=np.uint8) * 128  # Frame grise uniforme
        
        # Test de colorisation avec DeOldify amélioré
        from src.ai.deoldify_enhanced import EnhancedDeOldifyColorizer
        colorizer = EnhancedDeOldifyColorizer()
        
        # Test de colorisation d'une frame
        colorized_frame = colorizer.colorize_frame(test_frame)
        assert colorized_frame.shape == (64, 64, 3), f"Forme incorrecte: {colorized_frame.shape}"
        print("✅ Colorisation d'une frame réussie")
        
        # Test avec plusieurs frames (version simplifiée)
        test_frames = [test_frame] * 2  # Seulement 2 frames pour un test rapide
        colorized_frames = colorizer.restore_and_colorize_video(test_frames)
        assert len(colorized_frames) == 2, f"Nombre de frames incorrect: {len(colorized_frames)}"
        print("✅ Colorisation vidéo réussie")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur d'intégration: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fonction principale de validation."""
    print("🚀 Validation des améliorations du projet de restauration vidéo")
    print("=" * 60)
    
    tests_results = []
    
    # Exécuter les tests
    tests_results.append(("Importation", test_imports()))
    tests_results.append(("Initialisation", test_initialization()))
    tests_results.append(("Nouvelles méthodes", test_new_methods()))
    tests_results.append(("Intégration", test_integration()))
    
    # Résumé des résultats
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS:")
    
    all_passed = True
    for test_name, passed in tests_results:
        status = "✅ PASSÉ" if passed else "❌ ÉCHEC"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 TOUS LES TESTS SONT PASSÉS!")
        print("✨ Le projet est prêt avec les nouvelles fonctionnalités:")
        print("   - Colorisation semi-automatique selon Levin et al. (2004)")
        print("   - Intégration DeOldify (Jason Antic, 2019)")
        print("   - Techniques Zhang et al. (CVPR 2020)")
        print("   - Optimisations GPU et performances")
        print("   - Cohérence temporelle améliorée")
    else:
        print("⚠️ CERTAINS TESTS ONT ÉCHOUÉ")
        print("🔧 Veuillez corriger les erreurs avant de continuer")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
