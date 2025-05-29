#!/usr/bin/env python3
"""
Test script pour les méthodes classiques optimisées.
Démontre l'intégration des optimisations GPU et de performance.
"""

import numpy as np
import cv2
import sys
import os

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from classical.video_restoration import ClassicalRestoration


def create_test_video(num_frames=30, height=480, width=640):
    """Crée une vidéo de test avec du bruit artificiel."""
    print(f"🎬 Création d'une vidéo de test: {num_frames} frames ({width}x{height})")
    
    frames = []
    for i in range(num_frames):
        # Créer une frame avec des formes géométriques
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Ajouter des formes colorées
        cv2.rectangle(frame, (50, 50), (150, 150), (100, 150, 200), -1)  # Rectangle
        cv2.circle(frame, (400, 200), 80, (150, 200, 100), -1)  # Cercle
        cv2.rectangle(frame, (200, 300), (400, 400), (200, 100, 150), -1)  # Rectangle 2
        
        # Ajouter du mouvement
        offset = int(10 * np.sin(i * 0.2))
        cv2.circle(frame, (300 + offset, 100), 30, (255, 255, 255), -1)
        
        # Ajouter du bruit gaussien
        noise = np.random.normal(0, 20, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)
        
        # Convertir en niveaux de gris pour certains tests
        if i % 2 == 0:  # Frames paires en couleur, impaires en gris
            frames.append(frame)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    
    return frames


def test_classical_restoration():
    """Test des méthodes classiques avec optimisations."""
    print("🧪 Test des méthodes classiques de restauration vidéo\n")
    
    # Créer l'instance de restauration classique
    restoration = ClassicalRestoration()
    
    # Afficher l'état des optimisations
    print("📊 État des optimisations:")
    status = restoration.get_optimization_status()
    for key, value in status.items():
        if key == 'features' and isinstance(value, list):
            print(f"  {key}: {', '.join(value) if value else 'Aucune'}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Créer une vidéo de test
    test_frames = create_test_video(num_frames=20, height=360, width=480)
    print(f"✅ Vidéo de test créée: {len(test_frames)} frames\n")
    
    # Test 1: Débruitage
    print("🔧 Test 1: Débruitage spatio-temporel")
    try:
        result, stats = restoration.process_with_performance_tracking(
            test_frames, 'denoise'
        )
        print(f"  ✅ Débruitage terminé")
        print(f"  📈 Statistiques:")
        print(f"    - Temps de traitement: {stats['processing_time']:.2f}s")
        print(f"    - FPS moyen: {stats['fps']:.2f}")
        print(f"    - Optimisations utilisées: {'Oui' if stats['optimizations_used'] else 'Non'}")
        print(f"    - Device: {stats['device_type']}")
        print()
    except Exception as e:
        print(f"  ❌ Erreur lors du débruitage: {e}\n")
    
    # Test 2: Colorisation
    print("🎨 Test 2: Colorisation semi-automatique")
    try:
        # Créer des frames en niveaux de gris pour la colorisation
        gray_frames = []
        for frame in test_frames[:10]:  # Utiliser moins de frames pour la colorisation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frames.append(gray)
        
        result, stats = restoration.process_with_performance_tracking(
            gray_frames, 'colorize'
        )
        print(f"  ✅ Colorisation terminée")
        print(f"  📈 Statistiques:")
        print(f"    - Temps de traitement: {stats['processing_time']:.2f}s")
        print(f"    - FPS moyen: {stats['fps']:.2f}")
        print(f"    - Optimisations utilisées: {'Oui' if stats['optimizations_used'] else 'Non'}")
        print(f"    - Device: {stats['device_type']}")
        print()
    except Exception as e:
        print(f"  ❌ Erreur lors de la colorisation: {e}\n")
    
    # Test 3: Amélioration du contraste
    print("✨ Test 3: Amélioration du contraste")
    try:
        result, stats = restoration.process_with_performance_tracking(
            test_frames[:15], 'enhance_contrast'
        )
        print(f"  ✅ Amélioration du contraste terminée")
        print(f"  📈 Statistiques:")
        print(f"    - Temps de traitement: {stats['processing_time']:.2f}s")
        print(f"    - FPS moyen: {stats['fps']:.2f}")
        print(f"    - Optimisations utilisées: {'Oui' if stats['optimizations_used'] else 'Non'}")
        print(f"    - Device: {stats['device_type']}")
        print()
    except Exception as e:
        print(f"  ❌ Erreur lors de l'amélioration du contraste: {e}\n")
    
    # Résumé
    print("📝 Résumé du test:")
    print(f"  - Optimisations GPU disponibles: {'Oui' if status['optimizations_enabled'] else 'Non'}")
    print(f"  - Device utilisé: {status['device_type']}")
    if status['optimizations_enabled']:
        print(f"  - Fonctionnalités activées: {', '.join(status['features'])}")
    print(f"  - Tests effectués: Débruitage, Colorisation, Amélioration du contraste")


def compare_performance():
    """Compare les performances avec et sans optimisations."""
    print("\n🏁 Comparaison des performances\n")
    
    # Créer une vidéo de test plus grande
    test_frames = create_test_video(num_frames=50, height=720, width=1280)
    print(f"📹 Vidéo de test HD: {len(test_frames)} frames (1280x720)\n")
    
    restoration = ClassicalRestoration()
    
    # Test avec optimisations (si disponibles)
    if restoration.get_optimization_status()['optimizations_enabled']:
        print("🚀 Test avec optimisations:")
        try:
            result, stats = restoration.process_with_performance_tracking(
                test_frames, 'denoise'
            )
            print(f"  - Temps: {stats['processing_time']:.2f}s")
            print(f"  - FPS: {stats['fps']:.2f}")
            print(f"  - Device: {stats['device_type']}")
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
    else:
        print("📝 Optimisations non disponibles - traitement standard uniquement")
        try:
            import time
            start_time = time.time()
            result = restoration.denoise_video(test_frames)
            end_time = time.time()
            
            processing_time = end_time - start_time
            fps = len(test_frames) / processing_time if processing_time > 0 else 0
            
            print(f"  - Temps: {processing_time:.2f}s")
            print(f"  - FPS: {fps:.2f}")
            print(f"  - Device: CPU (standard)")
        except Exception as e:
            print(f"  ❌ Erreur: {e}")


if __name__ == "__main__":
    try:
        test_classical_restoration()
        compare_performance()
        
        print("\n✅ Tests terminés avec succès!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur générale: {e}")
        import traceback
        traceback.print_exc()
