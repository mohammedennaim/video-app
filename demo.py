#!/usr/bin/env python3
"""
Script de démonstration pour la restauration et colorisation de vidéos.
Génère des données d'exemple et teste les différentes méthodes.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.append(str(Path(__file__).parent / 'src'))

from classical.video_restoration import ClassicalRestoration
from ai.gan_colorization import SimpleColorization
from evaluation.metrics import VideoMetrics, QualitativeEvaluation
from utils.video_utils import VideoProcessor

def create_demo_video(output_path: str, duration: int = 5, fps: int = 30) -> str:
    """
    Crée une vidéo de démonstration simulant un film ancien.
    
    Args:
        output_path: Chemin de sortie
        duration: Durée en secondes
        fps: Images par seconde
        
    Returns:
        Chemin de la vidéo créée
    """
    print("🎬 Création d'une vidéo de démonstration...")
    
    width, height = 640, 480
    total_frames = duration * fps
    
    # Créer le répertoire de sortie
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Configuration du codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_idx in range(total_frames):
        # Créer une frame avec du contenu synthétique
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Fond dégradé
        for y in range(height):
            intensity = int(50 + (200 * y / height))
            frame[y, :] = [intensity, intensity * 0.8, intensity * 0.6]
        
        # Ajouter des formes géométriques mobiles
        t = frame_idx / total_frames
        
        # Cercle mobile
        center_x = int(width * (0.2 + 0.6 * abs(np.sin(t * 2 * np.pi))))
        center_y = int(height * 0.3)
        cv2.circle(frame, (center_x, center_y), 50, (100, 150, 200), -1)
        
        # Rectangle mobile
        rect_x = int(width * (0.8 - 0.6 * abs(np.cos(t * 1.5 * np.pi))))
        rect_y = int(height * 0.6)
        cv2.rectangle(frame, (rect_x-30, rect_y-20), (rect_x+30, rect_y+20), (200, 100, 50), -1)
        
        # Ajouter du bruit pour simuler un film ancien
        noise = np.random.normal(0, 25, (height, width, 3)).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Ajouter des lignes verticales aléatoires (défauts de film)
        if np.random.random() < 0.1:  # 10% de chance
            x = np.random.randint(0, width)
            cv2.line(frame, (x, 0), (x, height), (255, 255, 255), 1)
        
        # Ajouter des taches aléatoires
        if np.random.random() < 0.05:  # 5% de chance
            spot_x = np.random.randint(0, width)
            spot_y = np.random.randint(0, height)
            cv2.circle(frame, (spot_x, spot_y), 3, (0, 0, 0), -1)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Vidéo de démonstration créée: {output_path}")
    return output_path

def create_old_film_effect(input_path: str, output_path: str) -> str:
    """
    Applique un effet de film ancien à une vidéo couleur.
    
    Args:
        input_path: Vidéo d'entrée
        output_path: Vidéo de sortie
        
    Returns:
        Chemin de la vidéo traitée
    """
    print("🎞️ Application d'un effet de film ancien...")
    
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Créer le répertoire de sortie
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Appliquer un effet sépia
        sepia_frame = np.zeros_like(frame)
        sepia_frame[:, :, 0] = np.clip(0.393 * gray + 0.769 * gray + 0.189 * gray, 0, 255)
        sepia_frame[:, :, 1] = np.clip(0.349 * gray + 0.686 * gray + 0.168 * gray, 0, 255)
        sepia_frame[:, :, 2] = np.clip(0.272 * gray + 0.534 * gray + 0.131 * gray, 0, 255)
        
        # Réduire la saturation
        sepia_frame = (sepia_frame * 0.7 + cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) * 0.3).astype(np.uint8)
        
        # Ajouter du bruit
        noise = np.random.normal(0, 15, sepia_frame.shape).astype(np.int16)
        noisy_frame = np.clip(sepia_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Ajouter des défauts occasionnels
        if frame_idx % 30 == 0:  # Tous les 30 frames
            # Ligne verticale
            x = np.random.randint(0, width)
            cv2.line(noisy_frame, (x, 0), (x, height), (200, 200, 200), 1)
        
        if frame_idx % 100 == 0:  # Tous les 100 frames
            # Tache
            spot_x = np.random.randint(50, width-50)
            spot_y = np.random.randint(50, height-50)
            cv2.circle(noisy_frame, (spot_x, spot_y), 8, (50, 50, 50), -1)
        
        # Variation de luminosité
        brightness_factor = 0.9 + 0.2 * np.sin(frame_idx * 0.1)
        adjusted_frame = np.clip(noisy_frame * brightness_factor, 0, 255).astype(np.uint8)
        
        out.write(adjusted_frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    
    print(f"✅ Effet de film ancien appliqué: {output_path}")
    return output_path

def demo_classical_restoration():
    """Démonstration des méthodes classiques."""
    print("\n🔧 === DÉMONSTRATION DES MÉTHODES CLASSIQUES ===")
    
    # Créer une vidéo de test si elle n'existe pas
    input_video = "data/input/demo_old_film.mp4"
    if not os.path.exists(input_video):
        # Créer d'abord une vidéo couleur
        color_demo = "data/input/demo_color.mp4"
        create_demo_video(color_demo)
        # Puis appliquer l'effet ancien
        create_old_film_effect(color_demo, input_video)
    
    # Charger la vidéo
    processor = VideoProcessor(input_video)
    print(f"📹 Vidéo chargée: {processor.frame_count} frames, {processor.fps:.1f} FPS")
    
    # Appliquer la restauration classique
    restoration = ClassicalRestoration()
    
    print("  Débruitage spatio-temporel...")
    denoised_frames = restoration.denoise_video(processor.frames[:30])  # Limiter pour la démo
    
    print("  Colorisation semi-automatique...")
    colorized_frames = restoration.colorize_video(denoised_frames)
    
    print("  Amélioration du contraste...")
    enhanced_frames = restoration.enhance_contrast(colorized_frames)
    
    # Sauvegarder le résultat
    output_path = "data/output/demo_classical_restored.mp4"
    processor.save_video(enhanced_frames, output_path)
    
    print(f"✅ Restauration classique terminée: {output_path}")
    return output_path

def demo_ai_colorization():
    """Démonstration de la colorisation IA."""
    print("\n🤖 === DÉMONSTRATION DE LA COLORISATION IA ===")
    
    # Utiliser la même vidéo que pour les méthodes classiques
    input_video = "data/input/demo_old_film.mp4"
    
    if not os.path.exists(input_video):
        print("⚠️ Vidéo de démonstration non trouvée, création en cours...")
        color_demo = "data/input/demo_color.mp4"
        create_demo_video(color_demo)
        create_old_film_effect(color_demo, input_video)
    
    # Charger la vidéo
    processor = VideoProcessor(input_video)
    
    # Colorisation IA simple (pour la démonstration)
    print("  Colorisation avec IA simple...")
    simple_colorizer = SimpleColorization()
    colorized_frames = simple_colorizer.colorize_video(processor.frames[:30])  # Limiter pour la démo
    
    # Sauvegarder le résultat
    output_path = "data/output/demo_ai_colorized.mp4"
    processor.save_video(colorized_frames, output_path)
    
    print(f"✅ Colorisation IA terminée: {output_path}")
    return output_path

def demo_evaluation():
    """Démonstration de l'évaluation des résultats."""
    print("\n📊 === DÉMONSTRATION DE L'ÉVALUATION ===")
    
    # Chemins des vidéos
    original_video = "data/input/demo_color.mp4"
    classical_video = "data/output/demo_classical_restored.mp4"
    ai_video = "data/output/demo_ai_colorized.mp4"
    
    # Vérifier que les fichiers existent
    videos_to_check = [classical_video, ai_video]
    existing_videos = {
        'classical': classical_video if os.path.exists(classical_video) else None,
        'ai_simple': ai_video if os.path.exists(ai_video) else None
    }
    
    # Filtrer les vidéos existantes
    existing_videos = {k: v for k, v in existing_videos.items() if v is not None}
    
    if not existing_videos:
        print("⚠️ Aucune vidéo restaurée trouvée pour l'évaluation")
        return
    
    # Calculer les métriques
    metrics_calculator = VideoMetrics()
    qualitative_eval = QualitativeEvaluation()
    
    results = {}
    
    for method, video_path in existing_videos.items():
        print(f"  Évaluation de la méthode: {method}")
        
        # Métriques quantitatives
        if os.path.exists(original_video):
            psnr, ssim = metrics_calculator.compare_videos(original_video, video_path)
        else:
            psnr, ssim = 0.0, 0.0
        
        # Métriques qualitatives
        processor = VideoProcessor(video_path)
        color_coherence = qualitative_eval.calculate_color_coherence(processor.frames)
        naturalness = qualitative_eval.assess_naturalness(processor.frames)
        
        results[method] = {
            'psnr_mean': psnr,
            'ssim_mean': ssim,
            'color_coherence': color_coherence,
            'naturalness': naturalness
        }
    
    # Afficher les résultats
    print("\n📈 RÉSULTATS D'ÉVALUATION:")
    print("=" * 60)
    print(f"{'Méthode':<15} {'PSNR (dB)':<12} {'SSIM':<8} {'Cohérence':<10} {'Naturel':<8}")
    print("=" * 60)
    
    for method, metrics in results.items():
        print(f"{method:<15} {metrics['psnr_mean']:<12.2f} {metrics['ssim_mean']:<8.4f} "
              f"{metrics['color_coherence']:<10.4f} {metrics['naturalness']:<8.4f}")
    
    # Sauvegarder les résultats détaillés
    detailed_results = metrics_calculator.calculate_detailed_metrics(
        original_video if os.path.exists(original_video) else existing_videos[list(existing_videos.keys())[0]],
        existing_videos
    )
    
    metrics_calculator.save_results(detailed_results, "results/demo_evaluation.json")
    metrics_calculator.generate_comparison_plots(detailed_results, "results/plots")
    metrics_calculator.generate_report(detailed_results, "results/demo_report.md")
    
    print(f"\n📄 Rapport détaillé généré dans: results/")

def create_sample_videos():
    """Crée des vidéos d'exemple pour les tests."""
    print("🎬 Création de vidéos d'exemple...")
    
    # Vidéo couleur originale
    color_video = "data/input/demo_color.mp4"
    create_demo_video(color_video, duration=3, fps=24)
    
    # Vidéo avec effet de film ancien
    old_film_video = "data/input/demo_old_film.mp4"
    create_old_film_effect(color_video, old_film_video)
    
    print("✅ Vidéos d'exemple créées")

def main():
    """Fonction principale de démonstration."""
    print("🎭 DÉMONSTRATION - RESTAURATION ET COLORISATION DE VIDÉOS")
    print("=" * 70)
    
    # Créer les répertoires nécessaires
    os.makedirs("data/input", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    try:
        # 1. Créer des vidéos d'exemple
        create_sample_videos()
        
        # 2. Démonstration des méthodes classiques
        classical_result = demo_classical_restoration()
        
        # 3. Démonstration de la colorisation IA
        ai_result = demo_ai_colorization()
        
        # 4. Évaluation des résultats
        demo_evaluation()
        
        print("\n🎉 DÉMONSTRATION TERMINÉE AVEC SUCCÈS!")
        print("=" * 70)
        print("📁 Fichiers générés:")
        print(f"  • Vidéos d'entrée: data/input/")
        print(f"  • Vidéos restaurées: data/output/")
        print(f"  • Rapports d'évaluation: results/")
        print("\n💡 Pour lancer l'interface graphique:")
        print("   python src/gui/main_window.py")
        
    except Exception as e:
        print(f"❌ Erreur lors de la démonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
