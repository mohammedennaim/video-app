#!/usr/bin/env python3
"""
Point d'entrée principal pour la restauration et colorisation de vidéos anciennes.
Support pour les approches classiques et IA avec interface graphique.
"""

import argparse
import sys
import os
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.append(str(Path(__file__).parent / 'src'))

from classical.video_restoration import ClassicalRestoration
from ai.gan_colorization import GANColorization
from gui.main_window import VideoRestorationApp
from utils.video_utils import VideoProcessor
from evaluation.metrics import VideoMetrics

def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Restauration et colorisation de vidéos anciennes"
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help="Chemin vers la vidéo d'entrée"
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help="Chemin vers la vidéo de sortie (optionnel)"
    )
    
    parser.add_argument(
        '--method',
        choices=['classical', 'ai', 'both'],
        default='both',
        help="Méthode de restauration à utiliser"
    )
    
    parser.add_argument(
        '--gui',
        action='store_true',
        help="Lancer l'interface graphique"
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help="Évaluer les résultats avec PSNR/SSIM"
    )
    
    return parser.parse_args()

def process_video_classical(input_path: str, output_path: str) -> str:
    """Traite la vidéo avec les méthodes classiques."""
    print("🔧 Traitement avec méthodes classiques...")
    
    restoration = ClassicalRestoration()
    processor = VideoProcessor(input_path)
    
    # Débruitage spatio-temporel
    denoised_frames = restoration.denoise_video(processor.frames)
    
    # Colorisation semi-automatique
    colorized_frames = restoration.colorize_video(denoised_frames)
    
    # Sauvegarde
    output_classical = output_path.replace('.mp4', '_classical.mp4')
    processor.save_video(colorized_frames, output_classical)
    
    print(f"✅ Restauration classique terminée : {output_classical}")
    return output_classical

def process_video_ai(input_path: str, output_path: str) -> str:
    """Traite la vidéo avec les méthodes IA."""
    print("🤖 Traitement avec méthodes IA...")
    
    gan_colorizer = GANColorization()
    processor = VideoProcessor(input_path)
    
    # Colorisation GAN
    colorized_frames = gan_colorizer.colorize_video(processor.frames)
    
    # Sauvegarde
    output_ai = output_path.replace('.mp4', '_ai.mp4')
    processor.save_video(colorized_frames, output_ai)
    
    print(f"✅ Restauration IA terminée : {output_ai}")
    return output_ai

def evaluate_results(original_path: str, restored_paths: list):
    """Évalue les résultats avec les métriques PSNR/SSIM."""
    print("📊 Évaluation des résultats...")
    
    metrics = VideoMetrics()
    results = {}
    
    for method, path in restored_paths.items():
        if os.path.exists(path):
            psnr, ssim = metrics.compare_videos(original_path, path)
            results[method] = {'PSNR': psnr, 'SSIM': ssim}
            print(f"📈 {method.upper()}: PSNR={psnr:.2f}dB, SSIM={ssim:.4f}")
    
    # Sauvegarder les résultats
    metrics.save_results(results, 'results/evaluation_results.json')
    return results

def main():
    """Fonction principale."""
    args = parse_arguments()
    
    # Vérifier que le fichier d'entrée existe
    if not os.path.exists(args.input):
        print(f"❌ Erreur: Le fichier {args.input} n'existe pas")
        return 1
    
    # Lancer l'interface graphique si demandé
    if args.gui:
        from PyQt5.QtWidgets import QApplication
        app = QApplication(sys.argv)
        window = VideoRestorationApp()
        window.show()
        return app.exec_()
      # Traitement en ligne de commande
    if not args.output:
        base_name = Path(args.input).stem
        args.output = f"data/output/{base_name}_restored.mp4"
    
    # Créer les répertoires de sortie
    output_dir = os.path.dirname(args.output)
    if output_dir:  # Only create directory if there's a path
        os.makedirs(output_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    restored_paths = {}
    
    # Traitement selon la méthode choisie
    if args.method in ['classical', 'both']:
        restored_paths['classical'] = process_video_classical(args.input, args.output)
    
    if args.method in ['ai', 'both']:
        restored_paths['ai'] = process_video_ai(args.input, args.output)
    
    # Évaluation si demandée
    if args.evaluate and restored_paths:
        evaluate_results(args.input, restored_paths)
    
    print("🎉 Traitement terminé avec succès!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
