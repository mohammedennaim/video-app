"""
Métriques d'évaluation pour la restauration vidéo.
Calcul de PSNR, SSIM et autres métriques de qualité.
"""

import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Optional
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class VideoMetrics:
    """Classe pour calculer les métriques d'évaluation vidéo."""
    
    def __init__(self):
        """Initialise le calculateur de métriques."""
        pass
    
    def compare_videos(self, original_path: str, restored_path: str) -> Tuple[float, float]:
        """
        Compare deux vidéos et calcule PSNR et SSIM moyens.
        
        Args:
            original_path: Chemin vers la vidéo originale
            restored_path: Chemin vers la vidéo restaurée
            
        Returns:
            Tuple (PSNR moyen, SSIM moyen)
        """
        # Charger les vidéos
        original_frames = self._load_video_frames(original_path)
        restored_frames = self._load_video_frames(restored_path)
        
        if not original_frames or not restored_frames:
            return 0.0, 0.0
        
        # S'assurer que les vidéos ont le même nombre de frames
        min_frames = min(len(original_frames), len(restored_frames))
        original_frames = original_frames[:min_frames]
        restored_frames = restored_frames[:min_frames]
        
        # Calculer les métriques pour chaque frame
        psnr_values = []
        ssim_values = []
        
        for orig, rest in zip(original_frames, restored_frames):
            # Redimensionner si nécessaire
            if orig.shape != rest.shape:
                rest = cv2.resize(rest, (orig.shape[1], orig.shape[0]))
            
            # Calculer PSNR
            psnr_val = self.calculate_psnr(orig, rest)
            psnr_values.append(psnr_val)
            
            # Calculer SSIM
            ssim_val = self.calculate_ssim(orig, rest)
            ssim_values.append(ssim_val)
        
        return np.mean(psnr_values), np.mean(ssim_values)
    
    def _load_video_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Charge les frames d'une vidéo.
        
        Args:
            video_path: Chemin vers la vidéo
            
        Returns:
            Liste des frames
        """
        if not os.path.exists(video_path):
            print(f"❌ Fichier non trouvé: {video_path}")
            return []
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return frames
    
    def calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calcule le PSNR entre deux images.
        
        Args:
            img1: Première image
            img2: Deuxième image
            
        Returns:
            Valeur PSNR en dB
        """
        try:
            # Convertir en niveaux de gris si nécessaire
            if len(img1.shape) == 3:
                img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            else:
                img1_gray = img1
                
            if len(img2.shape) == 3:
                img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else:
                img2_gray = img2
            
            return psnr(img1_gray, img2_gray)
        except Exception as e:
            print(f"Erreur calcul PSNR: {e}")
            return 0.0
    
    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calcule le SSIM entre deux images.
        
        Args:
            img1: Première image
            img2: Deuxième image
            
        Returns:
            Valeur SSIM (0-1)
        """
        try:
            # Convertir en niveaux de gris si nécessaire
            if len(img1.shape) == 3:
                img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            else:
                img1_gray = img1
                
            if len(img2.shape) == 3:
                img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else:
                img2_gray = img2
            
            return ssim(img1_gray, img2_gray)
        except Exception as e:
            print(f"Erreur calcul SSIM: {e}")
            return 0.0
    
    def calculate_mse(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calcule l'erreur quadratique moyenne (MSE).
        
        Args:
            img1: Première image
            img2: Deuxième image
            
        Returns:
            Valeur MSE
        """
        return np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    
    def calculate_mae(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calcule l'erreur absolue moyenne (MAE).
        
        Args:
            img1: Première image
            img2: Deuxième image
            
        Returns:
            Valeur MAE
        """
        return np.mean(np.abs(img1.astype(float) - img2.astype(float)))
    
    def calculate_detailed_metrics(self, original_path: str, 
                                 restored_paths: Dict[str, str]) -> Dict:
        """
        Calcule des métriques détaillées pour plusieurs méthodes.
        
        Args:
            original_path: Chemin vers la vidéo originale
            restored_paths: Dictionnaire {méthode: chemin_vidéo}
            
        Returns:
            Dictionnaire des métriques détaillées
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'original_video': original_path,
            'methods': {}
        }
        
        original_frames = self._load_video_frames(original_path)
        
        for method_name, restored_path in restored_paths.items():
            print(f"📊 Évaluation de la méthode: {method_name}")
            
            restored_frames = self._load_video_frames(restored_path)
            
            if not restored_frames:
                continue
            
            # S'assurer que les vidéos ont le même nombre de frames
            min_frames = min(len(original_frames), len(restored_frames))
            orig_subset = original_frames[:min_frames]
            rest_subset = restored_frames[:min_frames]
            
            # Calculer toutes les métriques
            method_results = {
                'video_path': restored_path,
                'frame_count': min_frames,
                'metrics': {
                    'psnr': [],
                    'ssim': [],
                    'mse': [],
                    'mae': []
                }
            }
            
            for orig, rest in zip(orig_subset, rest_subset):
                if orig.shape != rest.shape:
                    rest = cv2.resize(rest, (orig.shape[1], orig.shape[0]))
                
                method_results['metrics']['psnr'].append(
                    self.calculate_psnr(orig, rest)
                )
                method_results['metrics']['ssim'].append(
                    self.calculate_ssim(orig, rest)
                )
                method_results['metrics']['mse'].append(
                    self.calculate_mse(orig, rest)
                )
                method_results['metrics']['mae'].append(
                    self.calculate_mae(orig, rest)
                )
            
            # Calculer les statistiques
            for metric_name, values in method_results['metrics'].items():
                method_results['metrics'][metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
            
            results['methods'][method_name] = method_results
        
        return results
    
    def save_results(self, results: Dict, output_path: str):
        """
        Sauvegarde les résultats d'évaluation.
        
        Args:
            results: Dictionnaire des résultats
            output_path: Chemin de sauvegarde
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Résultats sauvegardés dans: {output_path}")
    
    def generate_comparison_plots(self, results: Dict, output_dir: str = "results/plots"):
        """
        Génère des graphiques de comparaison.
        
        Args:
            results: Résultats d'évaluation
            output_dir: Répertoire de sortie pour les graphiques
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Préparer les données pour les graphiques
        methods = list(results['methods'].keys())
        metrics_names = ['psnr', 'ssim', 'mse', 'mae']
        
        # Graphique en barres pour les moyennes
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric_name in enumerate(metrics_names):
            means = []
            stds = []
            
            for method in methods:
                method_data = results['methods'][method]['metrics'][metric_name]
                means.append(method_data['mean'])
                stds.append(method_data['std'])
            
            axes[i].bar(methods, means, yerr=stds, capsize=5, alpha=0.7)
            axes[i].set_title(f'{metric_name.upper()} Comparison')
            axes[i].set_ylabel(metric_name.upper())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Ajouter les valeurs sur les barres
            for j, (mean, std) in enumerate(zip(means, stds)):
                axes[i].text(j, mean + std, f'{mean:.3f}', 
                           ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Graphiques d'évolution temporelle
        for metric_name in ['psnr', 'ssim']:
            plt.figure(figsize=(12, 6))
            
            for method in methods:
                values = results['methods'][method]['metrics'][metric_name]['values']
                frames = range(len(values))
                plt.plot(frames, values, label=method, alpha=0.8)
            
            plt.title(f'{metric_name.upper()} Evolution over Frames')
            plt.xlabel('Frame Number')
            plt.ylabel(metric_name.upper())
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(output_dir, f'{metric_name}_evolution.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📈 Graphiques sauvegardés dans: {output_dir}")
    
    def generate_report(self, results: Dict, output_path: str = "results/evaluation_report.md"):
        """
        Génère un rapport d'évaluation en Markdown.
        
        Args:
            results: Résultats d'évaluation
            output_path: Chemin de sortie du rapport
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Rapport d'Évaluation - Restauration Vidéo\n\n")
            f.write(f"**Date d'évaluation:** {results['timestamp']}\n\n")
            f.write(f"**Vidéo originale:** {results['original_video']}\n\n")
            
            f.write("## Résumé des Résultats\n\n")
            f.write("| Méthode | PSNR (dB) | SSIM | MSE | MAE |\n")
            f.write("|---------|-----------|------|-----|-----|\n")
            
            for method_name, method_data in results['methods'].items():
                metrics = method_data['metrics']
                f.write(f"| {method_name} | ")
                f.write(f"{metrics['psnr']['mean']:.2f} ± {metrics['psnr']['std']:.2f} | ")
                f.write(f"{metrics['ssim']['mean']:.4f} ± {metrics['ssim']['std']:.4f} | ")
                f.write(f"{metrics['mse']['mean']:.2f} ± {metrics['mse']['std']:.2f} | ")
                f.write(f"{metrics['mae']['mean']:.2f} ± {metrics['mae']['std']:.2f} |\n")
            
            f.write("\n## Analyse Détaillée\n\n")
            
            for method_name, method_data in results['methods'].items():
                f.write(f"### {method_name.title()}\n\n")
                f.write(f"- **Nombre de frames:** {method_data['frame_count']}\n")
                f.write(f"- **Fichier vidéo:** {method_data['video_path']}\n\n")
                
                metrics = method_data['metrics']
                f.write("**Métriques détaillées:**\n\n")
                
                for metric_name, metric_data in metrics.items():
                    f.write(f"- **{metric_name.upper()}:**\n")
                    f.write(f"  - Moyenne: {metric_data['mean']:.4f}\n")
                    f.write(f"  - Écart-type: {metric_data['std']:.4f}\n")
                    f.write(f"  - Min: {metric_data['min']:.4f}\n")
                    f.write(f"  - Max: {metric_data['max']:.4f}\n\n")
            
            f.write("## Interprétation\n\n")
            f.write("### PSNR (Peak Signal-to-Noise Ratio)\n")
            f.write("- Plus élevé = meilleure qualité\n")
            f.write("- > 30 dB: Bonne qualité\n")
            f.write("- > 40 dB: Très bonne qualité\n\n")
            
            f.write("### SSIM (Structural Similarity Index)\n")
            f.write("- Valeurs entre 0 et 1\n")
            f.write("- Plus proche de 1 = meilleure similitude structurelle\n")
            f.write("- > 0.9: Excellente similarité\n\n")
            
            f.write("### MSE/MAE\n")
            f.write("- Plus faible = meilleure qualité\n")
            f.write("- Mesure l'erreur pixel par pixel\n\n")
        
        print(f"📄 Rapport généré: {output_path}")

class QualitativeEvaluation:
    """Évaluation qualitative basée sur la perception humaine."""
    
    def __init__(self):
        """Initialise l'évaluateur qualitatif."""
        pass
    
    def calculate_color_coherence(self, frames: List[np.ndarray]) -> float:
        """
        Calcule la cohérence des couleurs entre les frames.
        
        Args:
            frames: Liste des frames colorisées
            
        Returns:
            Score de cohérence (0-1)
        """
        if len(frames) < 2:
            return 1.0
        
        coherence_scores = []
        
        for i in range(len(frames) - 1):
            # Calculer l'histogramme des couleurs pour chaque frame
            hist1 = self._calculate_color_histogram(frames[i])
            hist2 = self._calculate_color_histogram(frames[i + 1])
            
            # Calculer la corrélation des histogrammes
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            coherence_scores.append(correlation)
        
        return np.mean(coherence_scores)
    
    def _calculate_color_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Calcule l'histogramme des couleurs d'une frame."""
        if len(frame.shape) == 3:
            # Frame couleur
            hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
            return np.concatenate([hist_b, hist_g, hist_r])
        else:
            # Frame en niveaux de gris
            return cv2.calcHist([frame], [0], None, [256], [0, 256])
    
    def assess_naturalness(self, frames: List[np.ndarray]) -> float:
        """
        Évalue le naturel des couleurs (score heuristique).
        
        Args:
            frames: Frames colorisées
            
        Returns:
            Score de naturel (0-1)
        """
        naturalness_scores = []
        
        for frame in frames:
            if len(frame.shape) == 3:
                # Analyser la distribution des couleurs
                score = self._analyze_color_distribution(frame)
                naturalness_scores.append(score)
        
        return np.mean(naturalness_scores) if naturalness_scores else 0.0
    
    def _analyze_color_distribution(self, frame: np.ndarray) -> float:
        """Analyse la distribution des couleurs pour évaluer le naturel."""
        # Convertir en HSV pour une meilleure analyse
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Analyser la saturation (trop saturé = moins naturel)
        saturation = hsv[:, :, 1]
        sat_mean = np.mean(saturation)
        sat_std = np.std(saturation)
        
        # Score basé sur des valeurs typiques de saturation naturelle
        sat_score = 1.0 - abs(sat_mean - 127) / 127  # Optimal autour de 127
        
        # Analyser la distribution des teintes
        hue = hsv[:, :, 0]
        hue_hist = cv2.calcHist([hue], [0], None, [180], [0, 180])
        hue_entropy = -np.sum(hue_hist * np.log(hue_hist + 1e-10))
        hue_score = min(1.0, hue_entropy / 1000)  # Normaliser
        
        return (sat_score + hue_score) / 2
