"""
Méthodes classiques de restauration vidéo.
Implémente le filtrage spatio-temporel et la colorisation semi-automatique.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy import ndimage
from skimage import restoration, filters
import matplotlib.pyplot as plt

# Import des modules d'optimisation
try:
    from ..utils.gpu_acceleration import AccelerationManager
    from ..utils.performance import OptimizedVideoProcessor
except ImportError:
    # Fallback si les modules ne sont pas disponibles
    AccelerationManager = None
    OptimizedVideoProcessor = None

class ClassicalRestoration:
    """Classe pour les méthodes classiques de restauration vidéo."""
    
    def __init__(self):
        """Initialise les paramètres de restauration."""
        self.temporal_window = 5  # Fenêtre temporelle pour le filtrage
        self.bilateral_params = {
            'd': 9,
            'sigmaColor': 75,
            'sigmaSpace': 75
        }
        self.median_kernel_size = 5
        
        # Initialisation des modules d'optimisation
        self.acceleration_manager = None
        self.optimized_processor = None
        self.device_type = 'cpu'
        
        if AccelerationManager and OptimizedVideoProcessor:
            try:
                # Initialiser le gestionnaire d'accélération GPU
                self.acceleration_manager = AccelerationManager()
                
                # Appliquer les optimisations GPU automatiquement
                self.acceleration_manager.apply_optimizations()
                
                # Obtenir la configuration optimale
                config = self.acceleration_manager.get_processing_config()
                self.device_type = config['device_type']
                
                # Initialiser le processeur vidéo optimisé
                self.optimized_processor = OptimizedVideoProcessor(
                    batch_size=config['batch_size'],
                    max_workers=config['num_workers'],
                    enable_cache=True,
                    memory_limit_percent=80.0                )
                
                print(f"🚀 Optimisations activées - Device: {self.device_type}")
                
            except Exception as e:
                print(f"⚠️ Impossible d'initialiser les optimisations: {e}")
                print("📝 Utilisation du mode standard")
                self.acceleration_manager = None
                self.optimized_processor = None
    
    def denoise_video(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Applique un débruitage spatio-temporel sur la vidéo.
        
        Args:
            frames: Liste des frames de la vidéo
            
        Returns:
            Liste des frames débruitées
        """
        print("🔧 Débruitage spatio-temporel en cours...")
        
        # Utiliser le processeur optimisé si disponible
        if self.optimized_processor:
            try:
                print(f"🚀 Utilisation du processeur optimisé ({self.device_type})")
                
                # Fonction de traitement pour une frame
                def denoise_frame(frame):
                    # Débruitage spatial avec filtre bilatéral
                    spatial_denoised = self._bilateral_filter(frame)
                    return spatial_denoised
                
                # Traitement optimisé avec cache et parallélisation
                if len(frames) > 50:  # Vidéo longue, utiliser parallélisation
                    denoised_frames = self.optimized_processor.process_video_optimized(
                        frames, 
                        denoise_frame,
                        operation_id="denoise_spatial",
                        use_parallel=True
                    )
                else:
                    # Vidéo courte, traitement par batch
                    denoised_frames = self.optimized_processor.process_video_optimized(
                        frames,
                        denoise_frame,
                        operation_id="denoise_spatial",
                        use_parallel=False
                    )
                
                # Appliquer le filtrage temporel sur les frames débruitées
                print("🔧 Application du filtrage temporel...")
                final_frames = []
                for i, spatial_denoised in enumerate(denoised_frames):
                    temporal_denoised = self._temporal_median_filter(
                        denoised_frames, i, spatial_denoised
                    )
                    final_frames.append(temporal_denoised)
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Progression: {i + 1}/{len(frames)} frames")
                
                return final_frames
                
            except Exception as e:
                print(f"⚠️ Erreur avec le processeur optimisé: {e}")
                print("📝 Basculement vers le mode standard")
        
        # Traitement standard
        denoised_frames = []
        
        for i, frame in enumerate(frames):
            # Débruitage spatial avec filtre bilatéral
            spatial_denoised = self._bilateral_filter(frame)
            
            # Débruitage temporel avec filtre médian
            temporal_denoised = self._temporal_median_filter(
                frames, i, spatial_denoised
            )
            
            denoised_frames.append(temporal_denoised)
            
            if (i + 1) % 10 == 0:
                print(f"  Progression: {i + 1}/{len(frames)} frames")
        
        return denoised_frames
    
    def _bilateral_filter(self, frame: np.ndarray) -> np.ndarray:
        """Applique un filtre bilatéral pour le débruitage spatial."""
        if len(frame.shape) == 3:
            # Frame couleur
            return cv2.bilateralFilter(
                frame,
                self.bilateral_params['d'],
                self.bilateral_params['sigmaColor'],
                self.bilateral_params['sigmaSpace']
            )
        else:
            # Frame en niveaux de gris
            return cv2.bilateralFilter(
                frame,
                self.bilateral_params['d'],
                self.bilateral_params['sigmaColor'],
                self.bilateral_params['sigmaSpace']
            )
    
    def _temporal_median_filter(self, frames: List[np.ndarray], 
                              current_idx: int, 
                              current_frame: np.ndarray) -> np.ndarray:
        """
        Applique un filtre médian temporel.
        
        Args:
            frames: Toutes les frames
            current_idx: Index de la frame courante
            current_frame: Frame courante déjà filtrée spatialement
            
        Returns:
            Frame filtrée temporellement
        """
        # Déterminer la fenêtre temporelle
        half_window = self.temporal_window // 2
        start_idx = max(0, current_idx - half_window)
        end_idx = min(len(frames), current_idx + half_window + 1)
        
        # Extraire les frames de la fenêtre
        window_frames = []
        for idx in range(start_idx, end_idx):
            if idx == current_idx:
                window_frames.append(current_frame)
            else:
                window_frames.append(self._bilateral_filter(frames[idx]))
        
        # Appliquer le filtre médian temporel
        if len(window_frames) >= 3:
            temporal_stack = np.stack(window_frames, axis=-1)
            return np.median(temporal_stack, axis=-1).astype(np.uint8)
        else:
            return current_frame
      def colorize_video(self, frames: List[np.ndarray], 
                      reference_points: Optional[Dict] = None) -> List[np.ndarray]:
        """
        Colorise la vidéo en utilisant la propagation de couleurs.
        Implémentation basée sur la méthode de Levin et al. (2004).
        
        Args:
            frames: Frames en niveaux de gris
            reference_points: Points de référence pour la colorisation
            
        Returns:
            Frames colorisées
        """
        print("🎨 Colorisation semi-automatique en cours...")
        
        # Si pas de points de référence, utiliser des couleurs par défaut
        if reference_points is None:
            reference_points = self._generate_default_colors()
        
        # Utiliser le processeur optimisé si disponible
        if self.optimized_processor:
            try:
                print(f"🚀 Utilisation du processeur optimisé ({self.device_type})")
                
                # Fonction de traitement pour une frame
                def colorize_frame(frame):
                    # Convertir en niveaux de gris si nécessaire
                    if len(frame.shape) == 3:
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_frame = frame.copy()
                    
                    # Coloriser la frame
                    return self._colorize_frame(gray_frame, reference_points)
                
                # Traitement optimisé avec cache et parallélisation
                if len(frames) > 30:  # Vidéo longue, utiliser parallélisation
                    colorized_frames = self.optimized_processor.process_video_optimized(
                        frames, 
                        colorize_frame,
                        operation_id="colorize_classical",
                        use_parallel=True
                    )
                else:
                    # Vidéo courte, traitement par batch
                    colorized_frames = self.optimized_processor.process_video_optimized(
                        frames,
                        colorize_frame,
                        operation_id="colorize_classical",
                        use_parallel=False
                    )
                
                return colorized_frames
                
            except Exception as e:
                print(f"⚠️ Erreur avec le processeur optimisé: {e}")
                print("📝 Basculement vers le mode standard")
        
        # Traitement standard
        colorized_frames = []
        
        for i, frame in enumerate(frames):
            # Convertir en niveaux de gris si nécessaire
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame.copy()
              # Coloriser la frame
            colorized = self._colorize_frame(gray_frame, reference_points)
            colorized_frames.append(colorized)
            
            if (i + 1) % 10 == 0:
                print(f"  Progression: {i + 1}/{len(frames)} frames")
        
        return colorized_frames
    
    def _generate_default_colors(self) -> Dict:
        """Génère des couleurs par défaut pour la colorisation."""
        return {
            'skin': np.array([180, 120, 90]),      # Couleur peau
            'sky': np.array([135, 206, 235]),      # Couleur ciel
            'grass': np.array([34, 139, 34]),      # Couleur herbe
            'wood': np.array([139, 69, 19]),       # Couleur bois
            'water': np.array([30, 144, 255])      # Couleur eau
        }
    
    def _colorize_frame(self, gray_frame: np.ndarray, 
                       reference_points: Dict) -> np.ndarray:
        """
        Colorise une frame individuelle.
        
        Args:
            gray_frame: Frame en niveaux de gris
            reference_points: Points de référence colorés
            
        Returns:
            Frame colorisée
        """
        # Assurer que gray_frame est en uint8
        if gray_frame.dtype != np.uint8:
            gray_frame = np.clip(gray_frame, 0, 255).astype(np.uint8)
            
        height, width = gray_frame.shape
        colorized = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Convertir la frame en espace LAB pour une meilleure colorisation
        gray_3ch = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        lab_frame = cv2.cvtColor(gray_3ch, cv2.COLOR_BGR2LAB)
          # Garder le canal L (luminance) original
        colorized_lab = lab_frame.copy()
        
        # Appliquer une colorisation basée sur l'intensité
        for region_name, color_bgr in reference_points.items():
            # Assurer que color_bgr est en uint8
            color_bgr = np.array(color_bgr, dtype=np.uint8)
            color_bgr_reshaped = color_bgr.reshape(1, 1, 3)
            color_lab = cv2.cvtColor(color_bgr_reshaped, cv2.COLOR_BGR2LAB)[0, 0]
            
            # Créer un masque basé sur l'intensité
            mask = self._create_intensity_mask(gray_frame, region_name)
            
            # Appliquer la couleur aux canaux A et B
            colorized_lab[mask, 1] = color_lab[1]  # Canal A
            colorized_lab[mask, 2] = color_lab[2]  # Canal B
        
        # Convertir de LAB vers BGR
        colorized = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
        
        return colorized
    
    def _create_intensity_mask(self, gray_frame: np.ndarray, 
                             region_name: str) -> np.ndarray:
        """
        Crée un masque basé sur l'intensité pour une région spécifique.
        
        Args:
            gray_frame: Frame en niveaux de gris
            region_name: Nom de la région
            
        Returns:
            Masque binaire
        """
        height, width = gray_frame.shape
        mask = np.zeros((height, width), dtype=bool)
        
        # Définir des plages d'intensité pour différentes régions
        intensity_ranges = {
            'skin': (80, 180),
            'sky': (150, 255),
            'grass': (30, 100),
            'wood': (50, 120),
            'water': (40, 140)
        }
        
        if region_name in intensity_ranges:
            min_val, max_val = intensity_ranges[region_name]
            mask = (gray_frame >= min_val) & (gray_frame <= max_val)
            
            # Appliquer des contraintes spatiales selon la région
            if region_name == 'sky':
                # Le ciel est généralement dans la partie supérieure
                mask[:height//3, :] = mask[:height//3, :] | (gray_frame[:height//3, :] > 120)
                mask[height//2:, :] = False
            elif region_name == 'grass':
                # L'herbe est généralement dans la partie inférieure
                mask[:height//2, :] = False
                mask[height*2//3:, :] = mask[height*2//3:, :] | (gray_frame[height*2//3:, :] < 80)
        
        # Appliquer un filtrage morphologique pour nettoyer le masque
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask.astype(bool)
    
    def enhance_contrast(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Améliore le contraste des frames.
        
        Args:
            frames: Liste des frames
            
        Returns:
            Frames avec contraste amélioré
        """
        enhanced_frames = []
        
        for frame in frames:
            # Égalisation d'histogramme adaptative (CLAHE)
            if len(frame.shape) == 3:
                # Frame couleur
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                # Frame en niveaux de gris
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(frame)
            
            enhanced_frames.append(enhanced)
        
        return enhanced_frames
    
    def stabilize_video(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Stabilise la vidéo en réduisant les mouvements de caméra.
        
        Args:
            frames: Liste des frames
            
        Returns:
            Frames stabilisées
        """
        if len(frames) < 2:
            return frames
        
        print("🎯 Stabilisation vidéo en cours...")
        stabilized_frames = [frames[0]]  # Première frame comme référence
        
        # Paramètres pour le détecteur de features
        feature_params = {
            'maxCorners': 200,
            'qualityLevel': 0.01,
            'minDistance': 30,
            'blockSize': 3
        }
        
        # Paramètres pour le flux optique
        lk_params = {
            'winSize': (15, 15),
            'maxLevel': 2,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }
        
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY) if len(frames[0].shape) == 3 else frames[0]
        
        for i in range(1, len(frames)):
            curr_frame = frames[i]
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if len(curr_frame.shape) == 3 else curr_frame
            
            # Détecter les features dans la frame précédente
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, **feature_params)
            
            if prev_pts is not None:
                # Calculer le flux optique
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, prev_pts, None, **lk_params
                )
                
                # Filtrer les bons points
                good_prev = prev_pts[status == 1]
                good_curr = curr_pts[status == 1]
                
                if len(good_prev) >= 4:
                    # Estimer la transformation
                    transform_matrix = cv2.estimateAffinePartial2D(
                        good_prev, good_curr
                    )[0]
                    
                    if transform_matrix is not None:
                        # Appliquer la transformation inverse pour stabiliser
                        h, w = curr_frame.shape[:2]
                        stabilized = cv2.warpAffine(
                            curr_frame, 
                            np.linalg.inv(transform_matrix[:2, :]), 
                            (w, h)
                        )
                        stabilized_frames.append(stabilized)
                    else:
                        stabilized_frames.append(curr_frame)
                else:
                    stabilized_frames.append(curr_frame)
            else:
                stabilized_frames.append(curr_frame)
            
            prev_gray = curr_gray
        
        return stabilized_frames
    
    def get_optimization_status(self) -> dict:
        """
        Retourne l'état des optimisations activées.
        
        Returns:
            Dictionnaire avec les informations d'optimisation
        """
        status = {
            'optimizations_enabled': self.optimized_processor is not None,
            'acceleration_enabled': self.acceleration_manager is not None,
            'device_type': self.device_type,
            'features': []
        }
        
        if self.acceleration_manager:
            config = self.acceleration_manager.get_processing_config()
            status.update({
                'batch_size': config['batch_size'],
                'num_workers': config['num_workers'],
                'use_gpu_acceleration': config['use_gpu_acceleration'],
                'memory_limit_gb': config['memory_limit_gb']
            })
            
            if config['use_gpu_acceleration']:
                status['features'].append('GPU Acceleration')
            status['features'].extend(['Batch Processing', 'Parallel Processing', 'Memory Management'])
        
        if self.optimized_processor:
            status['features'].extend(['Video Caching', 'Performance Monitoring'])
        
        return status
    
    def process_with_performance_tracking(self, frames, method_name='unknown'):
        """
        Traite les frames avec suivi des performances.
        
        Args:
            frames: Frames à traiter
            method_name: Nom de la méthode pour le suivi
            
        Returns:
            Tuple (résultat, statistiques)
        """
        import time
        
        start_time = time.time()
        
        # Appliquer la méthode selon le nom
        if method_name == 'denoise':
            result = self.denoise_video(frames)
        elif method_name == 'colorize':
            result = self.colorize_video(frames)
        elif method_name == 'enhance_contrast':
            result = self.enhance_contrast(frames)
        elif method_name == 'stabilize':
            result = self.stabilize_video(frames)
        else:
            raise ValueError(f"Méthode inconnue: {method_name}")
        
        end_time = time.time()
        
        stats = {
            'method': method_name,
            'processing_time': end_time - start_time,
            'frames_count': len(frames),
            'fps': len(frames) / (end_time - start_time) if end_time > start_time else 0,
            'optimizations_used': self.optimized_processor is not None,
            'device_type': self.device_type
        }
        
        return result, stats
