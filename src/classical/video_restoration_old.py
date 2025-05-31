"""
Méthodes classiques de restauration vidéo.
Implémente le filtrage spatio-temporel et la colorisation semi-automatique selon Levin et al. (2004).
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

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
                window_frames.append(self._bilateral_filter(frames[idx]))        # Appliquer le filtre médian temporel
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
        if reference_points is None:
            reference_points = self._generate_default_colors()
            
        colorized_frames = []
        
        for frame in frames:
            # Conversion en niveaux de gris si nécessaire
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame.copy()
                
            # Appliquer la colorisation de Levin
            colorized_frame = self._levin_colorization(gray_frame, reference_points)
            colorized_frames.append(colorized_frame)
            
        return colorized_frames
            reference_points: Points de référence pour la colorisation
            
        Returns:
            Frames colorisées
        """
        print("🎨 Colorisation semi-automatique (Levin et al., 2004) en cours...")
        
        # Si pas de points de référence, utiliser des couleurs par défaut
        if reference_points is None:
            reference_points = self._generate_default_colors()
        
        # Utiliser le processeur optimisé si disponible
        if self.optimized_processor:
            try:
                print(f"🚀 Utilisation du processeur optimisé ({self.device_type})")
                
                # Fonction de traitement pour une frame                def colorize_frame(frame):
                    # Convertir en niveaux de gris si nécessaire
                    if len(frame.shape) == 3:
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_frame = frame.copy()
                    
                    # Coloriser la frame avec la méthode de Levin
                    return self._levin_colorization(gray_frame, reference_points)
                
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
                print("📝 Basculement vers le mode standard")        # Traitement standard
        colorized_frames = []
        for i, frame in enumerate(frames):
            # Convertir en niveaux de gris si nécessaire
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame.copy()
            
            # Coloriser la frame avec la méthode de Levin
            colorized = self._levin_colorization(gray_frame, reference_points)
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
    
    def _levin_colorization(self, gray_frame: np.ndarray, reference_points: Dict) -> np.ndarray:
        """
        Implémentation de la méthode de colorisation de Levin et al. (2004).
        Utilise la propagation de couleurs basée sur l'affinité des pixels.
        
        Args:
            gray_frame: Frame en niveaux de gris
            reference_points: Dictionnaire des couleurs de référence
            
        Returns:
            Frame colorisée
        """
        height, width = gray_frame.shape
        
        # Normaliser l'image en niveaux de gris
        gray_normalized = gray_frame.astype(np.float64) / 255.0
        
        # Créer les canaux de couleur
        colorized = np.zeros((height, width, 3), dtype=np.float64)
        
        # Traiter chaque canal de couleur séparément (a et b dans l'espace LAB)
        for channel in range(2):  # Canaux a et b
            # Construire la matrice d'affinité selon Levin et al.
            A = self._build_affinity_matrix(gray_normalized)
            
            # Créer le vecteur de contraintes de couleur
            b_vector = self._create_color_constraints(gray_frame, reference_points, channel)
            
            # Résoudre le système linéaire Ax = b
            try:
                colors = spsolve(A, b_vector)
                colors = colors.reshape((height, width))
                colorized[:, :, channel + 1] = colors  # Canaux a et b
            except Exception as e:
                print(f"Erreur dans la résolution: {e}")
                # Fallback: interpolation simple
                colorized[:, :, channel + 1] = self._simple_color_propagation(
                    gray_frame, reference_points, channel
                )
        
        # Canal L (luminance) = niveaux de gris
        colorized[:, :, 0] = gray_normalized
        
        # Convertir de LAB vers BGR
        colorized_lab = (colorized * 255).astype(np.uint8)
        colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
        
        return colorized_bgr
    
    def _build_affinity_matrix(self, gray_image: np.ndarray) -> csc_matrix:
        """
        Construit la matrice d'affinité selon Levin et al. (2004).
        
        Args:
            gray_image: Image en niveaux de gris normalisée
            
        Returns:
            Matrice d'affinité sparse
        """
        height, width = gray_image.shape
        n_pixels = height * width
        
        # Paramètres de la méthode de Levin
        window_size = 3
        sigma = 0.1
        
        # Listes pour construire la matrice sparse
        rows, cols, vals = [], [], []
        
        for y in range(height):
            for x in range(width):
                center_idx = y * width + x
                
                # Fenêtre autour du pixel
                y_start = max(0, y - window_size // 2)
                y_end = min(height, y + window_size // 2 + 1)
                x_start = max(0, x - window_size // 2)
                x_end = min(width, x + window_size // 2 + 1)
                
                window = gray_image[y_start:y_end, x_start:x_end]
                window_flat = window.flatten()
                
                # Calculer les poids d'affinité
                center_val = gray_image[y, x]
                for wy in range(y_start, y_end):
                    for wx in range(x_start, x_end):
                        neighbor_idx = wy * width + wx
                        neighbor_val = gray_image[wy, wx]
                        
                        # Poids basé sur la similarité d'intensité
                        weight = np.exp(-((center_val - neighbor_val) ** 2) / (2 * sigma ** 2))
                        
                        rows.append(center_idx)
                        cols.append(neighbor_idx)
                        vals.append(weight)
        
        # Normaliser les poids et créer la matrice laplacienne
        A = csc_matrix((vals, (rows, cols)), shape=(n_pixels, n_pixels))
        
        # Normaliser chaque ligne
        row_sums = np.array(A.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1  # Éviter la division par zéro
        
        # Créer la matrice laplacienne
        D = csc_matrix((row_sums, (range(n_pixels), range(n_pixels))), shape=(n_pixels, n_pixels))
        L = D - A
        
        return L
    
    def _create_color_constraints(self, gray_frame: np.ndarray, reference_points: Dict, channel: int) -> np.ndarray:
        """
        Crée le vecteur de contraintes de couleur pour un canal donné.
        
        Args:
            gray_frame: Frame en niveaux de gris
            reference_points: Points de référence colorés
            channel: Canal de couleur (0 pour a, 1 pour b)
            
        Returns:
            Vecteur de contraintes
        """
        height, width = gray_frame.shape
        b_vector = np.zeros(height * width)
        
        # Générer des contraintes basées sur l'intensité des pixels
        for color_name, color_rgb in reference_points.items():
            # Convertir la couleur RGB vers LAB
            color_lab = cv2.cvtColor(
                np.uint8([[color_rgb]]), cv2.COLOR_RGB2LAB
            )[0, 0]
            
            # Déterminer les pixels qui correspondent à cette couleur
            mask = self._get_color_mask(gray_frame, color_name)
            pixel_indices = np.where(mask.flatten())[0]
            
            # Assigner la valeur du canal correspondant
            if channel == 0:  # Canal a
                b_vector[pixel_indices] = color_lab[1] / 255.0
            else:  # Canal b
                b_vector[pixel_indices] = color_lab[2] / 255.0
        
        return b_vector
    
    def _get_color_mask(self, gray_frame: np.ndarray, color_name: str) -> np.ndarray:
        """
        Génère un masque pour une catégorie de couleur donnée.
        
        Args:
            gray_frame: Frame en niveaux de gris
            color_name: Nom de la catégorie de couleur
            
        Returns:
            Masque binaire
        """
        height, width = gray_frame.shape
        mask = np.zeros((height, width), dtype=bool)
        
        # Heuristiques basées sur la position et l'intensité
        if color_name == 'sky':
            # Ciel généralement dans la partie supérieure et lumineux
            mask[:height//3, :] = gray_frame[:height//3, :] > 150
        elif color_name == 'grass':
            # Herbe généralement dans la partie inférieure
            mask[2*height//3:, :] = (gray_frame[2*height//3:, :] > 80) & (gray_frame[2*height//3:, :] < 150)
        elif color_name == 'skin':
            # Peau généralement dans la partie centrale avec intensité moyenne
            center_y, center_x = height//2, width//2
            mask[center_y-height//4:center_y+height//4, center_x-width//4:center_x+width//4] = \
                (gray_frame[center_y-height//4:center_y+height//4, center_x-width//4:center_x+width//4] > 100) & \
                (gray_frame[center_y-height//4:center_y+height//4, center_x-width//4:center_x+width//4] < 180)
        elif color_name == 'water':
            # Eau généralement sombre
            mask = (gray_frame > 50) & (gray_frame < 120)
        elif color_name == 'wood':
            # Bois avec intensité moyenne-faible
            mask = (gray_frame > 60) & (gray_frame < 130)
        
        return mask
    
    def _simple_color_propagation(self, gray_frame: np.ndarray, reference_points: Dict, channel: int) -> np.ndarray:
        """
        Propagation de couleur simple en cas d'échec de la méthode de Levin.
        
        Args:
            gray_frame: Frame en niveaux de gris
            reference_points: Points de référence colorés
            channel: Canal de couleur
            
        Returns:
            Canal de couleur propagé
        """
        height, width = gray_frame.shape
        color_channel = np.zeros((height, width), dtype=np.float64)
        
        for color_name, color_rgb in reference_points.items():
            # Convertir vers LAB
            color_lab = cv2.cvtColor(
                np.uint8([[color_rgb]]), cv2.COLOR_RGB2LAB
            )[0, 0]
            
            mask = self._get_color_mask(gray_frame, color_name)
            
            if channel == 0:  # Canal a
                color_channel[mask] = color_lab[1] / 255.0
            else:  # Canal b
                color_channel[mask] = color_lab[2] / 255.0
        
        # Interpolation pour les pixels non assignés
        if np.any(color_channel == 0):
            from scipy.interpolate import griddata
            
            # Points avec des valeurs assignées
            y_coords, x_coords = np.where(color_channel != 0)
            if len(y_coords) > 0:
                values = color_channel[y_coords, x_coords]
                
                # Grille complète
                grid_y, grid_x = np.mgrid[0:height, 0:width]
                
                # Interpolation
                try:
                    interpolated = griddata(
                        (y_coords, x_coords), values,
                        (grid_y, grid_x), method='linear',
                        fill_value=0
                    )
                    color_channel = interpolated
                except:
                    # Fallback: valeur constante
                    color_channel.fill(0.5)
        
        return color_channel
    
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
    
    def _levin_colorization(self, gray_frame: np.ndarray, 
                           reference_points: Dict) -> np.ndarray:
        """
        Implémente la méthode de colorisation de Levin et al. (2004).
        Propagation de couleurs à partir de points de référence.
        
        Args:
            gray_frame: Frame en niveaux de gris
            reference_points: Points de référence pour les couleurs
            
        Returns:
            Frame colorisée
        """
        # Assurer que gray_frame est en uint8
        if gray_frame.dtype != np.uint8:
            gray_frame = np.clip(gray_frame, 0, 255).astype(np.uint8)
            
        height, width = gray_frame.shape
        
        # Convertir en espace LAB pour une meilleure colorisation
        gray_3ch = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        lab_frame = cv2.cvtColor(gray_3ch, cv2.COLOR_BGR2LAB)
        
        # Initialiser les canaux de chrominance
        colorized_lab = lab_frame.copy()
        
        # Appliquer la propagation de couleurs selon Levin et al.
        for region_name, color_bgr in reference_points.items():
            # Convertir la couleur de référence en LAB
            color_bgr = np.array(color_bgr, dtype=np.uint8).reshape(1, 1, 3)
            color_lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)[0, 0]
            
            # Créer un masque pour la région
            mask = self._create_levin_mask(gray_frame, region_name)
            
            if np.any(mask):
                # Propager les couleurs selon la méthode de Levin
                colorized_lab = self._propagate_colors_levin(
                    gray_frame, colorized_lab, mask, color_lab, region_name
                )
        
        # Convertir de LAB vers BGR
        colorized = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
        
        return colorized
    
    def _create_levin_mask(self, gray_frame: np.ndarray, region_name: str) -> np.ndarray:
        """
        Crée un masque optimisé pour la méthode de Levin.
        
        Args:
            gray_frame: Frame en niveaux de gris
            region_name: Nom de la région
            
        Returns:
            Masque binaire
        """
        height, width = gray_frame.shape
        mask = np.zeros((height, width), dtype=bool)
        
        # Définir des plages d'intensité optimisées pour Levin et al.
        levin_ranges = {
            'skin': (80, 180),
            'sky': (150, 255),
            'grass': (30, 100),
            'wood': (50, 120),
            'water': (40, 140)
        }
        
        if region_name in levin_ranges:
            min_val, max_val = levin_ranges[region_name]
            intensity_mask = (gray_frame >= min_val) & (gray_frame <= max_val)
            
            # Appliquer des contraintes spatiales selon Levin et al.
            if region_name == 'sky':
                # Privilégier la partie supérieure
                spatial_weight = np.linspace(1.0, 0.3, height).reshape(-1, 1)
                intensity_mask = intensity_mask & (np.random.random((height, width)) < spatial_weight)
                mask[:height//2, :] = intensity_mask[:height//2, :]
            elif region_name == 'grass':
                # Privilégier la partie inférieure
                spatial_weight = np.linspace(0.3, 1.0, height).reshape(-1, 1)
                intensity_mask = intensity_mask & (np.random.random((height, width)) < spatial_weight)
                mask[height//2:, :] = intensity_mask[height//2:, :]
            else:
                mask = intensity_mask
        
        # Appliquer un filtrage morphologique selon Levin et al.
        if np.any(mask):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask.astype(bool)
    
    def _propagate_colors_levin(self, gray_frame: np.ndarray, lab_frame: np.ndarray,
                               mask: np.ndarray, color_lab: np.ndarray,
                               region_name: str) -> np.ndarray:
        """
        Propage les couleurs selon la méthode de Levin et al. (2004).
        
        Args:
            gray_frame: Frame en niveaux de gris
            lab_frame: Frame en espace LAB
            mask: Masque de la région
            color_lab: Couleur de référence en LAB
            region_name: Nom de la région
            
        Returns:
            Frame LAB avec couleurs propagées
        """
        height, width = gray_frame.shape
        result_lab = lab_frame.copy()
        
        # Calculer les gradients d'intensité (clé de la méthode de Levin)
        grad_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normaliser les gradients
        gradient_magnitude = gradient_magnitude / (np.max(gradient_magnitude) + 1e-8)
        
        # Créer un masque étendu basé sur la similarité d'intensité
        seed_points = np.where(mask)
        if len(seed_points[0]) > 0:
            # Calculer la moyenne d'intensité des points de référence
            mean_intensity = np.mean(gray_frame[mask])
            
            # Propager selon la similitude d'intensité et les gradients
            for i in range(height):
                for j in range(width):
                    if not mask[i, j]:  # Si pas déjà colorisé
                        current_intensity = gray_frame[i, j]
                        intensity_diff = abs(current_intensity - mean_intensity)
                        
                        # Calculer la distance aux points de référence
                        min_distance = float('inf')
                        for si, sj in zip(seed_points[0], seed_points[1]):
                            distance = np.sqrt((i - si)**2 + (j - sj)**2)
                            min_distance = min(min_distance, distance)
                        
                        # Facteur de propagation basé sur Levin et al.
                        similarity_factor = np.exp(-intensity_diff / 30.0)  # Seuil d'intensité
                        distance_factor = np.exp(-min_distance / 50.0)     # Seuil de distance
                        gradient_factor = np.exp(-gradient_magnitude[i, j] * 5.0)  # Préserver les contours
                        
                        propagation_strength = similarity_factor * distance_factor * gradient_factor
                        
                        # Appliquer la couleur si la propagation est suffisante
                        if propagation_strength > 0.1:  # Seuil de propagation
                            # Mélanger avec la couleur existante
                            alpha = propagation_strength * 0.8
                            result_lab[i, j, 1] = (1 - alpha) * result_lab[i, j, 1] + alpha * color_lab[1]
                            result_lab[i, j, 2] = (1 - alpha) * result_lab[i, j, 2] + alpha * color_lab[2]
        
        return result_lab
