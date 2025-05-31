"""
Méthodes classiques de restauration vidéo.
Implémente le filtrage spatio-temporel et la colorisation semi-automatique selon Levin et al. (2004).
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import time

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
                self.device_type = config.get('device_type', 'cpu')
                
                # Initialiser le processeur optimisé
                self.optimized_processor = OptimizedVideoProcessor(
                    use_gpu=config.get('use_gpu_acceleration', False)
                )
                
            except Exception as e:
                print(f"Warning: Could not initialize GPU acceleration: {e}")
                self.acceleration_manager = None
                self.optimized_processor = None
    
    def restore_video(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Restaure une séquence vidéo en appliquant un filtrage spatio-temporel.
        
        Args:
            frames: Liste des frames de la vidéo
            
        Returns:
            Frames restaurées
        """
        restored_frames = []
        
        for i, frame in enumerate(frames):
            # Appliquer le filtrage bilatéral spatial
            filtered_frame = self._bilateral_filter(frame)
            
            # Appliquer le filtrage médian temporel
            temporal_filtered = self._temporal_median_filter(frames, i, filtered_frame)
            
            restored_frames.append(temporal_filtered)
        
        return restored_frames
    
    def _bilateral_filter(self, frame: np.ndarray) -> np.ndarray:
        """
        Applique un filtre bilatéral pour réduire le bruit tout en préservant les contours.
        
        Args:
            frame: Frame d'entrée
            
        Returns:
            Frame filtrée
        """
        if len(frame.shape) == 3:
            # Frame couleur - appliquer sur chaque canal
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
    
    def _levin_colorization(self, gray_frame: np.ndarray, reference_points: Dict) -> np.ndarray:
        """
        Implémente la colorisation selon Levin et al. (2004).
        Utilise la propagation de couleurs basée sur l'affinité chromatique.
        
        Args:
            gray_frame: Frame en niveaux de gris
            reference_points: Dictionnaire des couleurs de référence
            
        Returns:
            Frame colorisée
        """
        height, width = gray_frame.shape
        
        # Convertir en LAB pour une meilleure propagation des couleurs
        lab_frame = cv2.cvtColor(np.stack([gray_frame, gray_frame, gray_frame], axis=-1), cv2.COLOR_BGR2LAB)
        colorized = lab_frame.astype(np.float32)
        
        # Initialiser les canaux a et b à zéro
        colorized[:, :, 1] = 0  # Canal a
        colorized[:, :, 2] = 0  # Canal b
        
        # Appliquer la propagation pour chaque canal de couleur (a et b)
        for channel in range(2):  # Canaux a et b seulement
            # Créer la matrice de contraintes selon Levin et al.
            A = self._build_affinity_matrix(gray_frame)
            
            # Créer le vecteur de contraintes de couleur
            b_vector = self._create_color_constraints(gray_frame, reference_points, channel)
            
            # Résoudre le système linéaire Ax = b
            try:
                colors = spsolve(A, b_vector)
                colors = colors.reshape((height, width))
                colorized[:, :, channel + 1] = colors  # Canaux a et b
            except Exception as e:
                print(f"Erreur dans la résolution: {e}")
                # Fallback simple
                self._apply_simple_colorization(colorized, gray_frame, reference_points, channel)
        
        # Convertir de LAB vers BGR
        colorized = np.clip(colorized, 0, 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        
        return result_bgr
    
    def _build_affinity_matrix(self, gray_image: np.ndarray, window_size: int = 3) -> csc_matrix:
        """
        Construit la matrice d'affinité selon Levin et al. (2004).
        
        Args:
            gray_image: Image en niveaux de gris
            window_size: Taille de la fenêtre locale
            
        Returns:
            Matrice d'affinité sparse
        """
        height, width = gray_image.shape
        n_pixels = height * width
        
        # Listes pour construire la matrice sparse
        rows, cols, data = [], [], []
        
        for y in range(height):
            for x in range(width):
                center_idx = y * width + x
                
                # Fenêtre autour du pixel
                y_start = max(0, y - window_size // 2)
                y_end = min(height, y + window_size // 2 + 1)
                x_start = max(0, x - window_size // 2)
                x_end = min(width, x + window_size // 2 + 1)
                
                window = gray_image[y_start:y_end, x_start:x_end]
                mean_intensity = np.mean(window)
                var_intensity = np.var(window) + 1e-6  # Éviter division par zéro
                
                # Calculer les poids d'affinité
                for ny in range(y_start, y_end):
                    for nx in range(x_start, x_end):
                        neighbor_idx = ny * width + nx
                        
                        # Poids basé sur la similitude d'intensité
                        intensity_diff = abs(gray_image[ny, nx] - gray_image[y, x])
                        weight = np.exp(-intensity_diff**2 / (2 * var_intensity))
                        
                        rows.append(center_idx)
                        cols.append(neighbor_idx)
                        data.append(weight)
        
        # Construire la matrice sparse
        A = csc_matrix((data, (rows, cols)), shape=(n_pixels, n_pixels))
        
        # Normaliser les lignes
        row_sums = np.array(A.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1  # Éviter division par zéro
        D_inv = csc_matrix(np.diag(1.0 / row_sums))
        A = D_inv @ A
        
        # Matrice laplacienne
        I = csc_matrix(np.eye(n_pixels))
        L = I - A
        
        return L
    
    def _create_color_constraints(self, gray_frame: np.ndarray, 
                                reference_points: Dict, channel: int) -> np.ndarray:
        """
        Crée les contraintes de couleur pour le système linéaire.
        
        Args:
            gray_frame: Frame en niveaux de gris
            reference_points: Points de référence
            channel: Canal de couleur (0=a, 1=b)
            
        Returns:
            Vecteur de contraintes
        """
        height, width = gray_frame.shape
        b_vector = np.zeros(height * width)
        
        for region_name, color_bgr in reference_points.items():
            # Convertir la couleur en LAB
            color_bgr_array = np.array(color_bgr, dtype=np.uint8).reshape(1, 1, 3)
            color_lab = cv2.cvtColor(color_bgr_array, cv2.COLOR_BGR2LAB)[0, 0]
            
            # Créer un masque pour cette région
            mask = self._create_region_mask(gray_frame, region_name)
            pixel_indices = np.where(mask.flatten())[0]
            
            # Assigner la valeur du canal correspondant
            if channel == 0:  # Canal a
                b_vector[pixel_indices] = color_lab[1] / 255.0
            else:  # Canal b
                b_vector[pixel_indices] = color_lab[2] / 255.0
        
        return b_vector
    
    def _create_region_mask(self, gray_frame: np.ndarray, region_name: str) -> np.ndarray:
        """
        Crée un masque pour une région spécifique basé sur l'intensité.
        
        Args:
            gray_frame: Frame en niveaux de gris
            region_name: Nom de la région
            
        Returns:
            Masque binaire
        """
        height, width = gray_frame.shape
        mask = np.zeros((height, width), dtype=bool)
        
        # Définir les gammes d'intensité pour chaque région
        intensity_ranges = {
            'sky': (180, 255),
            'grass': (60, 120),
            'skin': (120, 200),
            'water': (40, 100),
            'wood': (80, 140)
        }
        
        if region_name in intensity_ranges:
            min_val, max_val = intensity_ranges[region_name]
            intensity_mask = (gray_frame >= min_val) & (gray_frame <= max_val)
            
            # Appliquer des contraintes spatiales selon la région
            if region_name == 'sky':
                # Privilégier la partie haute de l'image
                spatial_weight = np.linspace(1.0, 0.3, height).reshape(-1, 1)
                intensity_mask = intensity_mask & (np.random.random((height, width)) < spatial_weight)
            elif region_name == 'grass':
                # Privilégier la partie basse
                spatial_weight = np.linspace(0.3, 1.0, height).reshape(-1, 1)
                intensity_mask = intensity_mask & (np.random.random((height, width)) < spatial_weight)
            elif region_name == 'skin':
                # Zone centrale pour les visages
                center_y, center_x = height // 2, width // 2
                mask[center_y-height//4:center_y+height//4, center_x-width//4:center_x+width//4] = \
                    (gray_frame[center_y-height//4:center_y+height//4, center_x-width//4:center_x+width//4] < 180)
            elif region_name == 'water':
                # Eau généralement sombre
                mask = intensity_mask & (gray_frame < 120)
            elif region_name == 'wood':
                # Bois avec intensité moyenne-faible
                mask = intensity_mask & (gray_frame > 70) & (gray_frame < 150)
            else:
                mask = intensity_mask
        
        return mask
    
    def _apply_simple_colorization(self, colorized: np.ndarray, gray_frame: np.ndarray, 
                                 reference_points: Dict, channel: int):
        """
        Applique une colorisation simple en cas d'échec de la méthode de Levin.
        
        Args:
            colorized: Frame LAB à modifier
            gray_frame: Frame en niveaux de gris
            reference_points: Points de référence
            channel: Canal de couleur
        """
        height, width = gray_frame.shape
        
        for region_name, color_bgr in reference_points.items():
            # Convertir la couleur
            color_bgr_array = np.array(color_bgr, dtype=np.uint8).reshape(1, 1, 3)
            color_lab = cv2.cvtColor(color_bgr_array, cv2.COLOR_BGR2LAB)[0, 0]
            
            # Créer un masque simple
            mask = self._create_region_mask(gray_frame, region_name)
            
            # Appliquer la couleur
            if channel == 0:  # Canal a
                colorized[:, :, 1][mask] = color_lab[1]
            else:  # Canal b
                colorized[:, :, 2][mask] = color_lab[2]
    
    def _generate_default_colors(self) -> Dict:
        """
        Génère des couleurs par défaut pour la colorisation.
        
        Returns:
            Dictionnaire des couleurs de référence
        """
        return {
            'sky': [135, 206, 235],      # Bleu ciel
            'grass': [34, 139, 34],      # Vert forêt
            'skin': [255, 220, 177],     # Couleur peau
            'water': [0, 100, 200],      # Bleu eau
            'wood': [139, 69, 19]        # Marron bois
        }
    
    def enhance_temporal_consistency(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Améliore la cohérence temporelle des frames colorisées.
        
        Args:
            frames: Frames colorisées
            
        Returns:
            Frames avec cohérence temporelle améliorée
        """
        if len(frames) < 2:
            return frames
        
        enhanced_frames = [frames[0]]  # Première frame inchangée
        
        for i in range(1, len(frames)):
            prev_frame = enhanced_frames[i-1]
            curr_frame = frames[i]
            
            # Calculer la similarité entre frames consécutives
            similarity = cv2.matchTemplate(
                cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY),
                cv2.TM_CCOEFF_NORMED
            )
            
            # Si les frames sont très similaires, appliquer un lissage temporel
            if np.max(similarity) > 0.8:
                alpha = 0.7  # Poids pour la frame précédente
                smoothed_frame = cv2.addWeighted(curr_frame, 1-alpha, prev_frame, alpha, 0)
                enhanced_frames.append(smoothed_frame)
            else:
                enhanced_frames.append(curr_frame)
        
        return enhanced_frames
    
    def get_processing_info(self) -> Dict:
        """
        Retourne des informations sur les capacités de traitement.
        
        Returns:
            Dictionnaire avec les informations de traitement
        """
        info = {
            'acceleration_available': self.acceleration_manager is not None,
            'device_type': self.device_type,
            'optimized_processor': self.optimized_processor is not None,
            'features': ['Bilateral Filtering', 'Temporal Median', 'Levin Colorization']
        }
        
        if self.acceleration_manager:
            device_info = self.acceleration_manager.get_device_info()
            info.update(device_info)
            if device_info.get('device_type') == 'cuda':
                info['features'].append('GPU Acceleration')
        
        if self.optimized_processor:
            info['features'].append('Batch Processing')
        else:
            info['features'].append('Sequential Processing')
        
        return info
    
    def process_with_metrics(self, frames: List[np.ndarray], 
                           method: str = 'restore') -> Dict:
        """
        Traite les frames avec mesure des performances.
        
        Args:
            frames: Frames à traiter
            method: Méthode à utiliser ('restore' ou 'colorize')
            
        Returns:
            Dictionnaire avec résultats et métriques
        """
        start_time = time.time()
        
        if method == 'restore':
            processed_frames = self.restore_video(frames)
            method_name = 'Classical Restoration'
        elif method == 'colorize':
            processed_frames = self.colorize_video(frames)
            method_name = 'Levin Colorization'
        else:
            raise ValueError(f"Unknown method: {method}")
        
        end_time = time.time()
        
        return {
            'frames': processed_frames,
            'method': method_name,
            'processing_time': end_time - start_time,
            'frames_processed': len(frames),
            'fps': len(frames) / (end_time - start_time),
            'device_type': self.device_type
        }
