"""
Module d'optimisation des performances pour le traitement vidéo.
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

class OptimizedVideoProcessor:
    """Processeur vidéo optimisé pour de meilleures performances."""
    
    def __init__(self, use_gpu: bool = True, num_workers: Optional[int] = None):
        """
        Initialise le processeur optimisé.
        
        Args:
            use_gpu: Utiliser l'accélération GPU si disponible
            num_workers: Nombre de workers pour le traitement parallèle
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.num_workers = num_workers or mp.cpu_count()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
    def process_frames_batch(self, frames: List[np.ndarray], 
                           processing_func, batch_size: int = 8) -> List[np.ndarray]:
        """
        Traite les frames par batch pour optimiser les performances.
        
        Args:
            frames: Liste des frames à traiter
            processing_func: Fonction de traitement à appliquer
            batch_size: Taille des batches
            
        Returns:
            Liste des frames traitées
        """
        processed_frames = []
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            
            if self.use_gpu:
                # Traitement GPU optimisé
                batch_result = self._process_gpu_batch(batch, processing_func)
            else:
                # Traitement CPU parallèle
                batch_result = self._process_cpu_batch(batch, processing_func)
                
            processed_frames.extend(batch_result)
            
        return processed_frames
        
    def _process_gpu_batch(self, batch: List[np.ndarray], processing_func) -> List[np.ndarray]:
        """Traite un batch sur GPU."""
        try:
            # Convertir en tensor GPU
            batch_tensor = torch.stack([
                torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                for frame in batch
            ]).to(self.device)
            
            # Traitement avec la fonction fournie
            with torch.no_grad():
                result_tensor = processing_func(batch_tensor)
            
            # Reconvertir en numpy
            result_frames = []
            for i in range(result_tensor.shape[0]):
                frame = result_tensor[i].permute(1, 2, 0).cpu().numpy()
                frame = (frame * 255).astype(np.uint8)
                result_frames.append(frame)
                
            return result_frames
            
        except Exception as e:
            print(f"Erreur GPU, fallback CPU: {e}")
            return self._process_cpu_batch(batch, processing_func)
    
    def _process_cpu_batch(self, batch: List[np.ndarray], processing_func) -> List[np.ndarray]:
        """Traite un batch sur CPU avec parallélisme."""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(processing_func, frame) for frame in batch]
            results = [future.result() for future in futures]
        return results
        
    def optimize_frame_size(self, frame: np.ndarray, 
                          max_dimension: int = 1024) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Optimise la taille des frames pour améliorer les performances.
        
        Args:
            frame: Frame à optimiser
            max_dimension: Dimension maximale
            
        Returns:
            Frame redimensionnée et dimensions originales
        """
        original_shape = frame.shape[:2]
        h, w = original_shape
        
        # Calculer le facteur de redimensionnement
        scale_factor = min(max_dimension / max(h, w), 1.0)
        
        if scale_factor < 1.0:
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            optimized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return optimized_frame, original_shape
        
        return frame, original_shape
        
    def restore_frame_size(self, frame: np.ndarray, 
                          original_shape: Tuple[int, int]) -> np.ndarray:
        """Restaure la taille originale d'une frame."""
        if frame.shape[:2] != original_shape:
            return cv2.resize(frame, (original_shape[1], original_shape[0]), 
                            interpolation=cv2.INTER_LANCZOS4)
        return frame
        
    def memory_efficient_processing(self, frames: List[np.ndarray],
                                  processing_func, chunk_size: int = 50) -> List[np.ndarray]:
        """
        Traitement économe en mémoire pour de grandes séquences.
        
        Args:
            frames: Liste des frames
            processing_func: Fonction de traitement
            chunk_size: Taille des chunks pour éviter les problèmes de mémoire
            
        Returns:
            Frames traitées
        """
        processed_frames = []
        
        for i in range(0, len(frames), chunk_size):
            chunk = frames[i:i + chunk_size]
            chunk_result = self.process_frames_batch(chunk, processing_func)
            processed_frames.extend(chunk_result)
            
            # Nettoyage mémoire
            if self.use_gpu:
                torch.cuda.empty_cache()
                
        return processed_frames