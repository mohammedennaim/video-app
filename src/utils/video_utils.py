"""
Utilitaires pour le traitement vidéo avec OpenCV.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import os
from pathlib import Path

class VideoProcessor:
    """Classe pour charger, traiter et sauvegarder des vidéos."""
    
    def __init__(self, video_path: str):
        """
        Initialise le processeur vidéo.
        
        Args:
            video_path: Chemin vers la vidéo à traiter
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frames = self._load_frames()
    
    def _load_frames(self) -> List[np.ndarray]:
        """Charge toutes les frames de la vidéo."""
        frames = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame)
        self.cap.release()
        return frames
    
    def save_video(self, frames: List[np.ndarray], output_path: str, 
                   fps: Optional[float] = None) -> bool:
        """
        Sauvegarde les frames en vidéo.
        
        Args:
            frames: Liste des frames à sauvegarder
            output_path: Chemin de sortie
            fps: FPS de la vidéo (utilise l'original si None)
            
        Returns:
            bool: True si succès
        """
        if not frames:
            return False
        
        if fps is None:
            fps = self.fps
          # Créer le répertoire de sortie
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create directory if there's a path
            os.makedirs(output_dir, exist_ok=True)
        
        # Configuration du codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frames[0].shape[:2]
        
        # Si les frames sont en noir et blanc, les convertir en couleur
        if len(frames[0].shape) == 2:
            frames = [cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) for frame in frames]
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        return True
    
    def get_info(self) -> dict:
        """Retourne les informations de la vidéo."""
        return {
            'path': self.video_path,
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'frame_count': self.frame_count,
            'duration': self.frame_count / self.fps if self.fps > 0 else 0
        }

def convert_to_grayscale(frames: List[np.ndarray]) -> List[np.ndarray]:
    """Convertit les frames en niveaux de gris."""
    return [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

def resize_frames(frames: List[np.ndarray], 
                  target_size: Tuple[int, int]) -> List[np.ndarray]:
    """
    Redimensionne les frames.
    
    Args:
        frames: Liste des frames
        target_size: Taille cible (width, height)
        
    Returns:
        Liste des frames redimensionnées
    """
    return [cv2.resize(frame, target_size) for frame in frames]

def extract_frames(video_path: str, output_dir: str, 
                   frame_interval: int = 1) -> List[str]:
    """
    Extrait les frames d'une vidéo.
    
    Args:
        video_path: Chemin vers la vidéo
        output_dir: Répertoire de sortie
        frame_interval: Intervalle entre les frames (1 = toutes)
        
    Returns:
        Liste des chemins des frames extraites
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        
        frame_idx += 1
    
    cap.release()
    return frame_paths

def create_video_from_frames(frame_dir: str, output_path: str, 
                           fps: float = 30.0) -> bool:
    """
    Crée une vidéo à partir d'images.
    
    Args:
        frame_dir: Répertoire contenant les frames
        output_path: Chemin de sortie
        fps: FPS de la vidéo
        
    Returns:
        bool: True si succès
    """
    frame_files = sorted([f for f in os.listdir(frame_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not frame_files:
        return False
    
    # Lire la première frame pour obtenir les dimensions
    first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    height, width = first_frame.shape[:2]
    
    # Créer le writer vidéo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        out.write(frame)
    
    out.release()
    return True

def calculate_optical_flow(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """
    Calcule le flux optique entre deux frames.
    
    Args:
        frame1: Première frame
        frame2: Deuxième frame
        
    Returns:
        Flux optique
    """
    # Convertir en niveaux de gris si nécessaire
    if len(frame1.shape) == 3:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = frame1
        
    if len(frame2.shape) == 3:
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = frame2
    
    # Calculer le flux optique dense
    flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, None, None)
    return flow
