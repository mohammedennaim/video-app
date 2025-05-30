"""
Module de gestion de l'accélération GPU pour optimiser les performances.
"""

import torch
import cv2
from typing import Dict, Any

class AccelerationManager:
    """Gestionnaire d'accélération GPU pour améliorer les performances."""
    
    def __init__(self):
        """Initialise le gestionnaire d'accélération."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_available = torch.cuda.is_available()
        self.optimizations_applied = False
        
    def apply_optimizations(self):
        """Applique les optimisations GPU disponibles."""
        if self.gpu_available:
            # Optimisations PyTorch
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Configuration mémoire GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        self.optimizations_applied = True
        
    def get_processing_config(self) -> Dict[str, Any]:
        """Retourne la configuration optimale pour le traitement."""
        config = {
            'device': self.device,
            'batch_size': 8 if self.gpu_available else 2,
            'num_workers': 4 if self.gpu_available else 1,
            'pin_memory': self.gpu_available,
            'use_gpu_acceleration': self.gpu_available
        }
        return config
        
    def optimize_opencv(self):
        """Optimise OpenCV pour l'utilisation GPU."""
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            cv2.setUseOptimized(True)
            cv2.setNumThreads(cv2.getNumberOfCPUs())
            
    def get_device_info(self) -> Dict[str, str]:
        """Retourne les informations sur le dispositif."""
        info = {
            'device_type': 'cuda' if self.gpu_available else 'cpu',
            'device_name': torch.cuda.get_device_name(0) if self.gpu_available else 'CPU'
        }
        
        if self.gpu_available:
            info['memory_total'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            info['compute_capability'] = f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"
            
        return info