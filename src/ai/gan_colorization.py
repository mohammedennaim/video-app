"""
Colorisation automatique basée sur les GANs.
Implémentation d'un modèle inspiré de DeOldify et techniques Deep Learning.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import os
from pathlib import Path
import requests
from PIL import Image
import torchvision.transforms as transforms

# Import des modules d'optimisation
try:
    from ..utils.gpu_acceleration import AccelerationManager
    from ..utils.performance import OptimizedVideoProcessor
except ImportError:
    # Fallback si les modules ne sont pas disponibles
    AccelerationManager = None
    OptimizedVideoProcessor = None

class UNetGenerator(nn.Module):
    """Générateur U-Net pour la colorisation."""
    
    def __init__(self, input_channels=1, output_channels=2, features=64):
        super(UNetGenerator, self).__init__()
        
        # Encoder (Contracting path)
        self.enc1 = self._conv_block(input_channels, features)
        self.enc2 = self._conv_block(features, features*2)
        self.enc3 = self._conv_block(features*2, features*4)
        self.enc4 = self._conv_block(features*4, features*8)
        
        # Bottleneck
        self.bottleneck = self._conv_block(features*8, features*16)
        
        # Decoder (Expanding path)
        self.dec4 = self._upconv_block(features*16, features*8)
        self.dec3 = self._upconv_block(features*16, features*4)
        self.dec2 = self._upconv_block(features*8, features*2)
        self.dec1 = self._upconv_block(features*4, features)
          # Output layer
        self.final = nn.Conv2d(features*2, output_channels, kernel_size=1)  # Fixed: features*2 for skip connections
        self.tanh = nn.Tanh()
        
        self.pool = nn.MaxPool2d(2)
    
    def _conv_block(self, in_channels, out_channels):
        """Bloc de convolution standard."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _upconv_block(self, in_channels, out_channels):
        """Bloc de déconvolution."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder avec skip connections
        dec4 = self.dec4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        
        dec3 = self.dec3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        
        dec2 = self.dec2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        
        dec1 = self.dec1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        
        # Output
        output = self.final(dec1)
        return self.tanh(output)

class GANColorization:
    """Classe principale pour la colorisation GAN."""
      def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialise le système de colorisation GAN avec optimisations.
        
        Args:
            model_path: Chemin vers un modèle pré-entraîné
            device: Device à utiliser ('cuda', 'cpu', ou None pour auto)
        """
        # Initialiser le gestionnaire d'accélération
        self.acceleration_manager = None
        self.optimized_processor = None
        
        if AccelerationManager is not None:
            try:
                self.acceleration_manager = AccelerationManager()
                self.acceleration_manager.apply_optimizations()
                
                # Utiliser le device recommandé si non spécifié
                if device is None:
                    device_type, device_info = self.acceleration_manager.detector.get_recommended_device()
                    if device_type == 'cuda':
                        device = f"cuda:{device_info['id']}"
                    else:
                        device = 'cpu'
                
                # Configuration du processeur optimisé
                config = self.acceleration_manager.get_processing_config()
                if OptimizedVideoProcessor is not None:
                    self.optimized_processor = OptimizedVideoProcessor(
                        batch_size=config['batch_size'],
                        max_workers=config['num_workers'],
                        enable_cache=True
                    )
                    
                print(f"🚀 Accélération GPU activée - Device: {device}")
                
            except Exception as e:
                print(f"⚠️  Erreur lors de l'initialisation GPU: {e}")
                device = device or 'cpu'
        else:
            device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = torch.device(device)
        print(f"🤖 Utilisation du device: {self.device}")
        
        # Initialiser le générateur
        self.generator = UNetGenerator().to(self.device)
        
        # Charger le modèle ou initialiser les poids
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print("⚠️  Modèle pré-entraîné non trouvé, utilisation d'un modèle initialisé aléatoirement")
            self._initialize_weights()
        
        # Transformations pour preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
          # Custom clamping function since transforms.Clamp is not available in all versions
        self.inverse_normalize = transforms.Normalize(mean=[-1], std=[2])
    
    def _load_model(self, model_path: str):
        """Charge un modèle pré-entraîné."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator'])
            print(f"✅ Modèle chargé depuis: {model_path}")
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle: {e}")
            self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialise les poids du réseau."""
        for m in self.generator.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
      def colorize_video(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Colorise une séquence vidéo avec optimisations.
        
        Args:
            frames: Liste des frames en niveaux de gris ou couleur
            
        Returns:
            Frames colorisées
        """
        print("🎨 Colorisation GAN en cours...")
        
        # Utiliser le processeur optimisé si disponible
        if self.optimized_processor is not None:
            try:
                print("🚀 Utilisation du processeur optimisé")
                
                # Définir la fonction de traitement pour une frame
                def process_single_frame(frame):
                    return self._colorize_frame(frame)
                
                # Traitement optimisé avec cache et parallélisation
                colorized_frames = self.optimized_processor.process_video_optimized(
                    frames=frames,
                    process_func=process_single_frame,
                    operation_id="gan_colorization",
                    use_parallel=len(frames) > 4
                )
                
                print(f"✅ Colorisation optimisée terminée: {len(colorized_frames)} frames")
                return colorized_frames
                
            except Exception as e:
                print(f"⚠️  Erreur processeur optimisé: {e}, fallback vers traitement standard")
        
        # Traitement standard (fallback)
        colorized_frames = []
        self.generator.eval()
        
        with torch.no_grad():
            for i, frame in enumerate(frames):
                # Coloriser la frame
                colorized = self._colorize_frame(frame)
                colorized_frames.append(colorized)
                
                if (i + 1) % 10 == 0:
                    print(f"  Progression: {i + 1}/{len(frames)} frames")
        
        return colorized_frames
    
    def _colorize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Colorise une frame individuelle.
        
        Args:
            frame: Frame à coloriser
            
        Returns:
            Frame colorisée
        """
        original_shape = frame.shape[:2]
        
        # Convertir en niveaux de gris si nécessaire
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame.copy()
        
        # Convertir en PIL Image et préprocesser
        pil_image = Image.fromarray(gray_frame).convert('L')
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Prédiction
        with torch.no_grad():
            ab_output = self.generator(input_tensor)
        
        # Post-processing
        colorized = self._lab_to_rgb(input_tensor, ab_output, original_shape)
        
        return colorized
    
    def _lab_to_rgb(self, l_tensor: torch.Tensor, ab_tensor: torch.Tensor, 
                    original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Convertit les canaux LAB en image RGB.
        
        Args:
            l_tensor: Canal L (luminance)
            ab_tensor: Canaux AB (chrominance)
            original_shape: Forme originale de l'image
            
        Returns:
            Image RGB
        """        # Dénormaliser les tenseurs
        l_channel = l_tensor.squeeze().cpu() 
        # Manual denormalization: inverse of transforms.Normalize(mean=[-1], std=[2])
        l_channel = (l_channel + 1) / 2  # Convert from [-1,1] to [0,1]
        l_channel = torch.clamp(l_channel, 0, 1)  # Ensure values are in [0,1]
        ab_channels = ab_tensor.squeeze().cpu() * 128  # Dénormaliser AB
        
        # Redimensionner si nécessaire
        if l_channel.shape[-2:] != ab_channels.shape[-2:]:
            ab_channels = F.interpolate(
                ab_channels.unsqueeze(0), 
                size=l_channel.shape[-2:], 
                mode='bilinear'
            ).squeeze(0)
        
        # Combiner les canaux LAB
        lab_image = torch.cat([
            l_channel.unsqueeze(0) * 100,  # L: 0-100
            ab_channels  # AB: -128 to 127
        ], dim=0)
        
        # Convertir en numpy
        lab_np = lab_image.permute(1, 2, 0).numpy()
        
        # Convertir LAB vers RGB
        lab_np = lab_np.astype(np.float32)
        rgb_image = cv2.cvtColor(lab_np, cv2.COLOR_LAB2RGB)
        
        # Normaliser et convertir en uint8
        rgb_image = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)
        
        # Redimensionner à la taille originale
        if rgb_image.shape[:2] != original_shape:
            rgb_image = cv2.resize(rgb_image, (original_shape[1], original_shape[0]))
        
        # Convertir RGB vers BGR pour OpenCV
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        return bgr_image
    
    def enhance_temporal_consistency(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Améliore la cohérence temporelle des couleurs.
        
        Args:
            frames: Frames colorisées
            
        Returns:
            Frames avec cohérence temporelle améliorée
        """
        if len(frames) < 3:
            return frames
        
        print("🔄 Amélioration de la cohérence temporelle...")
        enhanced_frames = [frames[0]]  # Première frame inchangée
        
        # Paramètres pour le lissage temporel
        alpha = 0.7  # Poids de la frame courante
        beta = 0.3   # Poids de la frame précédente
        
        for i in range(1, len(frames)):
            current_frame = frames[i].astype(np.float32)
            previous_frame = enhanced_frames[i-1].astype(np.float32)
            
            # Lissage temporel simple
            enhanced_frame = alpha * current_frame + beta * previous_frame
            enhanced_frame = np.clip(enhanced_frame, 0, 255).astype(np.uint8)
            
            enhanced_frames.append(enhanced_frame)
        
        return enhanced_frames
    
    def download_pretrained_model(self, save_path: str = "models/colorization_model.pth"):
        """
        Télécharge un modèle pré-entraîné (simulé).
        
        Args:
            save_path: Chemin de sauvegarde du modèle
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print("⬇️  Création d'un modèle de démonstration...")
        
        # Créer un checkpoint factice pour la démonstration
        checkpoint = {
            'generator': self.generator.state_dict(),
            'epoch': 100,
            'loss': 0.1
        }
        
        torch.save(checkpoint, save_path)
        print(f"✅ Modèle sauvegardé dans: {save_path}")
        
        return save_path

class SimpleColorization:
    """Version simplifiée pour les systèmes sans GPU puissant."""
    
    def __init__(self):
        """Initialise la colorisation simple."""
        pass
    
    def colorize_video(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Colorise avec une méthode simple basée sur des modèles pré-définis.
        
        Args:
            frames: Frames à coloriser
            
        Returns:
            Frames colorisées
        """
        print("🎨 Colorisation simple en cours...")
        colorized_frames = []
        
        for i, frame in enumerate(frames):
            colorized = self._simple_colorize(frame)
            colorized_frames.append(colorized)
            
            if (i + 1) % 10 == 0:
                print(f"  Progression: {i + 1}/{len(frames)} frames")
        
        return colorized_frames
    
    def _simple_colorize(self, frame: np.ndarray) -> np.ndarray:
        """
        Colorisation simple basée sur l'intensité.
        
        Args:
            frame: Frame à coloriser
            
        Returns:
            Frame colorisée
        """
        # Convertir en niveaux de gris si nécessaire
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Créer une version colorée basique
        colorized = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
        
        # Mélanger avec une version sépia pour un effet plus naturel
        sepia_kernel = np.array([[0.272, 0.534, 0.131],
                                [0.349, 0.686, 0.168],
                                [0.393, 0.769, 0.189]])
        
        sepia = cv2.transform(colorized, sepia_kernel)
        
        # Mélanger les deux versions
        result = cv2.addWeighted(colorized, 0.6, sepia, 0.4, 0)
        
        return result
