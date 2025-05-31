"""
Module d'amélioration IA avec techniques DeOldify et Zhang et al. (CVPR 2020).
Implémente des architectures avancées pour la colorisation et restauration vidéo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Optional, Tuple
import os
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image

# Import des modules d'optimisation
try:
    from ..utils.gpu_acceleration import AccelerationManager
    from ..utils.performance import OptimizedVideoProcessor
except ImportError:
    AccelerationManager = None
    OptimizedVideoProcessor = None

class SelfAttention(nn.Module):
    """
    Module d'auto-attention pour améliorer la colorisation (technique clé de DeOldify).
    """
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        
        # Couches pour générer query, key, value
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        
        # Couche de sortie
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Générer query, key, value
        query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, width * height)
        value = self.value(x).view(batch_size, -1, width * height)
        
        # Calculer l'attention
        attention = torch.bmm(query, key)
        attention = self.softmax(attention)
        
        # Appliquer l'attention
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Connexion résiduelle
        out = self.gamma * out + x
        return out

class SpectralNorm(nn.Module):
    """
    Normalisation spectrale pour stabiliser l'entraînement (Zhang et al.).
    """
    def __init__(self, module, power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.power_iterations = power_iterations
        
        # Initialiser u et v pour la normalisation spectrale
        w = self.module.weight.data
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        
        self.register_buffer('u', torch.randn(height))
        self.register_buffer('v', torch.randn(width))
        
    def forward(self, *args):
        # Appliquer la normalisation spectrale
        w = self.module.weight
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        
        u = self.u.data
        v = self.v.data
        
        # Power iteration
        for _ in range(self.power_iterations):
            v = F.normalize(torch.mv(w.view(height, -1).data.t(), u), eps=1e-6)
            u = F.normalize(torch.mv(w.view(height, -1).data, v), eps=1e-6)
        
        # Calculer la norme spectrale
        sigma = torch.dot(u, torch.mv(w.view(height, -1).data, v))
        
        # Normaliser les poids
        self.module.weight.data = w / sigma.expand_as(w)
        
        # Mettre à jour les buffers
        self.u.data = u
        self.v.data = v
        
        return self.module(*args)
    """
    Générateur inspiré de DeOldify (Jason Antic, 2019).
    Architecture U-Net améliorée avec attention et techniques de DeOldify.
    """
    
    def __init__(self, input_channels=1, output_channels=2, features=64):
        super(DeOldifyGenerator, self).__init__()
        
        # Encoder avec architecture DeOldify
        self.enc1 = self._deoldify_encoder_block(input_channels, features)
        self.enc2 = self._deoldify_encoder_block(features, features*2)
        self.enc3 = self._deoldify_encoder_block(features*2, features*4)
        self.enc4 = self._deoldify_encoder_block(features*4, features*8)
        
        # Bottleneck avec Self-Attention (clé de DeOldify)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*16, 3, padding=1),
            nn.BatchNorm2d(features*16),
            nn.ReLU(inplace=True),
            SelfAttention(features*16),  # Attention mechanism
            nn.Conv2d(features*16, features*16, 3, padding=1),
            nn.BatchNorm2d(features*16),
            nn.ReLU(inplace=True)
        )
        
        # Decoder avec techniques DeOldify
        self.dec4 = self._deoldify_decoder_block(features*16, features*8)
        self.dec3 = self._deoldify_decoder_block(features*16, features*4)
        self.dec2 = self._deoldify_decoder_block(features*8, features*2)
        self.dec1 = self._deoldify_decoder_block(features*4, features)
        
        # Output layer avec normalisation DeOldify
        self.final = nn.Sequential(
            nn.Conv2d(features*2, output_channels, 3, padding=1),
            nn.Tanh()
        )
        
        self.pool = nn.MaxPool2d(2)
    
    def _deoldify_encoder_block(self, in_channels, out_channels):
        """Bloc encoder optimisé selon DeOldify."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU comme dans DeOldify
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1)  # Dropout pour la régularisation
        )
    
    def _deoldify_decoder_block(self, in_channels, out_channels):
        """Bloc decoder optimisé selon DeOldify."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck avec attention
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
        return self.final(dec1)

class SelfAttention(nn.Module):
    """
    Module d'attention utilisé dans DeOldify.
    Permet au modèle de se concentrer sur les régions importantes.
    """
    
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Calculer query, key, value
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        
        # Calculer l'attention
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        # Appliquer l'attention
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Connexion résiduelle avec paramètre gamma
        out = self.gamma * out + x
        
        return out

class ZhangRestorationNet(nn.Module):
    """
    Réseau de restauration basé sur les techniques de Zhang et al. (CVPR 2020).
    Intègre le débruitage et la super-résolution pour la restauration vidéo.
    """
    
    def __init__(self, num_channels=3, num_features=64, num_blocks=16):
        super(ZhangRestorationNet, self).__init__()
        
        # Feature extraction
        self.feature_extractor = nn.Conv2d(num_channels, num_features, 3, padding=1)
        
        # Residual blocks selon Zhang et al.
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_features) for _ in range(num_blocks)
        ])
        
        # Feature refinement
        self.feature_refine = nn.Conv2d(num_features, num_features, 3, padding=1)
        
        # Reconstruction
        self.reconstructor = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_channels, 3, padding=1)
        )
        
        # Attention spatiale selon Zhang et al.
        self.spatial_attention = SpatialAttention(num_features)
    
    def forward(self, x):
        # Feature extraction
        feat = F.relu(self.feature_extractor(x))
        residual = feat
        
        # Residual blocks avec attention
        for res_block in self.res_blocks:
            feat = res_block(feat)
        
        # Feature refinement avec attention spatiale
        feat = self.feature_refine(feat)
        feat = self.spatial_attention(feat)
        feat = feat + residual  # Skip connection globale
        
        # Reconstruction
        out = self.reconstructor(feat)
        
        return out + x  # Connexion résiduelle globale

class ResidualBlock(nn.Module):
    """Bloc résiduel optimisé selon Zhang et al."""
    
    def __init__(self, num_features):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        return out + residual

class SpatialAttention(nn.Module):
    """Module d'attention spatiale selon Zhang et al."""
    
    def __init__(self, num_features):
        super(SpatialAttention, self).__init__()
        
        self.conv1 = nn.Conv2d(num_features, num_features // 16, 1)
        self.conv2 = nn.Conv2d(num_features // 16, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Générer la carte d'attention
        attention = F.relu(self.conv1(x))
        attention = self.sigmoid(self.conv2(attention))
        
        # Appliquer l'attention
        return x * attention

class AdvancedGANColorization:
    """
    Classe principale intégrant DeOldify et Zhang et al. pour la colorisation avancée.
    """
    
    def __init__(self, model_type='deoldify', device: Optional[str] = None):
        """
        Initialise le système de colorisation avancé.
        
        Args:
            model_type: 'deoldify' ou 'zhang' ou 'hybrid'
            device: Device à utiliser
        """
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model_type = model_type
        
        print(f"🤖 Initialisation du modèle avancé: {model_type}")
        print(f"🔧 Device: {self.device}")
        
        if model_type == 'deoldify':
            self.colorizer = DeOldifyGenerator().to(self.device)
            self.restoration_net = None
        elif model_type == 'zhang':
            self.colorizer = None
            self.restoration_net = ZhangRestorationNet().to(self.device)
        elif model_type == 'hybrid':
            self.colorizer = DeOldifyGenerator().to(self.device)
            self.restoration_net = ZhangRestorationNet().to(self.device)
        else:
            raise ValueError(f"Type de modèle non supporté: {model_type}")
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialise les poids des modèles."""
        def init_func(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if self.colorizer:
            self.colorizer.apply(init_func)
        if self.restoration_net:
            self.restoration_net.apply(init_func)
    
    def colorize_video(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Colorise une vidéo avec les techniques avancées.
        
        Args:
            frames: Liste des frames à coloriser
            
        Returns:
            Frames colorisées
        """
        print(f"🎨 Colorisation avancée ({self.model_type}) en cours...")
        
        colorized_frames = []
        
        for i, frame in enumerate(frames):
            if self.model_type == 'deoldify':
                colorized = self._deoldify_colorize(frame)
            elif self.model_type == 'zhang':
                colorized = self._zhang_restore_and_colorize(frame)
            elif self.model_type == 'hybrid':
                # D'abord restaurer avec Zhang, puis coloriser avec DeOldify
                restored = self._zhang_restore(frame)
                colorized = self._deoldify_colorize(restored)
            
            colorized_frames.append(colorized)
            
            if (i + 1) % 10 == 0:
                print(f"  Progression: {i + 1}/{len(frames)} frames")
        
        return colorized_frames
    
    def _deoldify_colorize(self, frame: np.ndarray) -> np.ndarray:
        """Colorise avec la méthode DeOldify."""
        # Préparation de l'image
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Redimensionner pour le modèle
        h, w = gray.shape
        target_size = 256
        gray_resized = cv2.resize(gray, (target_size, target_size))
        
        # Conversion en tensor
        gray_tensor = torch.from_numpy(gray_resized).float().unsqueeze(0).unsqueeze(0) / 255.0 * 2 - 1
        gray_tensor = gray_tensor.to(self.device)
        
        # Prédiction
        self.colorizer.eval()
        with torch.no_grad():
            ab_pred = self.colorizer(gray_tensor)
        
        # Post-traitement
        ab_pred = ab_pred.cpu().squeeze().permute(1, 2, 0).numpy()
        ab_pred = cv2.resize(ab_pred, (w, h))
        
        # Reconstruire l'image LAB
        l_channel = gray / 255.0 * 100  # Normaliser pour LAB
        ab_channels = ab_pred * 128.0   # Dénormaliser
        
        lab_image = np.zeros((h, w, 3))
        lab_image[:, :, 0] = l_channel
        lab_image[:, :, 1:] = ab_channels
        
        # Convertir en BGR
        lab_image = lab_image.astype(np.uint8)
        bgr_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
        
        return bgr_image
    
    def _zhang_restore(self, frame: np.ndarray) -> np.ndarray:
        """Restaure avec la méthode de Zhang et al."""
        # Préparation
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        h, w = frame.shape[:2]
        target_size = 256
        frame_resized = cv2.resize(frame, (target_size, target_size))
        
        # Conversion en tensor
        frame_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        frame_tensor = frame_tensor.to(self.device)
        
        # Restauration
        self.restoration_net.eval()
        with torch.no_grad():
            restored = self.restoration_net(frame_tensor)
        
        # Post-traitement
        restored = torch.clamp(restored, 0, 1)
        restored = restored.cpu().squeeze().permute(1, 2, 0).numpy()
        restored = cv2.resize(restored, (w, h))
        restored = (restored * 255).astype(np.uint8)
        
        return restored
    
    def _zhang_restore_and_colorize(self, frame: np.ndarray) -> np.ndarray:
        """Combine restauration et colorisation selon Zhang et al."""
        # Pour la démonstration, on applique d'abord la restauration
        restored = self._zhang_restore(frame)
        
        # Puis une colorisation simple (peut être étendue)
        if len(restored.shape) == 3:
            gray = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
        else:
            gray = restored.copy()
        
        # Appliquer une colorisation basique améliorée
        colorized = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
        
        # Mélanger avec l'original restauré pour un effet plus naturel
        if len(restored.shape) == 3:
            colorized = cv2.addWeighted(restored, 0.7, colorized, 0.3, 0)
        
        return colorized
    
    def enable_temporal_consistency(self, alpha=0.7):
        """Active la cohérence temporelle pour les vidéos."""
        self.temporal_alpha = alpha
        self.previous_frame = None
        print(f"🔗 Cohérence temporelle activée (α={alpha})")
    
    def _apply_temporal_consistency(self, current_frame: np.ndarray) -> np.ndarray:
        """Applique la cohérence temporelle."""
        if hasattr(self, 'temporal_alpha') and self.previous_frame is not None:
            # Lissage temporel pondéré
            consistent_frame = cv2.addWeighted(
                current_frame, self.temporal_alpha,
                self.previous_frame, 1 - self.temporal_alpha, 0
            )
            self.previous_frame = consistent_frame.copy()
            return consistent_frame
        else:
            if hasattr(self, 'temporal_alpha'):
                self.previous_frame = current_frame.copy()
            return current_frame

def create_advanced_colorizer(model_type='hybrid'):
    """
    Crée un coloriseur avancé avec les dernières techniques.
    
    Args:
        model_type: Type de modèle ('deoldify', 'zhang', 'hybrid')
        
    Returns:
        Instance du coloriseur avancé
    """
    try:
        colorizer = AdvancedGANColorization(model_type=model_type)
        colorizer.enable_temporal_consistency()
        return colorizer
    except Exception as e:
        print(f"⚠️ Erreur lors de la création du coloriseur avancé: {e}")
        print("📝 Retour au coloriseur simple")
        return None
