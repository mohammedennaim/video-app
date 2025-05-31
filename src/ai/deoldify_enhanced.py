"""
Module d'amélioration IA avec techniques DeOldify et Zhang et al. (CVPR 2020).
Implémente des architectures avancées pour la colorisation et restauration vidéo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List
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


class TemporalConsistencyModule(nn.Module):
    """
    Module pour assurer la cohérence temporelle (Zhang et al., CVPR 2020).
    """
    def __init__(self, channels):
        super(TemporalConsistencyModule, self).__init__()
        self.channels = channels
        
        # Convolutions 3D pour capturer les relations temporelles
        self.temporal_conv = nn.Conv3d(channels, channels, (3, 3, 3), padding=(1, 1, 1))
        self.temporal_norm = nn.BatchNorm3d(channels)
        
        # Attention temporelle
        self.temporal_attention = nn.Sequential(
            nn.Conv3d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor de forme (batch, channels, frames, height, width)
        """
        # Convolution temporelle
        temporal_features = F.relu(self.temporal_norm(self.temporal_conv(x)))
        
        # Attention temporelle
        attention_weights = self.temporal_attention(temporal_features)
        
        # Appliquer l'attention
        enhanced_features = temporal_features * attention_weights
        
        # Connexion résiduelle
        return x + enhanced_features


class DeOldifyGenerator(nn.Module):
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
          # Decoder avec techniques DeOldify (ajustés pour les skip connections)
        self.dec4 = self._deoldify_decoder_block(features*16 + features*8, features*8)  # bottleneck + enc4
        self.dec3 = self._deoldify_decoder_block(features*8 + features*4, features*4)   # dec4 + enc3
        self.dec2 = self._deoldify_decoder_block(features*4 + features*2, features*2)   # dec3 + enc2
        self.dec1 = self._deoldify_decoder_block(features*2 + features, features)       # dec2 + enc1
          # Output layer avec normalisation DeOldify
        self.final = nn.Sequential(
            nn.Conv2d(features, output_channels, 3, padding=1),
            nn.Tanh()
        )
        
        # Pooling et upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def _deoldify_encoder_block(self, in_channels, out_channels):
        """Bloc encoder optimisé selon DeOldify."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)  # Dropout pour la régularisation
        )
    
    def _deoldify_decoder_block(self, in_channels, out_channels):
        """Bloc decoder avec skip connections optimisées."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc1_pool = self.pool(enc1)
        
        enc2 = self.enc2(enc1_pool)
        enc2_pool = self.pool(enc2)
        
        enc3 = self.enc3(enc2_pool)
        enc3_pool = self.pool(enc3)
        
        enc4 = self.enc4(enc3_pool)
        enc4_pool = self.pool(enc4)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4_pool)
        
        # Decoder path avec skip connections
        dec4 = self.upsample(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upsample(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upsample(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upsample(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Output
        output = self.final(dec1)
        return output


class ZhangVideoRestorer(nn.Module):
    """
    Réseau de restauration vidéo basé sur Zhang et al. (CVPR 2020).
    Combine restauration spatiale et cohérence temporelle.
    """
    
    def __init__(self, input_channels=3, output_channels=3, features=64):
        super(ZhangVideoRestorer, self).__init__()
        
        # Extracteur de features spatial
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(input_channels, features, 7, padding=3),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(features, features*2, 3, padding=1, stride=2),
            nn.BatchNorm2d(features*2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(features*2, features*4, 3, padding=1, stride=2),
            nn.BatchNorm2d(features*4),
            nn.ReLU(inplace=True),
        )
        
        # Module de cohérence temporelle
        self.temporal_module = TemporalConsistencyModule(features*4)
        
        # Décodeur spatial
        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose2d(features*4, features*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(features*2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(features*2, features, 4, stride=2, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(features, output_channels, 7, padding=3),
            nn.Tanh()
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor de forme (batch, channels, frames, height, width)
        """
        batch, channels, frames, height, width = x.shape
        
        # Traiter chaque frame individuellement
        spatial_features = []
        for t in range(frames):
            frame = x[:, :, t, :, :]
            features = self.spatial_encoder(frame)
            spatial_features.append(features)
        
        # Stack temporal features
        spatial_features = torch.stack(spatial_features, dim=2)
        
        # Appliquer le module temporel
        enhanced_features = self.temporal_module(spatial_features)
        
        # Décoder chaque frame
        output_frames = []
        for t in range(frames):
            features = enhanced_features[:, :, t, :, :]
            restored_frame = self.spatial_decoder(features)
            output_frames.append(restored_frame)
        
        # Stack output frames
        output = torch.stack(output_frames, dim=2)
        return output


class EnhancedDeOldifyColorizer:
    """
    Coloriseur avancé combinant DeOldify et Zhang et al.
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.temporal_alpha = 0.7  # Facteur de lissage temporel
        
        # Modèles
        self.colorizer = DeOldifyGenerator(input_channels=1, output_channels=2)
        self.video_restorer = ZhangVideoRestorer()
        
        # Charger sur le device
        self.colorizer.to(self.device)
        self.video_restorer.to(self.device)
        
        # Mode évaluation
        self.colorizer.eval()
        self.video_restorer.eval()
        
        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Optimisations
        self.acceleration_manager = None
        self.optimized_processor = None
        
        if AccelerationManager and OptimizedVideoProcessor:
            try:
                self.acceleration_manager = AccelerationManager()
                self.acceleration_manager.apply_optimizations()
                self.optimized_processor = OptimizedVideoProcessor()
            except:
                pass
    
    def colorize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Colorise une frame individuelle avec DeOldify.
        
        Args:
            frame: Frame en niveaux de gris (H, W) ou (H, W, 1)
            
        Returns:
            Frame colorisée (H, W, 3)
        """
        h, w = frame.shape[:2]
        
        # Préparation de l'image
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
        
        # Convertir vers LAB
        lab_image = np.zeros((h, w, 3), dtype=np.float32)
        lab_image[:, :, 0] = gray_frame.astype(np.float32) / 255.0 * 100  # Canal L
        
        # Preprocessing pour le modèle
        pil_image = Image.fromarray(gray_frame).convert('L')
        input_tensor = self.transform(pil_image)
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
        else:
            # Fallback si transform ne retourne pas un tensor
            input_tensor = torch.from_numpy(np.array(pil_image)).float().unsqueeze(0).unsqueeze(0).to(self.device)
            input_tensor = (input_tensor / 255.0 - 0.5) / 0.5
        
        with torch.no_grad():
            # Prédiction des canaux AB
            ab_pred = self.colorizer(input_tensor)
            ab_pred = ab_pred.squeeze().cpu().numpy()
            
            # Redimensionner aux dimensions originales
            ab_pred = np.transpose(ab_pred, (1, 2, 0))
            ab_pred = cv2.resize(ab_pred, (w, h))
            
            # Dénormaliser
            ab_pred = ab_pred * 128  # Dénormaliser de [-1, 1] vers [-128, 128]
            
            # Combiner avec le canal L
            lab_image[:, :, 1:] = ab_pred
            
            # Convertir vers BGR
            lab_image = lab_image.astype(np.uint8)
            bgr_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
            
            return bgr_image
    
    def restore_and_colorize_video(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Combine restauration et colorisation selon Zhang et al.
        
        Args:
            frames: Liste de frames en niveaux de gris
            
        Returns:
            Liste de frames restaurées et colorisées
        """
        print("🎨 Restauration et colorisation avancées (DeOldify + Zhang et al.) en cours...")
        
        # Utiliser le processeur optimisé si disponible
        if self.optimized_processor:
            try:
                print(f"🚀 Utilisation du processeur optimisé")
                
                def process_frame(frame):
                    return self.colorize_frame(frame)
                
                if len(frames) > 30:
                    colorized_frames = self.optimized_processor.process_frames_batch(
                        frames, 
                        process_frame,
                        batch_size=8
                    )
                else:
                    colorized_frames = self.optimized_processor.process_frames_batch(
                        frames,
                        process_frame,
                        batch_size=4
                    )
                
                # Appliquer la cohérence temporelle
                return self._apply_temporal_consistency(colorized_frames)
                
            except Exception as e:
                print(f"⚠️ Erreur avec le processeur optimisé: {e}")
                print("📝 Basculement vers le mode standard")
        
        # Traitement standard
        colorized_frames = []
        
        for i, frame in enumerate(frames):
            colorized = self.colorize_frame(frame)
            
            # Lissage temporel
            if i > 0 and len(colorized_frames) > 0:
                colorized = cv2.addWeighted(
                    colorized_frames[-1], self.temporal_alpha,
                    colorized, 1 - self.temporal_alpha,
                    0
                )
            
            colorized_frames.append(colorized)
            
            if (i + 1) % 10 == 0:
                print(f"  Progression: {i + 1}/{len(frames)} frames")
        
        return colorized_frames
    
    def _apply_temporal_consistency(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Applique la cohérence temporelle avec le modèle Zhang et al.
        
        Args:
            frames: Liste de frames colorisées
            
        Returns:
            Liste de frames avec cohérence temporelle améliorée
        """
        if len(frames) < 3:
            return frames
        
        print("🔄 Application de la cohérence temporelle...")
        
        # Préparer les données pour le modèle temporel
        batch_size = 8  # Traiter par batch de 8 frames
        consistent_frames = []
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            
            if len(batch_frames) < 3:
                # Pas assez de frames pour le traitement temporel
                consistent_frames.extend(batch_frames)
                continue
            
            # Convertir en tensor
            frame_tensors = []
            for frame in batch_frames:
                frame_tensor = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
                frame_tensors.append(frame_tensor)
            
            # Stack en format (1, channels, frames, height, width)
            video_tensor = torch.stack(frame_tensors, dim=1).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Appliquer le modèle de restauration temporelle
                enhanced_video = self.video_restorer(video_tensor)
                enhanced_video = enhanced_video.squeeze(0)
                
                # Convertir back to numpy
                for t in range(enhanced_video.shape[1]):
                    enhanced_frame = enhanced_video[:, t, :, :].cpu().numpy()
                    enhanced_frame = np.transpose(enhanced_frame, (1, 2, 0))
                    enhanced_frame = (enhanced_frame * 255).astype(np.uint8)
                    consistent_frames.append(enhanced_frame)
        
        return consistent_frames
    
    def get_model_info(self) -> dict:
        """
        Retourne des informations sur les modèles chargés.
        """
        return {
            'deoldify_params': sum(p.numel() for p in self.colorizer.parameters()),
            'zhang_params': sum(p.numel() for p in self.video_restorer.parameters()),
            'device': self.device,
            'optimizations_enabled': self.optimized_processor is not None
        }
