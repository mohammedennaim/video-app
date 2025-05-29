"""
Module d'entraînement pour les modèles de Deep Learning.
Permet d'entraîner ou fine-tuner les modèles GAN pour la colorisation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
from typing import Tuple, List, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from .gan_colorization import UNetGenerator

class VideoColorizationDataset(Dataset):
    """Dataset pour l'entraînement de colorisation vidéo."""
    
    def __init__(self, data_dir: str, transform=None, max_samples: Optional[int] = None):
        """
        Initialise le dataset.
        
        Args:
            data_dir: Répertoire contenant les vidéos d'entraînement
            transform: Transformations à appliquer
            max_samples: Nombre maximum d'échantillons (pour débug)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = self._load_samples(max_samples)
    
    def _load_samples(self, max_samples: Optional[int]) -> List[Tuple[str, str]]:
        """Charge les paires (frame_gray, frame_color)."""
        samples = []
        
        # Supposons une structure : data_dir/color/ et data_dir/gray/
        color_dir = self.data_dir / "color"
        gray_dir = self.data_dir / "gray"
        
        if not (color_dir.exists() and gray_dir.exists()):
            print("⚠️ Répertoires d'entraînement non trouvés, création d'un dataset de démonstration")
            return self._create_demo_samples()
        
        color_files = sorted(list(color_dir.glob("*.jpg")) + list(color_dir.glob("*.png")))
        
        for color_file in color_files:
            gray_file = gray_dir / color_file.name
            if gray_file.exists():
                samples.append((str(gray_file), str(color_file)))
                
                if max_samples and len(samples) >= max_samples:
                    break
        
        return samples
    
    def _create_demo_samples(self) -> List[Tuple[str, str]]:
        """Crée un dataset de démonstration."""
        # Pour la démonstration, créer quelques échantillons synthétiques
        return []
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retourne un échantillon (gray_image, color_image)."""
        gray_path, color_path = self.samples[idx]
        
        # Charger les images
        gray_img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        color_img = cv2.imread(color_path, cv2.IMREAD_COLOR)
        
        # Convertir en LAB
        color_lab = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB)
        
        # Séparer les canaux
        l_channel = color_lab[:, :, 0]  # Luminance
        ab_channels = color_lab[:, :, 1:]  # Chrominance
        
        # Redimensionner
        target_size = (256, 256)
        l_channel = cv2.resize(l_channel, target_size)
        ab_channels = cv2.resize(ab_channels, target_size)
        
        # Normaliser
        l_tensor = torch.from_numpy(l_channel).float().unsqueeze(0) / 255.0 * 2 - 1
        ab_tensor = torch.from_numpy(ab_channels).float().permute(2, 0, 1) / 128.0
        
        return l_tensor, ab_tensor

class GANTrainer:
    """Classe pour l'entraînement des modèles GAN."""
    
    def __init__(self, device: str = None):
        """
        Initialise l'entraîneur.
        
        Args:
            device: Device PyTorch ('cuda' ou 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🏋️ Entraînement sur: {self.device}")
        
        # Initialiser le générateur
        self.generator = UNetGenerator().to(self.device)
        
        # Initialiser le discriminateur (simple pour ce projet)
        self.discriminator = self._create_discriminator().to(self.device)
        
        # Optimiseurs
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Fonctions de perte
        self.criterion_GAN = nn.MSELoss()
        self.criterion_L1 = nn.L1Loss()
        
        # Historique d'entraînement
        self.train_history = {
            'g_loss': [],
            'd_loss': [],
            'l1_loss': []
        }
    
    def _create_discriminator(self) -> nn.Module:
        """Crée un discriminateur simple."""
        return nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def train(self, dataloader: DataLoader, num_epochs: int = 100, 
              save_interval: int = 10, checkpoint_dir: str = "models/checkpoints"):
        """
        Entraîne le modèle GAN.
        
        Args:
            dataloader: DataLoader pour l'entraînement
            num_epochs: Nombre d'époques
            save_interval: Intervalle de sauvegarde
            checkpoint_dir: Répertoire de sauvegarde
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"🚀 Début de l'entraînement pour {num_epochs} époques")
        
        for epoch in range(num_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_l1_loss = 0.0
            
            pbar = tqdm(dataloader, desc=f"Époque {epoch+1}/{num_epochs}")
            
            for batch_idx, (l_real, ab_real) in enumerate(pbar):
                batch_size = l_real.size(0)
                
                # Déplacer sur le device
                l_real = l_real.to(self.device)
                ab_real = ab_real.to(self.device)
                
                # Labels pour le GAN
                real_label = torch.ones(batch_size, 1, device=self.device)
                fake_label = torch.zeros(batch_size, 1, device=self.device)
                
                # ==================== Entraîner le Discriminateur ====================
                self.optimizer_D.zero_grad()
                
                # Images réelles
                real_images = torch.cat([l_real, ab_real], dim=1)
                output_real = self.discriminator(real_images)
                d_loss_real = self.criterion_GAN(output_real.view(-1, 1), real_label)
                
                # Images fausses
                ab_fake = self.generator(l_real)
                fake_images = torch.cat([l_real, ab_fake.detach()], dim=1)
                output_fake = self.discriminator(fake_images)
                d_loss_fake = self.criterion_GAN(output_fake.view(-1, 1), fake_label)
                
                # Perte totale du discriminateur
                d_loss = (d_loss_real + d_loss_fake) * 0.5
                d_loss.backward()
                self.optimizer_D.step()
                
                # ==================== Entraîner le Générateur ====================
                self.optimizer_G.zero_grad()
                
                # Perte GAN
                fake_images = torch.cat([l_real, ab_fake], dim=1)
                output = self.discriminator(fake_images)
                g_loss_gan = self.criterion_GAN(output.view(-1, 1), real_label)
                
                # Perte L1
                g_loss_l1 = self.criterion_L1(ab_fake, ab_real)
                
                # Perte totale du générateur
                g_loss = g_loss_gan + 100 * g_loss_l1  # Lambda = 100 pour L1
                g_loss.backward()
                self.optimizer_G.step()
                
                # Accumulation des pertes
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                epoch_l1_loss += g_loss_l1.item()
                
                # Mise à jour de la barre de progression
                pbar.set_postfix({
                    'G_Loss': f"{g_loss.item():.4f}",
                    'D_Loss': f"{d_loss.item():.4f}",
                    'L1_Loss': f"{g_loss_l1.item():.4f}"
                })
            
            # Moyennes des pertes pour l'époque
            avg_g_loss = epoch_g_loss / len(dataloader)
            avg_d_loss = epoch_d_loss / len(dataloader)
            avg_l1_loss = epoch_l1_loss / len(dataloader)
            
            # Sauvegarder l'historique
            self.train_history['g_loss'].append(avg_g_loss)
            self.train_history['d_loss'].append(avg_d_loss)
            self.train_history['l1_loss'].append(avg_l1_loss)
            
            print(f"Époque {epoch+1}: G_Loss={avg_g_loss:.4f}, D_Loss={avg_d_loss:.4f}, L1_Loss={avg_l1_loss:.4f}")
            
            # Sauvegarder le modèle périodiquement
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1, checkpoint_dir)
                self.plot_training_history(checkpoint_dir)
        
        print("✅ Entraînement terminé!")
    
    def save_checkpoint(self, epoch: int, checkpoint_dir: str):
        """Sauvegarde un checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'train_history': self.train_history
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Sauvegarder aussi le meilleur modèle
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save({
            'generator': self.generator.state_dict(),
            'epoch': epoch,
            'loss': self.train_history['g_loss'][-1]
        }, best_model_path)
        
        print(f"💾 Checkpoint sauvegardé: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Charge un checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.train_history = checkpoint['train_history']
        
        print(f"📥 Checkpoint chargé: {checkpoint_path}")
        return checkpoint['epoch']
    
    def plot_training_history(self, save_dir: str):
        """Génère les graphiques d'entraînement."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Perte du générateur
        axes[0].plot(self.train_history['g_loss'], label='Generator Loss')
        axes[0].set_title('Generator Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Perte du discriminateur
        axes[1].plot(self.train_history['d_loss'], label='Discriminator Loss', color='orange')
        axes[1].set_title('Discriminator Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        # Perte L1
        axes[2].plot(self.train_history['l1_loss'], label='L1 Loss', color='green')
        axes[2].set_title('L1 Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300)
        plt.close()

def create_training_data_from_videos(video_dir: str, output_dir: str, 
                                   frame_interval: int = 30):
    """
    Crée un dataset d'entraînement à partir de vidéos couleur.
    
    Args:
        video_dir: Répertoire contenant les vidéos couleur
        output_dir: Répertoire de sortie pour le dataset
        frame_interval: Intervalle entre les frames extraites
    """
    print("📁 Création du dataset d'entraînement...")
    
    color_dir = os.path.join(output_dir, "color")
    gray_dir = os.path.join(output_dir, "gray")
    
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(gray_dir, exist_ok=True)
    
    video_files = [f for f in os.listdir(video_dir) 
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    frame_count = 0
    
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                # Sauvegarder la frame couleur
                color_filename = f"frame_{frame_count:06d}.jpg"
                color_path = os.path.join(color_dir, color_filename)
                cv2.imwrite(color_path, frame)
                
                # Créer et sauvegarder la version en niveaux de gris
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_path = os.path.join(gray_dir, color_filename)
                cv2.imwrite(gray_path, gray_frame)
                
                frame_count += 1
            
            frame_idx += 1
        
        cap.release()
        print(f"  Vidéo traitée: {video_file} ({frame_count} frames extraites)")
    
    print(f"✅ Dataset créé avec {frame_count} paires d'images")

def main_training():
    """Fonction principale pour l'entraînement."""
    # Configuration
    data_dir = "data/training"
    checkpoint_dir = "models/checkpoints"
    batch_size = 8
    num_epochs = 50
    
    # Créer les répertoires
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Dataset et DataLoader
    dataset = VideoColorizationDataset(data_dir, max_samples=100)  # Limité pour la démo
    
    if len(dataset) == 0:
        print("⚠️ Aucun échantillon d'entraînement trouvé.")
        print("   Placez vos vidéos dans data/training/ et exécutez create_training_data_from_videos()")
        return
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Entraîneur
    trainer = GANTrainer()
    
    # Entraînement
    trainer.train(dataloader, num_epochs, save_interval=10, checkpoint_dir=checkpoint_dir)
    
    print("🎉 Entraînement terminé!")

if __name__ == "__main__":
    main_training()
