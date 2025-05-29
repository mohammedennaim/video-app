# Application de Restauration et Colorisation de Vidéos

## 🎯 Description

Cette application implémente des techniques avancées de restauration et colorisation de vidéos anciennes en utilisant à la fois des méthodes classiques de traitement d'images et des approches basées sur l'intelligence artificielle (GANs).

## ✨ Fonctionnalités

### Méthodes Classiques
- **Débruitage spatio-temporel** : Réduction du bruit en exploitant la cohérence temporelle
- **Interpolation pour colorisation** : Colorisation semi-automatique basée sur des algos classiques
- **Amélioration de contraste** : Amélioration automatique de la qualité visuelle

### Méthodes IA (GAN)
- **Architecture U-Net** : Générateur avec skip connections pour la colorisation
- **Modèle adversarial** : Discriminateur pour améliorer la qualité
- **Cohérence temporelle** : Lissage temporel des résultats

### Interface Utilisateur
- **Interface graphique** : Interface tkinter intuitive
- **Interface en ligne de commande** : Traitement automatisé en batch
- **Évaluation automatique** : Métriques PSNR, SSIM et LPIPS

## 📁 Structure du Projet

```
souad/
├── src/
│   ├── classical/
│   │   └── video_restoration.py    # Méthodes classiques
│   ├── ai/
│   │   └── gan_colorization.py     # Modèles GAN
│   ├── utils/
│   │   └── video_utils.py          # Utilitaires vidéo
│   ├── evaluation/
│   │   └── metrics.py              # Métriques d'évaluation
│   └── gui/
│       └── main_window.py          # Interface graphique
├── data/
│   ├── input/                      # Vidéos d'entrée
│   └── output/                     # Vidéos temporaires
├── models/                         # Modèles pré-entraînés
├── results/                        # Résultats et évaluations
├── main.py                         # Point d'entrée principal
├── demo.py                         # Script de démonstration
└── requirements.txt                # Dépendances
```

## 🚀 Installation

1. **Cloner le projet** :
```bash
cd souad
```

2. **Installer les dépendances** :
```bash
pip install -r requirements.txt
```

3. **Créer les dossiers nécessaires** :
```bash
mkdir -p data/input data/output models results
```

## 💻 Utilisation

### Interface en Ligne de Commande

**Traitement avec méthode classique** :
```bash
python main.py --input "data/input/video.mp4" --method classical --output "results/restored.mp4" --evaluate
```

**Traitement avec IA** :
```bash
python main.py --input "data/input/video.mp4" --method ai --output "results/colorized.mp4" --evaluate
```

**Comparaison des deux méthodes** :
```bash
python main.py --input "data/input/video.mp4" --method both --evaluate
```

### Interface Graphique

```bash
python main.py --gui
```

### Démonstration

```bash
python demo.py
```

## 📊 Métriques d'Évaluation

L'application calcule automatiquement :

- **PSNR (Peak Signal-to-Noise Ratio)** : Qualité de reconstruction
- **SSIM (Structural Similarity Index)** : Similarité structurelle
- **LPIPS (Learned Perceptual Image Patch Similarity)** : Qualité perceptuelle

### Résultats Typiques

| Méthode | PSNR (dB) | SSIM | Description |
|---------|-----------|------|-------------|
| Classique | 22.78 | 0.238 | Bon débruitage, colorisation basique |
| IA (GAN) | 20.32 | 0.264 | Colorisation plus naturelle, légèrement plus bruitée |

## 🎛️ Configuration

### Paramètres Classiques (src/classical/video_restoration.py)
```python
# Débruitage
sigma_spatial = 75      # Intensité lissage spatial
sigma_temporal = 10     # Intensité lissage temporel

# Colorisation
color_saturation = 1.2  # Saturation des couleurs
```

### Paramètres IA (src/ai/gan_colorization.py)
```python
# Architecture U-Net
input_size = (256, 256)    # Taille d'entrée
features = 64              # Nombre de features de base

# Post-processing
temporal_smoothing = True  # Lissage temporel
alpha = 0.7               # Coefficient de lissage
```

## 🔧 Développement

### Ajouter une Nouvelle Méthode

1. **Créer le module** dans `src/nouvelle_methode/`
2. **Implémenter l'interface** :
```python
class NouvelleMethode:
    def process_video(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        # Votre implémentation
        return processed_frames
```
3. **Intégrer dans main.py**

### Tests

```bash
# Lancer les tests automatiques
python -m pytest tests/

# Test de performance
python demo.py --benchmark
```

## 📈 Performances

### Configuration Testée
- **Processeur** : Intel i7 / AMD Ryzen 7
- **Mémoire** : 16 GB RAM
- **GPU** : NVIDIA GTX 1660 (optionnel)

### Temps de Traitement (72 frames, 480p)
- **Classique** : ~15 secondes
- **IA (CPU)** : ~45 secondes  
- **IA (GPU)** : ~12 secondes

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit les changements (`git commit -am 'Ajout nouvelle fonctionnalité'`)
4. Push vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Créer une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🙏 Remerciements

- **OpenCV** pour les outils de traitement d'images
- **PyTorch** pour les modèles de deep learning
- **scikit-image** pour les métriques d'évaluation
- **tkinter** pour l'interface graphique

## 📞 Support

Pour toute question ou problème :
- Ouvrir une issue sur GitHub
- Consulter la documentation dans `docs/`
- Contacter l'équipe de développement

---

**Version** : 1.0.0  
**Dernière mise à jour** : Décembre 2024
