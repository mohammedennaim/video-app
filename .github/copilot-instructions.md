<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Instructions Copilot - Restauration et Colorisation de Vidéos

## Contexte du Projet
Ce projet implémente des techniques de restauration et colorisation de vidéos anciennes avec :
- Approches classiques (filtrage spatio-temporel, colorisation semi-automatique)
- Approches IA (GAN, Deep Learning)
- Interface graphique de comparaison
- Métriques d'évaluation (PSNR/SSIM)

## Guidelines de Code

### Structure
- Utilisez des classes orientées objet pour les processeurs vidéo
- Séparez clairement les méthodes classiques et IA
- Implémentez des interfaces communes pour la comparaison

### Traitement Vidéo
- Utilisez OpenCV pour les opérations de base
- Préservez la cohérence temporelle dans les traitements
- Optimisez pour les vidéos de grande taille

### IA et Deep Learning
- Intégrez TensorFlow/PyTorch pour les modèles
- Utilisez des modèles pré-entraînés quand possible
- Implémentez le fine-tuning si nécessaire

### Interface Utilisateur
- Utilisez PyQt5 pour l'interface graphique
- Affichez la progression des traitements longs
- Permettez la comparaison côte-à-côte

### Évaluation
- Calculez PSNR et SSIM automatiquement
- Générez des rapports comparatifs
- Sauvegardez les métriques pour analyse

### Bonnes Pratiques
- Documentez tous les algorithmes implémentés
- Utilisez des type hints Python
- Gérez les erreurs gracieusement
- Loggez les opérations importantes
