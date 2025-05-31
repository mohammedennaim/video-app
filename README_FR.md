# Application de Restauration et Colorisation Vidéo

Une application Python pour restaurer et coloriser des vidéos en noir et blanc en utilisant des méthodes basées sur l'IA et des méthodes classiques. L'application dispose d'une interface graphique moderne basée sur Streamlit et prend en charge diverses techniques de traitement vidéo.

## Fonctionnalités

- **Colorisation basée sur l'IA** : Utilise DeOldify (basé sur GAN) pour la colorisation automatique de vidéos
- **Colorisation Classique** : Sélection manuelle des couleurs avec traitement image par image
- **Restauration Vidéo** : Implémente diverses méthodes de débruitage :
  - Filtre Médian
  - Filtre Bilatéral
- **Métriques de Qualité** : Calcule le PSNR et le SSIM pour l'évaluation de la qualité
- **Interface Moderne** : Construite avec Streamlit pour une expérience utilisateur intuitive
- **Comparaison Côte à Côte** : Visualisation simultanée des vidéos originales et traitées
- **Support de Téléchargement** : Sauvegarde des vidéos traitées localement

## Prérequis

- Python 3.8+
- GPU compatible CUDA (recommandé pour la colorisation basée sur l'IA)
- Voir `requirements.txt` pour les dépendances des packages Python

## Installation

1. Clonez le dépôt :
```bash
git clone https://github.com/yourusername/video-colorization-app.git
cd video-colorization-app
```

2. Créez un environnement virtuel (recommandé) :
```bash
python -m venv deoldify-env
source deoldify-env/bin/activate  # Sous Windows : deoldify-env\Scripts\activate
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

4. Installez DeOldify (ou placez-le dans un répertoire parent) :
```bash
cd ..
git clone https://github.com/jantic/DeOldify.git
cd DeOldify
pip install -r requirements.txt
cd ../video-app
```

5. Vérifiez l'installation :
```bash
python check_install.py
```

## Utilisation

1. Démarrez l'application :
```bash
# Sous Linux/macOS
./run.sh

# Sous Windows
run.bat
```

2. Ouvrez votre navigateur web et accédez à l'URL indiquée dans le terminal (généralement http://localhost:8501)

3. Téléchargez une vidéo en noir et blanc en utilisant l'outil de téléchargement de fichiers

4. Choisissez votre méthode de colorisation préférée :
   - Basée sur l'IA (DeOldify)
   - Classique (Sélection manuelle des couleurs)

5. Appliquez éventuellement des filtres de débruitage

6. Visualisez les résultats et téléchargez la vidéo traitée

## Structure du Projet

```
video-app/
├── app.py                      # Logique et routage de l'application Streamlit
├── colorization/
│   ├── ai_model.py             # Intégration du modèle DeOldify
│   ├── classical_methods.py    # Méthodes de colorisation manuelle
│   ├── filters.py              # Filtres de débruitage
│   ├── utils.py                # Fonctions utilitaires
├── static/
│   ├── samples/                # Vidéos d'exemple
│   └── output/                 # Vidéos traitées
├── requirements.txt
├── run.sh                      # Script de démarrage pour Linux/macOS
├── run.bat                     # Script de démarrage pour Windows
└── README.md
```

## Dépannage

Si vous rencontrez des problèmes d'installation ou d'exécution :

1. Vérifiez que toutes les dépendances sont installées :
```bash
python check_install.py
```

2. Assurez-vous que DeOldify est correctement installé et accessible

3. Si le GPU n'est pas détecté, la colorisation basée sur l'IA fonctionnera toujours mais sera beaucoup plus lente

## Licence

Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.

## Remerciements

- DeOldify par Jason Antic pour le modèle de colorisation basé sur l'IA
- OpenCV pour les capacités de traitement vidéo
- Streamlit pour le framework d'interface web
