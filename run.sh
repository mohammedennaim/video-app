#!/bin/bash

# Script de démarrage pour l'application de colorisation vidéo
echo "Démarrage de l'application de restauration et colorisation vidéo..."

# Vérifier si l'environnement virtuel existe
if [ -d "deoldify-env/Scripts" ]; then
    # Activer l'environnement virtuel
    source deoldify-env/Scripts/activate
    echo "Environnement virtuel activé"
else
    echo "Environnement virtuel non trouvé. Veuillez créer un environnement virtuel avec:"
    echo "python -m venv deoldify-env"
    echo "deoldify-env\\Scripts\\activate"
    echo "pip install -r requirements.txt"
    exit 1
fi

# Créer les répertoires nécessaires
mkdir -p static/output

# Lancer l'application
streamlit run app.py
