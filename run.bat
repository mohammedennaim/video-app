@echo off
REM Script de démarrage pour l'application de colorisation vidéo
echo Démarrage de l'application de restauration et colorisation vidéo...

REM Vérifier si l'environnement virtuel existe
if exist "deoldify-env\Scripts\activate.bat" (
    REM Activer l'environnement virtuel
    call deoldify-env\Scripts\activate.bat
    echo Environnement virtuel activé
) else (
    echo Environnement virtuel non trouvé. Veuillez créer un environnement virtuel avec:
    echo python -m venv deoldify-env
    echo deoldify-env\Scripts\activate
    echo pip install -r requirements.txt
    exit /b 1
)

REM Créer les répertoires nécessaires
if not exist "static\output" mkdir static\output

REM Lancer l'application
streamlit run app.py
