@echo off
REM Script de démarrage amélioré pour Windows - Video AI Colorization Studio

setlocal enabledelayedexpansion

REM Configuration des couleurs (codes ANSI pour Windows 10+)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "PURPLE=[95m"
set "CYAN=[96m"
set "WHITE=[97m"
set "NC=[0m"

REM Banner ASCII
echo %PURPLE%
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║        🎨 VIDEO AI COLORIZATION STUDIO 🎨                   ║
echo ║                                                              ║
echo ║     Restauration et Colorisation de Vidéos par IA          ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo %NC%

echo.
echo %BLUE%[INFO]%NC% Démarrage de Video AI Colorization Studio...

REM Vérifier si nous sommes dans le bon répertoire
if not exist "app_enhanced.py" (
    if not exist "app.py" (
        echo %RED%[ERROR]%NC% Fichiers d'application non trouvés. Êtes-vous dans le bon répertoire ?
        pause
        exit /b 1
    )
)

REM Fonction pour vérifier Python
echo %BLUE%[INFO]%NC% Vérification de Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%[ERROR]%NC% Python n'est pas installé ou pas dans le PATH.
    echo Veuillez installer Python 3.8 ou plus récent depuis https://python.org
    pause
    exit /b 1
)

echo %GREEN%[SUCCESS]%NC% Python détecté ✓

REM Vérifier/créer l'environnement virtuel
echo %BLUE%[INFO]%NC% Configuration de l'environnement virtuel...

if exist "deoldify-env\Scripts\activate.bat" (
    echo %BLUE%[INFO]%NC% Activation de l'environnement virtuel existant...
    call deoldify-env\Scripts\activate.bat
    echo %GREEN%[SUCCESS]%NC% Environnement virtuel activé ✓
) else (
    echo %BLUE%[INFO]%NC% Création d'un nouvel environnement virtuel...
    python -m venv deoldify-env
    if errorlevel 1 (
        echo %RED%[ERROR]%NC% Échec de création de l'environnement virtuel
        pause
        exit /b 1
    )
    
    call deoldify-env\Scripts\activate.bat
    echo %GREEN%[SUCCESS]%NC% Environnement virtuel créé et activé ✓
)

REM Mettre à jour pip
echo %BLUE%[INFO]%NC% Mise à jour de pip...
python -m pip install --upgrade pip --quiet

REM Installer les dépendances
echo %BLUE%[INFO]%NC% Installation des dépendances...
if exist "requirements.txt" (
    pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo %YELLOW%[WARNING]%NC% Erreur lors de l'installation groupée. Tentative individuelle...
        
        REM Packages critiques
        echo %BLUE%[INFO]%NC% Installation de streamlit...
        pip install streamlit>=1.28.0 --quiet
        
        echo %BLUE%[INFO]%NC% Installation d'opencv-python...
        pip install opencv-python>=4.8.0 --quiet
        
        echo %BLUE%[INFO]%NC% Installation de numpy...
        pip install numpy>=1.21.0 --quiet
        
        echo %BLUE%[INFO]%NC% Installation de pillow...
        pip install pillow>=9.0.0 --quiet
        
        echo %BLUE%[INFO]%NC% Installation de plotly...
        pip install plotly>=5.15.0 --quiet
        
        echo %BLUE%[INFO]%NC% Installation de streamlit-option-menu...
        pip install streamlit-option-menu>=0.3.6 --quiet
    )
    echo %GREEN%[SUCCESS]%NC% Dépendances installées ✓
) else (
    echo %RED%[ERROR]%NC% Fichier requirements.txt non trouvé
    pause
    exit /b 1
)

REM Créer les répertoires nécessaires
echo %BLUE%[INFO]%NC% Création des répertoires nécessaires...
if not exist "static\output" mkdir static\output
if not exist "static\history" mkdir static\history
if not exist "static\templates" mkdir static\templates
if not exist "data\input" mkdir data\input
if not exist "data\output" mkdir data\output
if not exist "models" mkdir models
if not exist "logs" mkdir logs

echo %GREEN%[SUCCESS]%NC% Répertoires créés ✓

REM Vérifier le GPU (optionnel)
echo %BLUE%[INFO]%NC% Vérification du support GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo %YELLOW%[WARNING]%NC% Aucun GPU NVIDIA détecté. L'application fonctionnera sur CPU.
) else (
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2^>nul') do (
        echo %GREEN%[SUCCESS]%NC% GPU détecté: %%i ✓
        goto :gpu_found
    )
    :gpu_found
)

REM Démarrer l'application
echo.
echo %CYAN%==========================================
echo   🚀 APPLICATION EN COURS DE DÉMARRAGE
echo ==========================================%NC%
echo.

REM Choisir le fichier d'application
set "app_file=app_enhanced.py"
if not exist "%app_file%" (
    set "app_file=app.py"
    echo %YELLOW%[WARNING]%NC% Utilisation de l'application standard (app.py)
) else (
    echo %GREEN%[SUCCESS]%NC% Utilisation de l'application améliorée (app_enhanced.py)
)

REM Afficher les URLs d'accès
echo %GREEN%🌐 L'application sera disponible sur:
echo    👉 Local:   http://localhost:8501
echo.
echo 📱 Pour arrêter l'application, appuyez sur Ctrl+C%NC%
echo.

REM Lancer Streamlit avec les options optimisées
streamlit run %app_file% ^
    --server.port=8501 ^
    --server.address=0.0.0.0 ^
    --server.headless=true ^
    --server.enableCORS=false ^
    --server.enableXsrfProtection=false ^
    --theme.base=light ^
    --theme.primaryColor=#667eea ^
    --theme.backgroundColor=#ffffff ^
    --theme.secondaryBackgroundColor=#f0f2f6

REM Nettoyage à la sortie
echo.
echo %YELLOW%🛑 Arrêt de l'application...%NC%
echo %GREEN%[SUCCESS]%NC% Application arrêtée proprement ✓

pause
