#!/bin/bash

# Script de démarrage amélioré pour l'application de colorisation vidéo
echo "🎨 Démarrage de Video AI Colorization Studio..."

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Fonction d'affichage avec couleurs
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner ASCII
echo -e "${PURPLE}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        🎨 VIDEO AI COLORIZATION STUDIO 🎨                   ║
║                                                              ║
║     Restauration et Colorisation de Vidéos par IA          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Vérifier si nous sommes dans le bon répertoire
if [ ! -f "app_enhanced.py" ]; then
    print_error "Fichier app_enhanced.py non trouvé. Êtes-vous dans le bon répertoire ?"
    exit 1
fi

# Fonction pour vérifier les dépendances système
check_system_deps() {
    print_status "Vérification des dépendances système..."
    
    # Vérifier Python
    if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
        print_error "Python n'est pas installé. Veuillez installer Python 3.8 ou plus récent."
        exit 1
    fi
    
    # Vérifier la version de Python
    python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1-2)
    required_version="3.8"
    
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
        print_error "Python $required_version ou plus récent requis. Version détectée: $python_version"
        exit 1
    fi
    
    print_success "Python $python_version détecté ✓"
}

# Fonction pour créer ou activer l'environnement virtuel
setup_virtual_env() {
    ENV_NAME="deoldify-env"
    
    if [ -d "$ENV_NAME" ]; then
        print_status "Activation de l'environnement virtuel existant..."
        source "$ENV_NAME/bin/activate" 2>/dev/null || source "$ENV_NAME/Scripts/activate" 2>/dev/null
        
        if [ $? -eq 0 ]; then
            print_success "Environnement virtuel activé ✓"
        else
            print_warning "Échec d'activation, recréation de l'environnement..."
            rm -rf "$ENV_NAME"
            create_new_env
        fi
    else
        create_new_env
    fi
}

create_new_env() {
    print_status "Création d'un nouvel environnement virtuel..."
    python3 -m venv deoldify-env
    
    if [ $? -eq 0 ]; then
        print_success "Environnement virtuel créé ✓"
        source deoldify-env/bin/activate 2>/dev/null || source deoldify-env/Scripts/activate 2>/dev/null
        
        if [ $? -eq 0 ]; then
            print_success "Environnement virtuel activé ✓"
        else
            print_error "Échec d'activation de l'environnement virtuel"
            exit 1
        fi
    else
        print_error "Échec de création de l'environnement virtuel"
        exit 1
    fi
}

# Fonction pour installer les dépendances
install_dependencies() {
    print_status "Vérification et installation des dépendances..."
    
    # Mettre à jour pip
    pip install --upgrade pip --quiet
    
    # Installer les dépendances
    if [ -f "requirements.txt" ]; then
        print_status "Installation des dépendances depuis requirements.txt..."
        pip install -r requirements.txt --quiet
        
        if [ $? -eq 0 ]; then
            print_success "Dépendances installées ✓"
        else
            print_error "Échec d'installation des dépendances"
            print_warning "Tentative d'installation individuelle..."
            
            # Essayer d'installer les packages critiques un par un
            critical_packages=(
                "streamlit>=1.28.0"
                "opencv-python>=4.8.0"
                "numpy>=1.21.0"
                "pillow>=9.0.0"
                "plotly>=5.15.0"
                "streamlit-option-menu>=0.3.6"
            )
            
            for package in "${critical_packages[@]}"; do
                print_status "Installation de $package..."
                pip install "$package" --quiet
            done
        fi
    else
        print_error "Fichier requirements.txt non trouvé"
        exit 1
    fi
}

# Fonction pour créer les répertoires nécessaires
create_directories() {
    print_status "Création des répertoires nécessaires..."
    
    directories=(
        "static/output"
        "static/history"
        "static/templates"
        "data/input"
        "data/output"
        "models"
        "logs"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Créé: $dir"
        fi
    done
}

# Fonction pour vérifier la configuration GPU (optionnel)
check_gpu() {
    print_status "Vérification du support GPU..."
    
    # Vérifier NVIDIA-SMI
    if command -v nvidia-smi &> /dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ $? -eq 0 ] && [ ! -z "$gpu_info" ]; then
            print_success "GPU détecté: $gpu_info ✓"
            return 0
        fi
    fi
    
    print_warning "Aucun GPU NVIDIA détecté. L'application fonctionnera sur CPU."
    return 1
}

# Fonction pour démarrer l'application
start_app() {
    print_status "Démarrage de l'application Streamlit..."
    echo -e "${CYAN}"
    echo "=========================================="
    echo "  🚀 APPLICATION EN COURS DE DÉMARRAGE"
    echo "=========================================="
    echo -e "${NC}"
    
    # Choisir le fichier d'application
    app_file="app_enhanced.py"
    if [ ! -f "$app_file" ]; then
        app_file="app.py"
        print_warning "Utilisation de l'application standard (app.py)"
    else
        print_success "Utilisation de l'application améliorée (app_enhanced.py)"
    fi
    
    # Options Streamlit
    streamlit_args=(
        "--server.port=8501"
        "--server.address=0.0.0.0"
        "--server.headless=true"
        "--server.enableCORS=false"
        "--server.enableXsrfProtection=false"
        "--theme.base=light"
        "--theme.primaryColor=#667eea"
        "--theme.backgroundColor=#ffffff"
        "--theme.secondaryBackgroundColor=#f0f2f6"
    )
    
    # Lancer Streamlit
    echo -e "${GREEN}"
    echo "🌐 L'application sera disponible sur:"
    echo "   👉 Local:   http://localhost:8501"
    echo "   👉 Réseau:  http://$(hostname -I | awk '{print $1}'):8501"
    echo ""
    echo "📱 Pour arrêter l'application, appuyez sur Ctrl+C"
    echo -e "${NC}"
    
    streamlit run "$app_file" "${streamlit_args[@]}"
}

# Fonction de nettoyage en cas d'interruption
cleanup() {
    echo -e "\n${YELLOW}🛑 Arrêt de l'application...${NC}"
    print_status "Nettoyage en cours..."
    
    # Tuer les processus Streamlit restants
    pkill -f streamlit 2>/dev/null
    
    print_success "Application arrêtée proprement ✓"
    exit 0
}

# Capturer les signaux d'interruption
trap cleanup SIGINT SIGTERM

# Menu principal
show_menu() {
    echo -e "${BLUE}"
    echo "Choisissez une option:"
    echo "1) 🚀 Démarrage complet (recommandé)"
    echo "2) 🔧 Installation/mise à jour des dépendances uniquement"
    echo "3) 🏃 Démarrage rapide (sans vérifications)"
    echo "4) 🧹 Nettoyage et réinstallation complète"
    echo "5) ❌ Quitter"
    echo -e "${NC}"
    read -p "Votre choix [1-5]: " choice
}

# Fonction de nettoyage complet
full_cleanup() {
    print_warning "⚠️  Nettoyage complet - Cela supprimera l'environnement virtuel existant"
    read -p "Êtes-vous sûr ? (y/N): " confirm
    
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        print_status "Suppression de l'environnement virtuel..."
        rm -rf deoldify-env
        
        print_status "Nettoyage des fichiers temporaires..."
        find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
        find . -name "*.pyc" -delete 2>/dev/null
        
        print_success "Nettoyage terminé ✓"
        
        # Réinstaller
        check_system_deps
        setup_virtual_env
        install_dependencies
        create_directories
    else
        print_status "Nettoyage annulé"
    fi
}

# Exécution principale
main() {
    case "${1:-menu}" in
        "full"|"1")
            check_system_deps
            setup_virtual_env
            install_dependencies
            create_directories
            check_gpu
            start_app
            ;;
        "deps"|"2")
            check_system_deps
            setup_virtual_env
            install_dependencies
            print_success "Installation terminée ✓"
            ;;
        "quick"|"3")
            if [ -d "deoldify-env" ]; then
                source deoldify-env/bin/activate 2>/dev/null || source deoldify-env/Scripts/activate 2>/dev/null
                start_app
            else
                print_error "Environnement virtuel non trouvé. Utilisez l'option 1 pour une installation complète."
                exit 1
            fi
            ;;
        "clean"|"4")
            full_cleanup
            ;;
        "menu")
            while true; do
                show_menu
                case $choice in
                    1) main "full"; break;;
                    2) main "deps"; break;;
                    3) main "quick"; break;;
                    4) main "clean"; break;;
                    5) print_success "Au revoir ! 👋"; exit 0;;
                    *) print_error "Option invalide. Veuillez choisir 1-5.";;
                esac
            done
            ;;
        *)
            echo "Usage: $0 [full|deps|quick|clean|menu]"
            echo "  full  - Installation complète et démarrage"
            echo "  deps  - Installation des dépendances uniquement"
            echo "  quick - Démarrage rapide"
            echo "  clean - Nettoyage et réinstallation"
            echo "  menu  - Afficher le menu interactif"
            exit 1
            ;;
    esac
}

# Lancer le script
main "$@"
