#!/usr/bin/env python
"""
Script de test pour vérifier l'installation de l'application de colorisation vidéo.
"""
import sys
import os
import importlib
import subprocess
import platform

def check_import(module_name, package_name=None):
    """Vérifie si un module peut être importé."""
    if package_name is None:
        package_name = module_name
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} est installé correctement")
        return True
    except ImportError as e:
        print(f"❌ {module_name} n'est pas installé correctement: {e}")
        print(f"   Essayez: pip install {package_name}")
        return False

def check_directory(directory):
    """Vérifie si un répertoire existe et le crée si nécessaire."""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"✅ Répertoire créé: {directory}")
        except Exception as e:
            print(f"❌ Impossible de créer le répertoire {directory}: {e}")
            return False
    else:
        print(f"✅ Répertoire trouvé: {directory}")
    return True

def check_deoldify():
    """Vérifie si DeOldify est correctement installé."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    deoldify_path = os.path.join(os.path.dirname(current_dir), 'DeOldify')
    
    if os.path.exists(deoldify_path):
        print(f"✅ DeOldify trouvé à: {deoldify_path}")
        # Vérifier si le modèle est présent
        model_path = os.path.join(deoldify_path, 'models', 'ColorizeArtistic_gen.pth')
        if os.path.exists(model_path):
            print("✅ Modèle DeOldify trouvé")
        else:
            print("❌ Modèle DeOldify non trouvé, téléchargez-le depuis: https://www.dropbox.com/s/usf7uifrctqw9rl/ColorizeArtistic_gen.pth")
    elif os.path.exists(os.path.join(current_dir, 'DeOldify')):
        print(f"✅ DeOldify trouvé dans le répertoire du projet")
    else:
        print("❌ DeOldify non trouvé. Clonez-le depuis: https://github.com/jantic/DeOldify")
        print("   git clone https://github.com/jantic/DeOldify")
        return False
    return True

def main():
    """Fonction principale pour vérifier l'installation."""
    print("\n🔍 Vérification de l'installation de l'application de colorisation vidéo...\n")
    
    # Vérifier les modules Python
    deps = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("fastai", "fastai"),
        ("cv2", "opencv-python"),
        ("moviepy", "moviepy"),
        ("streamlit", "streamlit"),
        ("skimage", "scikit-image"),
        ("numpy", "numpy"),
        ("PIL", "pillow"),
        ("tqdm", "tqdm"),
        ("scipy", "scipy"),
        ("streamlit_drawable_canvas", "streamlit-drawable-canvas")
    ]
    
    all_deps_ok = True
    for module, package in deps:
        if not check_import(module, package):
            all_deps_ok = False
    
    # Vérifier les répertoires
    print("\n📁 Vérification des répertoires...")
    dirs_ok = True
    for directory in ["static", "static/output", "static/samples"]:
        if not check_directory(directory):
            dirs_ok = False
    
    # Vérifier DeOldify
    print("\n🎨 Vérification de DeOldify...")
    deoldify_ok = check_deoldify()
    
    # Résumé
    print("\n📋 Résumé de l'installation:")
    if all_deps_ok and dirs_ok and deoldify_ok:
        print("\n✅ Tout est correctement installé ! Vous pouvez lancer l'application avec:")
        if platform.system() == "Windows":
            print("   run.bat")
        else:
            print("   ./run.sh")
    else:
        print("\n❌ Certains problèmes ont été détectés. Veuillez les corriger avant de lancer l'application.")
    
    return 0 if all_deps_ok and dirs_ok and deoldify_ok else 1

if __name__ == "__main__":
    sys.exit(main())
