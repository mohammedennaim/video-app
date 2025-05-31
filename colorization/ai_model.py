"""
AI Colorization module using DeOldify
Handles video colorization with AI models and fallback methods
"""
import sys
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Import utilities
try:
    from .utils import ensure_directory
except ImportError:
    from utils import ensure_directory

# Configuration and setup
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
deoldify_path = os.path.join(current_dir, 'DeOldify')

# Try to setup DeOldify
DEOLDIFY_AVAILABLE = False

if os.path.exists(deoldify_path):
    try:
        if deoldify_path not in sys.path:
            sys.path.insert(0, deoldify_path)
        from deoldify.visualize import get_image_colorizer
        DEOLDIFY_AVAILABLE = True
        print("✅ DeOldify importé avec succès")
    except Exception as e:
        print(f"❌ Erreur d'import DeOldify: {e}")
        DEOLDIFY_AVAILABLE = False
else:
    print("❌ DeOldify non trouvé dans le répertoire du projet")
    DEOLDIFY_AVAILABLE = False


class AIColorizer:
    """AI-based video colorization using DeOldify with fallback methods"""
    
    def __init__(self):
        """Initialize the colorizer"""
        self.colorizer = None
        
        if DEOLDIFY_AVAILABLE:
            try:
                self.colorizer = get_image_colorizer(artistic=True)
                print("✅ Coloriseur AI initialisé avec succès")
            except Exception as e:
                print(f"❌ Erreur lors de l'initialisation du coloriseur: {e}")
                self.colorizer = None
        else:
            print("⚠️ DeOldify n'est pas disponible. Utilisation d'une méthode de colorisation de base.")
            # Try to show streamlit warning if available
            try:
                import streamlit as st
                st.warning("⚠️ DeOldify n'est pas disponible. Utilisation d'une méthode de colorisation de base.")
            except ImportError:
                pass  # Streamlit not available, continue without UI feedback

    def colorize(self, video_path):
        """
        Colorize a video using AI or fallback method
        
        Args:
            video_path (str): Path to input video file
            
        Returns:
            str: Path to colorized video output
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Ensure output directory exists
        output_dir = os.path.join(current_dir, 'static', 'output')
        ensure_directory(output_dir)
        
        output_path = os.path.join(output_dir, 'deoldify_colorized.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            if self.colorizer is None:
                # Use fallback colorization
                self._colorize_fallback(cap, out, total_frames)
            else:
                # Use DeOldify colorization
                self._colorize_deoldify(cap, out, total_frames, width, height)
        finally:
            cap.release()
            out.release()
        
        return output_path
    
    def _colorize_fallback(self, cap, out, total_frames):
        """Fallback colorization method when DeOldify is not available"""
        print("💡 Utilisation d'une méthode de colorisation de base (DeOldify non disponible)")
        try:
            import streamlit as st
            st.info("💡 Utilisation d'une méthode de colorisation de base (DeOldify non disponible)")
        except ImportError:
            pass
            
        with tqdm(total=total_frames, desc="Basic Colorizing video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # Apply basic colorization (sepia-like effect)
                colorized = self._basic_colorize(frame)
                out.write(colorized)
                pbar.update(1)
    
    def _colorize_deoldify(self, cap, out, total_frames, width, height):
        """DeOldify-based colorization"""
        with tqdm(total=total_frames, desc="AI Colorizing video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                colorized = self.colorizer.get_transformed_image(pil_img)
                colorized = colorized.resize((width, height))
                colorized = cv2.cvtColor(np.array(colorized), cv2.COLOR_RGB2BGR)
                out.write(colorized)
                pbar.update(1)
    
    def _basic_colorize(self, frame):
        """Basic colorization method using sepia effect"""
        # Convert to grayscale first
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Apply sepia effect
        sepia_kernel = np.array([[0.272, 0.534, 0.131],
                                [0.349, 0.686, 0.168],
                                [0.393, 0.769, 0.189]])
        
        # Convert grayscale to 3-channel
        gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Apply sepia transformation
        sepia = cv2.transform(gray_3channel, sepia_kernel)
        
        # Ensure values are in valid range
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        
        return sepia
