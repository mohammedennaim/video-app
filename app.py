import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from streamlit_drawable_canvas import st_canvas
from colorization.ai_model import AIColorizer
from colorization.classical_methods import ClassicalColorizer, COCO_CLASSES
from colorization.filters import apply_denoising
from colorization.utils import calculate_metrics, save_video, ensure_directory
import time
import sys

# Configuration des chemins
current_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure output directories exist
ensure_directory(os.path.join(current_dir, 'static', 'output'))
ensure_directory(os.path.join(current_dir, 'static', 'templates'))
ensure_directory(os.path.join(current_dir, 'static', 'history'))

def get_scribble_and_mask(canvas_data, orig_img):
    if canvas_data is None:
        return np.array(orig_img), np.zeros(orig_img.shape[:2], dtype=np.uint8)
    # The canvas returns RGBA, convert to RGB and mask
    scribble_img = Image.fromarray((canvas_data[:, :, :3]).astype(np.uint8))
    scribble_np = np.array(scribble_img)
    mask = np.any(scribble_np != np.array(orig_img), axis=2).astype(np.uint8)
    return scribble_np, mask

# Set page config
st.set_page_config(
    page_title="Video Restoration & Colorization",
    page_icon="🎨",
    layout="wide"
)

# Initialize session state
if 'processed_video' not in st.session_state:
    st.session_state.processed_video = None
if 'original_video' not in st.session_state:
    st.session_state.original_video = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

def main():
    st.title("🎨 Video Restoration & Colorization")
    st.write("Restore and colorize your black and white videos using AI or classical methods")

    # File uploader
    uploaded_file = st.file_uploader("Upload a black-and-white video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        # Display original video
        st.video(video_path)
        st.session_state.original_video = video_path

        # Method selection
        method = st.radio(
            "Choose colorization method:",
            ["AI-based (DeOldify)", "Classical (Object-aware)"]
        )

        if method == "AI-based (DeOldify)":
            if st.button("Start AI Colorization"):
                with st.spinner("Processing video with AI..."):
                    colorizer = AIColorizer()
                    output_path = colorizer.colorize(video_path)
                    st.session_state.processed_video = output_path
                    st.video(output_path)
                    
                    # Calculate metrics
                    metrics = calculate_metrics(video_path, output_path)
                    st.session_state.metrics = metrics
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("PSNR", f"{metrics['psnr']:.2f}")
                    with col2:
                        st.metric("SSIM", f"{metrics['ssim']:.2f}")

        else:  # Classical method
            st.write("Object-aware colorization:")
            # Extract first frame for class preview
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                st.error("Could not read video.")
                return
            colorizer = ClassicalColorizer()
            seg = colorizer.segment(frame)
            unique_classes = np.unique(seg)
            st.write("Detected object classes in the first frame:")
            class_to_color = {}
            for class_idx in unique_classes:
                class_name = COCO_CLASSES[class_idx] if class_idx < len(COCO_CLASSES) else f"class_{class_idx}"
                default_color = '#%02x%02x%02x' % tuple(np.random.randint(0, 255, 3))
                color = st.color_picker(f"Pick color for {class_name}", default_color)
                rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                class_to_color[class_idx] = rgb
            if st.button("Apply Object-aware Colorization"):
                with st.spinner("Processing video with object-aware colorization..."):
                    output_path = colorizer.colorize(video_path, class_to_color)
                    st.session_state.processed_video = output_path
                    st.video(output_path)
                    
                    # Calculate metrics
                    metrics = calculate_metrics(video_path, output_path)
                    st.session_state.metrics = metrics
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("PSNR", f"{metrics['psnr']:.2f}")
                    with col2:
                        st.metric("SSIM", f"{metrics['ssim']:.2f}")

        # Denoising options
        st.subheader("Video Restoration Options")
        denoising_method = st.selectbox(
            "Select denoising method:",
            ["None", "Median Filter", "Bilateral Filter"]
        )
        
        if denoising_method != "None" and st.session_state.processed_video:
            if st.button("Apply Denoising"):
                with st.spinner("Applying denoising..."):
                    output_path = apply_denoising(
                        st.session_state.processed_video,
                        method=denoising_method.lower().replace(" ", "_")
                    )
                    st.session_state.processed_video = output_path
                    st.video(output_path)

        # Download button
        if st.session_state.processed_video:
            with open(st.session_state.processed_video, 'rb') as f:
                st.download_button(
                    label="Download processed video",
                    data=f,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )

if __name__ == "__main__":
    main()