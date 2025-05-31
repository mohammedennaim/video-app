import cv2
import numpy as np
import os
from tqdm import tqdm
from .utils import ensure_directory

def apply_denoising(video_path, method='median_filter'):
    """Apply denoising to the video using the specified method"""
    # Read video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Ensure output directory exists
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(current_dir, 'static', 'output')
    ensure_directory(output_dir)

    # Create output video writer
    output_path = os.path.join(output_dir, f'denoised_{method}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process frames
    with tqdm(total=total_frames, desc=f"Applying {method}") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Apply denoising based on method
            if method == 'median_filter':
                denoised_frame = cv2.medianBlur(frame, 5)
            elif method == 'bilateral_filter':
                denoised_frame = cv2.bilateralFilter(frame, 9, 75, 75)
            else:
                denoised_frame = frame

            # Write frame
            out.write(denoised_frame)
            pbar.update(1)

    # Cleanup
    cap.release()
    out.release()

    return output_path 