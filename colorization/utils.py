import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os

def calculate_metrics(original_path, processed_path):
    """Calculate PSNR and SSIM metrics between original and processed videos"""
    # Read videos
    cap_orig = cv2.VideoCapture(original_path)
    cap_proc = cv2.VideoCapture(processed_path)
    
    total_psnr = 0
    total_ssim = 0
    frame_count = 0
    
    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_proc, frame_proc = cap_proc.read()
        
        if not ret_orig or not ret_proc:
            break
            
        # Convert frames to grayscale for metrics calculation
        if len(frame_orig.shape) == 3:
            frame_orig_gray = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
        else:
            frame_orig_gray = frame_orig
            
        if len(frame_proc.shape) == 3:
            frame_proc_gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
        else:
            frame_proc_gray = frame_proc
        
        # Calculate metrics
        frame_psnr = psnr(frame_orig_gray, frame_proc_gray)
        frame_ssim = ssim(frame_orig_gray, frame_proc_gray)
        
        total_psnr += frame_psnr
        total_ssim += frame_ssim
        frame_count += 1
    
    # Cleanup
    cap_orig.release()
    cap_proc.release()
    
    # Calculate averages
    avg_psnr = total_psnr / frame_count if frame_count > 0 else 0
    avg_ssim = total_ssim / frame_count if frame_count > 0 else 0
    
    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim
    }

def save_video(frames, output_path, fps=30):
    """Save a list of frames as a video"""
    if not frames:
        return None
        
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    return output_path

def ensure_directory(directory):
    """Ensure that a directory exists, create if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)