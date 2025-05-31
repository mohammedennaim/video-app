import cv2
import numpy as np
import os
from tqdm import tqdm
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
import torch
import torchvision
from PIL import Image
from .utils import ensure_directory

COCO_CLASSES = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
    'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table',
    'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush', 'hair brush'
]

class ClassicalColorizer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seg_model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True).to(self.device).eval()

    def segment(self, frame):
        # frame: BGR numpy array
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        preprocess = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),        ])
        input_tensor = preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.seg_model(input_tensor)['out'][0]
        seg = output.argmax(0).cpu().numpy()
        return seg
    
    def colorize(self, video_path, class_to_color):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Ensure output directory exists
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(current_dir, 'static', 'output')
        ensure_directory(output_dir)
        
        output_path = os.path.join(output_dir, 'object_aware_classical_colorized.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        with tqdm(total=total_frames, desc="Object-aware Colorizing video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                seg = self.segment(frame)
                colorized = np.zeros_like(frame)
                for class_idx, color in class_to_color.items():
                    mask = (seg == class_idx)
                    colorized[mask] = color
                # Blend with original for realism
                colorized = cv2.addWeighted(frame, 0.3, colorized, 0.7, 0)
                out.write(colorized)
                pbar.update(1)
        cap.release()
        out.release()
        return output_path

    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _apply_color_to_frame(self, frame, color):
        """Apply color to a single frame"""
        # Convert hex color to RGB
        rgb_color = self._hex_to_rgb(color)
        
        # Convert frame to RGB if it's grayscale
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        
        # Create color overlay
        color_overlay = np.full_like(frame, rgb_color)
        
        # Blend original frame with color overlay
        alpha = 0.5  # Adjust this value to control color intensity
        colored_frame = cv2.addWeighted(frame, 1-alpha, color_overlay, alpha, 0)
        
        return colored_frame

    @staticmethod
    def extract_mask_and_scribble(scribble_img, orig_img):
        # Mask: where scribble_img differs from orig_img
        mask = np.any(scribble_img != orig_img, axis=2).astype(np.uint8)
        return mask, scribble_img

    def _levin_colorization(self, gray, scribbles, mask):
        h, w = gray.shape
        N = h * w
        indsM = np.arange(N).reshape(h, w)
        known = mask > 0
        win_rad = 1
        win_size = (2 * win_rad + 1) ** 2
        row_inds, col_inds, vals = [], [], []
        for y in range(win_rad, h - win_rad):
            for x in range(win_rad, w - win_rad):
                win_inds = indsM[y - win_rad:y + win_rad + 1, x - win_rad:x + win_rad + 1].flatten()
                winI = gray[y - win_rad:y + win_rad + 1, x - win_rad:x + win_rad + 1].flatten()
                win_mu = np.mean(winI)
                win_var = np.var(winI) + 1e-6
                winI = winI - win_mu
                G = np.outer(winI, winI) / win_var
                G = (G + np.eye(win_size)) / win_size
                for ii in range(win_size):
                    for jj in range(win_size):
                        row_inds.append(win_inds[ii])
                        col_inds.append(win_inds[jj])
                        vals.append(-G[ii, jj] if ii != jj else 1 - G[ii, jj])
        L = csr_matrix((vals, (row_inds, col_inds)), shape=(N, N))
        result = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(3):
            b = np.zeros(N)
            b[known.flatten()] = scribbles[:, :, c].flatten()[known.flatten()]
            D = diags(known.flatten().astype(float))
            x = spsolve(L + D, b)
            result[:, :, c] = np.clip(x.reshape(h, w), 0, 255)
        return result 