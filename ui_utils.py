"""
Utilitaires pour l'interface utilisateur améliorée
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
import io
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional, Tuple
import tempfile
import os

def get_video_thumbnail(video_path: str, timestamp: float = 1.0) -> Image.Image:
    """
    Extrait une miniature de la vidéo à un timestamp donné
    
    Args:
        video_path: Chemin vers la vidéo
        timestamp: Temps en secondes pour extraire la frame
        
    Returns:
        Image PIL de la miniature
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps > 0:
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    else:
        # Image par défaut si échec
        return Image.new('RGB', (640, 480), color='gray')

def create_video_preview_grid(video_path: str, num_previews: int = 6) -> List[Image.Image]:
    """
    Crée une grille de prévisualisations de la vidéo
    
    Args:
        video_path: Chemin vers la vidéo
        num_previews: Nombre de prévisualisations à générer
        
    Returns:
        Liste d'images PIL
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    previews = []
    frame_indices = np.linspace(0, total_frames - 1, num_previews, dtype=int)
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Redimensionner pour la prévisualisation
            height, width = frame_rgb.shape[:2]
            if width > 300:
                new_width = 300
                new_height = int(height * (new_width / width))
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
            
            previews.append(Image.fromarray(frame_rgb))
        else:
            previews.append(Image.new('RGB', (300, 200), color='gray'))
    
    cap.release()
    return previews

def display_video_preview_grid(video_path: str, title: str = "Aperçu de la vidéo"):
    """
    Affiche une grille de prévisualisations de la vidéo
    
    Args:
        video_path: Chemin vers la vidéo
        title: Titre de la section
    """
    st.subheader(title)
    
    previews = create_video_preview_grid(video_path)
    
    # Afficher en grille 3x2
    cols = st.columns(3)
    for i, preview in enumerate(previews):
        with cols[i % 3]:
            st.image(preview, caption=f"Frame {i+1}", use_column_width=True)

def create_quality_radar_chart(metrics: Dict[str, float]) -> go.Figure:
    """
    Crée un graphique radar pour visualiser la qualité
    
    Args:
        metrics: Dictionnaire des métriques
        
    Returns:
        Figure Plotly
    """
    # Normaliser les métriques pour le radar chart
    normalized_metrics = {}
    
    # PSNR: normaliser sur une échelle 0-100 (max théorique ~50dB)
    if 'psnr' in metrics:
        normalized_metrics['PSNR'] = min(metrics['psnr'] * 2, 100)
    
    # SSIM: déjà sur une échelle 0-1, multiplier par 100
    if 'ssim' in metrics:
        normalized_metrics['SSIM'] = metrics['ssim'] * 100
    
    # Ajouter d'autres métriques fictives pour démo
    normalized_metrics.update({
        'Saturation': min(metrics.get('saturation', 50), 100),
        'Contraste': min(metrics.get('contrast', 60), 100),
        'Netteté': min(metrics.get('sharpness', 70), 100),
        'Cohérence': min(metrics.get('temporal_consistency', 80), 100)
    })
    
    categories = list(normalized_metrics.keys())
    values = list(normalized_metrics.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Qualité',
        line=dict(color='#667eea', width=2),
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            )
        ),
        showlegend=True,
        title="Analyse de Qualité",
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_processing_timeline(steps: List[str], current_step: int = 0) -> str:
    """
    Crée une timeline de traitement visuelle
    
    Args:
        steps: Liste des étapes de traitement
        current_step: Étape actuelle (0-indexé)
        
    Returns:
        HTML de la timeline
    """
    timeline_html = '<div class="processing-timeline">'
    
    for i, step in enumerate(steps):
        if i < current_step:
            status_class = "completed"
            icon = "✅"
        elif i == current_step:
            status_class = "current"
            icon = "⏳"
        else:
            status_class = "pending"
            icon = "⏸️"
        
        timeline_html += f'''
        <div class="timeline-step {status_class}">
            <div class="timeline-icon">{icon}</div>
            <div class="timeline-content">
                <h4>{step}</h4>
            </div>
        </div>
        '''
    
    timeline_html += '</div>'
    
    # Ajouter le CSS pour la timeline
    timeline_css = '''
    <style>
    .processing-timeline {
        margin: 20px 0;
    }
    .timeline-step {
        display: flex;
        align-items: center;
        margin: 10px 0;
        padding: 10px;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .timeline-step.completed {
        background: #d4edda;
        border-left: 4px solid #28a745;
    }
    .timeline-step.current {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        animation: pulse 1.5s infinite;
    }
    .timeline-step.pending {
        background: #f8f9fa;
        border-left: 4px solid #6c757d;
        opacity: 0.7;
    }
    .timeline-icon {
        margin-right: 10px;
        font-size: 18px;
    }
    .timeline-content h4 {
        margin: 0;
        font-size: 14px;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    </style>
    '''
    
    return timeline_css + timeline_html

def create_comparison_slider(before_image: np.ndarray, after_image: np.ndarray) -> str:
    """
    Crée un slider de comparaison avant/après
    
    Args:
        before_image: Image avant traitement
        after_image: Image après traitement
        
    Returns:
        HTML du slider de comparaison
    """
    # Convertir les images en base64 pour l'affichage
    before_pil = Image.fromarray(cv2.cvtColor(before_image, cv2.COLOR_BGR2RGB))
    after_pil = Image.fromarray(cv2.cvtColor(after_image, cv2.COLOR_BGR2RGB))
    
    # Redimensionner si nécessaire
    max_width = 600
    if before_pil.width > max_width:
        ratio = max_width / before_pil.width
        new_size = (max_width, int(before_pil.height * ratio))
        before_pil = before_pil.resize(new_size)
        after_pil = after_pil.resize(new_size)
    
    # Convertir en base64
    before_buffer = io.BytesIO()
    after_buffer = io.BytesIO()
    
    before_pil.save(before_buffer, format='PNG')
    after_pil.save(after_buffer, format='PNG')
    
    before_b64 = base64.b64encode(before_buffer.getvalue()).decode()
    after_b64 = base64.b64encode(after_buffer.getvalue()).decode()
    
    slider_html = f'''
    <div class="comparison-slider">
        <div class="comparison-container">
            <img src="data:image/png;base64,{before_b64}" class="comparison-before" alt="Avant">
            <div class="comparison-overlay">
                <img src="data:image/png;base64,{after_b64}" class="comparison-after" alt="Après">
            </div>
            <input type="range" min="0" max="100" value="50" class="comparison-range" oninput="updateComparison(this.value)">
        </div>
    </div>
    
    <script>
    function updateComparison(value) {{
        const overlay = document.querySelector('.comparison-overlay');
        overlay.style.width = value + '%';
    }}
    </script>
    
    <style>
    .comparison-slider {{
        position: relative;
        margin: 20px 0;
    }}
    .comparison-container {{
        position: relative;
        width: 100%;
        height: auto;
        overflow: hidden;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }}
    .comparison-before, .comparison-after {{
        width: 100%;
        height: auto;
        display: block;
    }}
    .comparison-overlay {{
        position: absolute;
        top: 0;
        left: 0;
        width: 50%;
        height: 100%;
        overflow: hidden;
    }}
    .comparison-range {{
        position: absolute;
        bottom: 10px;
        left: 50%;
        transform: translateX(-50%);
        width: 80%;
        height: 5px;
        background: #ddd;
        outline: none;
        border-radius: 5px;
    }}
    .comparison-range::-webkit-slider-thumb {{
        appearance: none;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: #667eea;
        cursor: pointer;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }}
    </style>
    '''
    
    return slider_html

def display_progress_with_eta(current: int, total: int, start_time: float, task_name: str = "Traitement"):
    """
    Affiche une barre de progression avec ETA
    
    Args:
        current: Étape actuelle
        total: Total d'étapes
        start_time: Temps de début (timestamp)
        task_name: Nom de la tâche
    """
    import time
    
    progress = current / total if total > 0 else 0
    elapsed = time.time() - start_time
    
    if current > 0:
        eta = (elapsed / current) * (total - current)
        eta_str = f"{int(eta//60):02d}:{int(eta%60):02d}"
    else:
        eta_str = "--:--"
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.progress(progress)
    
    with col2:
        st.write(f"**{current}/{total}**")
    
    with col3:
        st.write(f"**ETA: {eta_str}**")
    
    st.write(f"🔄 {task_name} en cours... ({progress*100:.1f}%)")

def create_feature_comparison_table() -> str:
    """
    Crée un tableau de comparaison des fonctionnalités
    
    Returns:
        HTML du tableau
    """
    return '''
    <div class="feature-table">
        <table>
            <thead>
                <tr>
                    <th>Fonctionnalité</th>
                    <th>IA Avancée</th>
                    <th>Classique</th>
                    <th>Hybrid</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Facilité d'utilisation</strong></td>
                    <td><span class="rating-5">⭐⭐⭐⭐⭐</span></td>
                    <td><span class="rating-3">⭐⭐⭐</span></td>
                    <td><span class="rating-4">⭐⭐⭐⭐</span></td>
                </tr>
                <tr>
                    <td><strong>Qualité des résultats</strong></td>
                    <td><span class="rating-5">⭐⭐⭐⭐⭐</span></td>
                    <td><span class="rating-3">⭐⭐⭐</span></td>
                    <td><span class="rating-5">⭐⭐⭐⭐⭐</span></td>
                </tr>
                <tr>
                    <td><strong>Vitesse de traitement</strong></td>
                    <td><span class="rating-3">⭐⭐⭐</span></td>
                    <td><span class="rating-4">⭐⭐⭐⭐</span></td>
                    <td><span class="rating-3">⭐⭐⭐</span></td>
                </tr>
                <tr>
                    <td><strong>Contrôle précis</strong></td>
                    <td><span class="rating-2">⭐⭐</span></td>
                    <td><span class="rating-5">⭐⭐⭐⭐⭐</span></td>
                    <td><span class="rating-4">⭐⭐⭐⭐</span></td>
                </tr>
                <tr>
                    <td><strong>Cohérence temporelle</strong></td>
                    <td><span class="rating-4">⭐⭐⭐⭐</span></td>
                    <td><span class="rating-3">⭐⭐⭐</span></td>
                    <td><span class="rating-5">⭐⭐⭐⭐⭐</span></td>
                </tr>
            </tbody>
        </table>
    </div>
    
    <style>
    .feature-table {
        margin: 20px 0;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .feature-table table {
        width: 100%;
        border-collapse: collapse;
        background: white;
    }
    .feature-table th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        text-align: left;
        font-weight: 600;
    }
    .feature-table td {
        padding: 12px 15px;
        border-bottom: 1px solid #eee;
    }
    .feature-table tr:last-child td {
        border-bottom: none;
    }
    .feature-table tr:hover {
        background: #f8f9fa;
    }
    .rating-5 { color: #28a745; }
    .rating-4 { color: #17a2b8; }
    .rating-3 { color: #ffc107; }
    .rating-2 { color: #fd7e14; }
    .rating-1 { color: #dc3545; }
    </style>
    '''

def create_stats_dashboard(processing_history: List[Dict]) -> Dict[str, Any]:
    """
    Crée un tableau de bord statistique
    
    Args:
        processing_history: Historique des traitements
        
    Returns:
        Dictionnaire avec les statistiques et graphiques
    """
    if not processing_history:
        return {}
    
    # Calculer les statistiques
    total_videos = len(processing_history)
    ai_count = sum(1 for entry in processing_history if 'IA' in entry.get('method', ''))
    classical_count = sum(1 for entry in processing_history if 'Classique' in entry.get('method', ''))
    
    # Moyennes des métriques
    psnr_values = [entry['metrics'].get('psnr', 0) for entry in processing_history if entry.get('metrics')]
    ssim_values = [entry['metrics'].get('ssim', 0) for entry in processing_history if entry.get('metrics')]
    
    avg_psnr = np.mean(psnr_values) if psnr_values else 0
    avg_ssim = np.mean(ssim_values) if ssim_values else 0
    
    # Graphique de distribution des méthodes
    methods_fig = px.pie(
        values=[ai_count, classical_count],
        names=['IA', 'Classique'],
        title="Répartition des Méthodes Utilisées",
        color_discrete_sequence=['#667eea', '#764ba2']
    )
    
    # Graphique d'évolution des métriques dans le temps
    if len(processing_history) > 1:
        dates = [entry['timestamp'][:10] for entry in processing_history[-10:]]
        psnr_recent = [entry['metrics'].get('psnr', 0) for entry in processing_history[-10:] if entry.get('metrics')]
        
        evolution_fig = px.line(
            x=dates[:len(psnr_recent)],
            y=psnr_recent,
            title="Évolution de la Qualité (PSNR)",
            labels={'x': 'Date', 'y': 'PSNR (dB)'}
        )
        evolution_fig.update_traces(line_color='#667eea', line_width=3)
    else:
        evolution_fig = None
    
    return {
        'total_videos': total_videos,
        'ai_count': ai_count,
        'classical_count': classical_count,
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        'methods_distribution': methods_fig,
        'quality_evolution': evolution_fig
    }
