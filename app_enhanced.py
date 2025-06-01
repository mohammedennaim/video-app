"""
Application Web Améliorée pour la Restauration et Colorisation de Vidéos
Interface moderne avec fonctionnalités avancées
"""

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
from streamlit_option_menu import option_menu
import base64
from typing import Dict, List, Any, Optional

# Imports des modules de colorisation
try:
    from colorization.ai_model import AIColorizer
    from colorization.classical_methods import ClassicalColorizer, COCO_CLASSES
    from colorization.filters import apply_denoising
    from colorization.utils import calculate_metrics, save_video, ensure_directory
except ImportError as e:
    st.error(f"Erreur d'importation des modules: {e}")

# Configuration
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'static', 'output')
HISTORY_DIR = os.path.join(CURRENT_DIR, 'static', 'history')

# Assurer l'existence des répertoires
for directory in [OUTPUT_DIR, HISTORY_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configuration de la page
st.set_page_config(
    page_title="Video AI Colorization Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour une interface moderne
def load_custom_css():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .processing-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    
    .success-card {
        background: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .video-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    /* Animation pour les boutons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

def save_processing_history(video_name: str, method: str, metrics: Dict[str, float], output_path: str):
    """Sauvegarde l'historique des traitements"""
    history_file = os.path.join(HISTORY_DIR, 'processing_history.json')
    
    history_entry = {
        'timestamp': datetime.now().isoformat(),
        'video_name': video_name,
        'method': method,
        'metrics': metrics,
        'output_path': output_path
    }
    
    # Charger l'historique existant
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except:
            history = []
    
    # Ajouter la nouvelle entrée
    history.append(history_entry)
    
    # Limiter à 50 entrées max
    history = history[-50:]
    
    # Sauvegarder
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def load_processing_history() -> List[Dict]:
    """Charge l'historique des traitements"""
    history_file = os.path.join(HISTORY_DIR, 'processing_history.json')
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def display_video_info(video_path: str):
    """Affiche les informations de la vidéo"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Résolution", f"{width}×{height}")
    with col2:
        st.metric("FPS", f"{fps:.1f}")
    with col3:
        st.metric("Durée", f"{duration:.1f}s")
    with col4:
        st.metric("Frames", f"{frame_count}")

def create_comparison_chart(metrics_history: List[Dict]):
    """Crée un graphique de comparaison des métriques"""
    if not metrics_history:
        return None
    
    # Préparer les données
    timestamps = []
    psnr_values = []
    ssim_values = []
    methods = []
    
    for entry in metrics_history[-10:]:  # Dernières 10 entrées
        if 'metrics' in entry and entry['metrics']:
            timestamps.append(entry['timestamp'][:19])  # Format datetime
            psnr_values.append(entry['metrics'].get('psnr', 0))
            ssim_values.append(entry['metrics'].get('ssim', 0))
            methods.append(entry['method'])
    
    if not timestamps:
        return None
    
    # Créer le graphique
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=psnr_values,
        mode='lines+markers',
        name='PSNR',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8)
    ))
    
    # Axe secondaire pour SSIM
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=[v * 100 for v in ssim_values],  # Multiplier par 100 pour la visualisation
        mode='lines+markers',
        name='SSIM ×100',
        line=dict(color='#764ba2', width=3),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Évolution des Métriques de Qualité',
        xaxis_title='Timestamp',
        yaxis_title='PSNR (dB)',
        yaxis2=dict(
            title='SSIM ×100',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig

def render_main_interface():
    """Interface principale de colorisation"""
    # En-tête principal
    st.markdown("""
    <div class="main-header">
        <h1>🎨 Video AI Colorization Studio</h1>
        <p>Restaurez et colorisez vos vidéos en noir et blanc avec l'IA</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload de fichier avec glisser-déposer amélioré
    uploaded_file = st.file_uploader(
        "📁 Glissez-déposez votre vidéo en noir et blanc",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Formats supportés: MP4, AVI, MOV, MKV (Max: 200MB)"
    )
    
    if uploaded_file is not None:
        # Sauvegarder temporairement
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        st.session_state.original_video = video_path
        st.session_state.video_name = uploaded_file.name
        
        # Afficher la vidéo originale avec informations
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📹 Vidéo Originale")
            with st.container():
                st.markdown('<div class="video-container">', unsafe_allow_html=True)
                st.video(video_path)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("📊 Informations")
            display_video_info(video_path)
        
        # Interface de traitement
        st.markdown("---")
        render_processing_interface(video_path)

def render_processing_interface(video_path: str):
    """Interface de traitement avec options avancées"""
    st.subheader("🔧 Options de Traitement")
    
    # Onglets pour différentes méthodes
    tab1, tab2, tab3 = st.tabs(["🤖 IA Avancée", "🎨 Colorisation Classique", "🔧 Préprocessing"])
    
    with tab1:
        render_ai_colorization_tab(video_path)
    
    with tab2:
        render_classical_colorization_tab(video_path)
    
    with tab3:
        render_preprocessing_tab(video_path)

def render_ai_colorization_tab(video_path: str):
    """Onglet de colorisation IA"""
    st.markdown("""
    <div class="processing-card">
        <h4>🧠 Colorisation par Intelligence Artificielle</h4>
        <p>Utilise des modèles GAN pré-entraînés pour une colorisation automatique réaliste.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Modèle IA",
            ["DeOldify (Stable)", "DeOldify (Artistic)", "GAN Custom"],
            help="DeOldify Stable: Meilleur pour les visages et personnes\nDeOldify Artistic: Plus créatif pour les paysages"
        )
        
        quality = st.select_slider(
            "Qualité de rendu",
            options=["Rapide", "Équilibré", "Haute qualité"],
            value="Équilibré"
        )
    
    with col2:
        temporal_consistency = st.checkbox(
            "Cohérence temporelle",
            value=True,
            help="Améliore la consistance des couleurs entre les frames"
        )
        
        enhance_faces = st.checkbox(
            "Amélioration des visages",
            value=False,
            help="Optimisation spéciale pour les visages détectés"
        )
    
    # Options avancées
    with st.expander("⚙️ Paramètres Avancés"):
        batch_size = st.slider("Taille de batch", 1, 8, 4, help="Plus élevé = plus rapide mais plus de mémoire")
        color_intensity = st.slider("Intensité des couleurs", 0.5, 2.0, 1.0, 0.1)
        saturation_boost = st.slider("Boost de saturation", 0.8, 1.5, 1.0, 0.1)
    
    if st.button("🚀 Lancer la Colorisation IA", type="primary"):
        process_with_ai(video_path, {
            'model_type': model_type,
            'quality': quality,
            'temporal_consistency': temporal_consistency,
            'enhance_faces': enhance_faces,
            'batch_size': batch_size,
            'color_intensity': color_intensity,
            'saturation_boost': saturation_boost
        })

def render_classical_colorization_tab(video_path: str):
    """Onglet de colorisation classique"""
    st.markdown("""
    <div class="processing-card">
        <h4>🎨 Colorisation Classique</h4>
        <p>Colorisation basée sur la segmentation d'objets et l'attribution manuelle de couleurs.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Extraction de la première frame pour prévisualisation
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Segmentation pour prévisualisation
        try:
            colorizer = ClassicalColorizer()
            seg = colorizer.segment(frame)
            unique_classes = np.unique(seg)
            
            st.subheader("🔍 Objets détectés dans la première frame")
            
            # Afficher la frame segmentée
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(frame, caption="Frame originale", use_column_width=True)
            
            with col2:
                # Créer une image de segmentation colorée
                seg_colored = np.zeros_like(frame)
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_classes)))
                for i, class_idx in enumerate(unique_classes):
                    mask = seg == class_idx
                    seg_colored[mask] = (colors[i][:3] * 255).astype(np.uint8)
                
                st.image(seg_colored, caption="Segmentation", use_column_width=True)
            
            # Interface de sélection des couleurs
            st.subheader("🎨 Attribution des couleurs")
            class_to_color = {}
            
            cols = st.columns(min(3, len(unique_classes)))
            for i, class_idx in enumerate(unique_classes):
                with cols[i % 3]:
                    class_name = COCO_CLASSES[class_idx] if class_idx < len(COCO_CLASSES) else f"Classe {class_idx}"
                    
                    # Couleur par défaut basée sur la classe
                    default_colors = {
                        'person': '#FDB462',
                        'car': '#80B1D3',
                        'building': '#BEBADA',
                        'tree': '#98DF8A',
                        'sky': '#87CEEB',
                        'road': '#696969'
                    }
                    
                    default_color = default_colors.get(class_name.lower(), '#%02x%02x%02x' % tuple(np.random.randint(100, 255, 3)))
                    
                    color = st.color_picker(
                        f"{class_name}",
                        default_color,
                        key=f"color_{class_idx}"
                    )
                    
                    rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    class_to_color[class_idx] = rgb
            
            # Options de colorisation classique
            with st.expander("⚙️ Options Avancées"):
                edge_smoothing = st.slider("Lissage des contours", 0, 10, 3)
                color_blending = st.slider("Mélange des couleurs", 0.1, 1.0, 0.7, 0.1)
                preserve_details = st.checkbox("Préserver les détails", True)
            
            if st.button("🎨 Appliquer la Colorisation Classique", type="primary"):
                process_with_classical(video_path, class_to_color, {
                    'edge_smoothing': edge_smoothing,
                    'color_blending': color_blending,
                    'preserve_details': preserve_details
                })
                
        except Exception as e:
            st.error(f"Erreur lors de la segmentation: {e}")
    else:
        st.error("Impossible de lire la vidéo.")

def render_preprocessing_tab(video_path: str):
    """Onglet de préprocessing"""
    st.markdown("""
    <div class="processing-card">
        <h4>🔧 Prétraitement et Restauration</h4>
        <p>Améliorez la qualité de votre vidéo avant colorisation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧹 Débruitage")
        denoising_method = st.selectbox(
            "Méthode de débruitage",
            ["Aucun", "Filtre médian", "Filtre bilatéral", "Non-local means", "BM3D"]
        )
        
        if denoising_method != "Aucun":
            denoising_strength = st.slider("Intensité du débruitage", 1, 10, 5)
        
        st.subheader("📈 Amélioration du contraste")
        contrast_method = st.selectbox(
            "Méthode de contraste",
            ["Aucun", "CLAHE", "Égalisation d'histogramme", "Gamma correction"]
        )
        
        if contrast_method != "Aucun":
            contrast_strength = st.slider("Intensité du contraste", 0.5, 2.0, 1.0, 0.1)
    
    with col2:
        st.subheader("🔍 Amélioration de la netteté")
        sharpening = st.checkbox("Activer l'amélioration de netteté")
        
        if sharpening:
            sharpening_strength = st.slider("Intensité de la netteté", 0.1, 2.0, 1.0, 0.1)
        
        st.subheader("📏 Redimensionnement")
        resize_option = st.selectbox(
            "Redimensionnement",
            ["Aucun", "720p", "1080p", "4K", "Personnalisé"]
        )
        
        if resize_option == "Personnalisé":
            col_w, col_h = st.columns(2)
            with col_w:
                custom_width = st.number_input("Largeur", 100, 4000, 1920)
            with col_h:
                custom_height = st.number_input("Hauteur", 100, 4000, 1080)
    
    if st.button("🔧 Appliquer le Prétraitement", type="primary"):
        preprocessing_params = {
            'denoising_method': denoising_method,
            'denoising_strength': denoising_strength if denoising_method != "Aucun" else 0,
            'contrast_method': contrast_method,
            'contrast_strength': contrast_strength if contrast_method != "Aucun" else 1.0,
            'sharpening': sharpening,
            'sharpening_strength': sharpening_strength if sharpening else 1.0,
            'resize_option': resize_option,
            'custom_width': custom_width if resize_option == "Personnalisé" else None,
            'custom_height': custom_height if resize_option == "Personnalisé" else None
        }
        process_preprocessing(video_path, preprocessing_params)

def process_with_ai(video_path: str, params: Dict):
    """Traitement avec IA"""
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔄 Initialisation du modèle IA...")
            progress_bar.progress(10)
            
            colorizer = AIColorizer()
            
            status_text.text("🎨 Colorisation en cours...")
            progress_bar.progress(30)
            
            # Simuler le traitement progressif
            import time
            for i in range(30, 90, 10):
                time.sleep(1)
                progress_bar.progress(i)
                status_text.text(f"🎨 Traitement... {i}%")
            
            output_path = colorizer.colorize(video_path)
            
            progress_bar.progress(95)
            status_text.text("📊 Calcul des métriques...")
            
            metrics = calculate_metrics(video_path, output_path)
            
            progress_bar.progress(100)
            status_text.text("✅ Traitement terminé!")
            
            # Sauvegarder dans l'historique
            save_processing_history(
                st.session_state.get('video_name', 'video'),
                f"IA - {params['model_type']}",
                metrics,
                output_path
            )
            
            # Afficher les résultats
            display_results(output_path, metrics, "IA")
            
        except Exception as e:
            st.error(f"Erreur lors du traitement IA: {e}")
            progress_bar.empty()
            status_text.empty()

def process_with_classical(video_path: str, class_to_color: Dict, params: Dict):
    """Traitement avec méthode classique"""
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔄 Initialisation de la segmentation...")
            progress_bar.progress(20)
            
            colorizer = ClassicalColorizer()
            
            status_text.text("🎨 Colorisation par segmentation...")
            progress_bar.progress(50)
            
            output_path = colorizer.colorize(video_path, class_to_color)
            
            progress_bar.progress(80)
            status_text.text("📊 Calcul des métriques...")
            
            metrics = calculate_metrics(video_path, output_path)
            
            progress_bar.progress(100)
            status_text.text("✅ Traitement terminé!")
            
            # Sauvegarder dans l'historique
            save_processing_history(
                st.session_state.get('video_name', 'video'),
                "Classique - Segmentation",
                metrics,
                output_path
            )
            
            # Afficher les résultats
            display_results(output_path, metrics, "Classique")
            
        except Exception as e:
            st.error(f"Erreur lors du traitement classique: {e}")
            progress_bar.empty()
            status_text.empty()

def process_preprocessing(video_path: str, params: Dict):
    """Traitement de préprocessing"""
    st.info("🔧 Fonctionnalité de préprocessing en cours de développement...")

def display_results(output_path: str, metrics: Dict, method: str):
    """Affiche les résultats du traitement"""
    st.markdown("---")
    st.subheader("🎉 Résultats du Traitement")
    
    # Vidéo résultante
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"### 🎬 Vidéo Colorisée ({method})")
        with st.container():
            st.markdown('<div class="video-container">', unsafe_allow_html=True)
            st.video(output_path)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Bouton de téléchargement
        with open(output_path, 'rb') as f:
            st.download_button(
                label="📥 Télécharger la vidéo colorisée",
                data=f,
                file_name=f"colorized_{method.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                mime="video/mp4",
                type="primary"
            )
    
    with col2:
        st.markdown("### 📊 Métriques de Qualité")
        
        # Cartes de métriques
        if metrics:
            st.markdown(f"""
            <div class="metric-card">
                <h4>PSNR</h4>
                <h2>{metrics.get('psnr', 0):.2f} dB</h2>
                <p>Rapport Signal/Bruit</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>SSIM</h4>
                <h2>{metrics.get('ssim', 0):.3f}</h2>
                <p>Similarité Structurelle</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Indicateur de qualité
        if metrics and 'psnr' in metrics:
            psnr = metrics['psnr']
            if psnr > 25:
                quality = "Excellente 🌟"
                color = "#28a745"
            elif psnr > 20:
                quality = "Bonne 👍"
                color = "#ffc107"
            else:
                quality = "Acceptable ⚠️"
                color = "#dc3545"
            
            st.markdown(f"""
            <div style="background: {color}20; padding: 1rem; border-radius: 10px; border-left: 4px solid {color};">
                <h4 style="color: {color}; margin: 0;">Qualité: {quality}</h4>
            </div>
            """, unsafe_allow_html=True)

def render_history_page():
    """Page d'historique des traitements"""
    st.markdown("""
    <div class="main-header">
        <h1>📚 Historique des Traitements</h1>
        <p>Consultez vos traitements précédents et leurs métriques</p>
    </div>
    """, unsafe_allow_html=True)
    
    history = load_processing_history()
    
    if not history:
        st.info("📝 Aucun traitement dans l'historique.")
        return
    
    # Graphique des métriques
    chart = create_comparison_chart(history)
    if chart:
        st.plotly_chart(chart, use_container_width=True)
    
    # Tableau de l'historique
    st.subheader("📋 Historique Détaillé")
    
    for i, entry in enumerate(reversed(history[-20:])):  # 20 dernières entrées
        with st.expander(f"🎬 {entry['video_name']} - {entry['method']} ({entry['timestamp'][:19]})"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Méthode:** {entry['method']}")
                st.write(f"**Date:** {entry['timestamp'][:19]}")
            
            with col2:
                if entry.get('metrics'):
                    st.write(f"**PSNR:** {entry['metrics'].get('psnr', 'N/A'):.2f} dB")
                    st.write(f"**SSIM:** {entry['metrics'].get('ssim', 'N/A'):.3f}")
            
            with col3:
                if os.path.exists(entry['output_path']):
                    with open(entry['output_path'], 'rb') as f:
                        st.download_button(
                            "📥 Télécharger",
                            data=f,
                            file_name=f"redownload_{i}.mp4",
                            mime="video/mp4"
                        )
                else:
                    st.write("❌ Fichier non disponible")

def render_help_page():
    """Page d'aide et documentation"""
    st.markdown("""
    <div class="main-header">
        <h1>❓ Aide et Documentation</h1>
        <p>Guide d'utilisation et conseils pour de meilleurs résultats</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Démarrage Rapide", "🤖 IA vs Classique", "⚙️ Paramètres", "🔧 Dépannage"])
    
    with tab1:
        st.markdown("""
        ## 🚀 Guide de Démarrage Rapide
        
        ### 1. Préparez votre vidéo
        - Format supporté: MP4, AVI, MOV, MKV
        - Taille maximum: 200MB
        - Résolution recommandée: 720p-1080p
        
        ### 2. Choisissez votre méthode
        - **IA Avancée**: Pour des résultats automatiques et réalistes
        - **Classique**: Pour un contrôle précis des couleurs
        - **Préprocessing**: Pour améliorer la qualité avant colorisation
        
        ### 3. Ajustez les paramètres
        - Commencez avec les paramètres par défaut
        - Ajustez selon vos besoins spécifiques
        
        ### 4. Téléchargez le résultat
        - Comparez les métriques PSNR/SSIM
        - Téléchargez votre vidéo colorisée
        """)
    
    with tab2:
        st.markdown("""
        ## 🤖 Comparaison des Méthodes
        
        | Aspect | IA Avancée | Classique |
        |--------|------------|-----------|
        | **Facilité** | ⭐⭐⭐⭐⭐ Automatique | ⭐⭐⭐ Manuel |
        | **Réalisme** | ⭐⭐⭐⭐⭐ Très réaliste | ⭐⭐⭐ Contrôlé |
        | **Vitesse** | ⭐⭐⭐ Moyenne | ⭐⭐⭐⭐ Rapide |
        | **Contrôle** | ⭐⭐ Limité | ⭐⭐⭐⭐⭐ Total |
        | **Visages** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Bon |
        | **Paysages** | ⭐⭐⭐⭐ Très bon | ⭐⭐⭐⭐ Bon |
        
        ### Quand utiliser l'IA ?
        - Vidéos avec des personnes
        - Scènes complexes
        - Recherche de réalisme maximum
        
        ### Quand utiliser la méthode classique ?
        - Besoin de couleurs spécifiques
        - Objets aux couleurs non-naturelles
        - Contrôle artistique précis
        """)
    
    with tab3:
        st.markdown("""
        ## ⚙️ Guide des Paramètres
        
        ### Paramètres IA
        - **Modèle**: 
          - *Stable*: Meilleur pour les visages
          - *Artistic*: Plus créatif pour les paysages
        - **Qualité**: Balance vitesse/qualité
        - **Cohérence temporelle**: Évite le scintillement
        - **Batch size**: Plus élevé = plus rapide (mais plus de mémoire)
        
        ### Paramètres Classiques
        - **Lissage des contours**: Réduit les artefacts aux bordures
        - **Mélange des couleurs**: Transition douce entre zones
        - **Préserver les détails**: Maintient la netteté originale
        
        ### Métriques de Qualité
        - **PSNR > 25 dB**: Excellente qualité
        - **PSNR 20-25 dB**: Bonne qualité
        - **SSIM > 0.8**: Très bonne similarité structurelle
        """)
    
    with tab4:
        st.markdown("""
        ## 🔧 Dépannage
        
        ### Problèmes Courants
        
        **❌ "Erreur de chargement de vidéo"**
        - Vérifiez le format (MP4 recommandé)
        - Réduisez la taille du fichier
        - Essayez un autre navigateur
        
        **❌ "Traitement très lent"**
        - Réduisez la résolution de la vidéo
        - Diminuez le batch size
        - Utilisez la qualité "Rapide"
        
        **❌ "Couleurs non réalistes"**
        - Essayez le modèle "Stable" au lieu d'"Artistic"
        - Activez l'amélioration des visages
        - Ajustez l'intensité des couleurs
        
        **❌ "Scintillement dans la vidéo"**
        - Activez la cohérence temporelle
        - Augmentez le lissage temporel
        - Réduisez l'intensité des couleurs
        
        ### Contact Support
        Si le problème persiste, vérifiez les logs dans la console du navigateur (F12).
        """)

def main():
    """Fonction principale de l'application"""
    # Charger le CSS personnalisé
    load_custom_css()
    
    # Navigation principale
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/667eea/white?text=AI+Studio", use_column_width=True)
        
        selected = option_menu(
            menu_title="Navigation",
            options=["🎨 Colorisation", "📚 Historique", "❓ Aide"],
            icons=["palette", "clock-history", "question-circle"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#667eea", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#667eea"},
            },
        )
    
    # Affichage des pages selon la sélection
    if selected == "🎨 Colorisation":
        render_main_interface()
    elif selected == "📚 Historique":
        render_history_page()
    elif selected == "❓ Aide":
        render_help_page()

if __name__ == "__main__":
    main()
