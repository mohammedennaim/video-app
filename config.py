"""
Configuration et styles pour l'application de colorisation vidéo
"""

import streamlit as st

# Configuration des couleurs du thème
THEME_COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

# Configuration de l'application
APP_CONFIG = {
    'page_title': 'Video AI Colorization Studio',
    'page_icon': '🎨',
    'layout': 'wide',
    'max_file_size_mb': 200,
    'supported_formats': ['mp4', 'avi', 'mov', 'mkv'],
    'output_dir': 'static/output',
    'history_dir': 'static/history',
    'max_history_entries': 50
}

# CSS personnalisé avancé
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --success-color: #28a745;
    --warning-color: #ffc107;
    --danger-color: #dc3545;
    --light-color: #f8f9fa;
    --dark-color: #343a40;
    --border-radius: 10px;
    --box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    --transition: all 0.3s ease;
}

/* Base styles */
.main .block-container {
    padding-top: 1rem;
    font-family: 'Inter', sans-serif;
}

/* Header styles */
.main-header {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    padding: 2rem;
    border-radius: var(--border-radius);
    text-align: center;
    color: white;
    margin-bottom: 2rem;
    box-shadow: var(--box-shadow);
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: repeating-linear-gradient(
        45deg,
        transparent,
        transparent 10px,
        rgba(255,255,255,0.1) 10px,
        rgba(255,255,255,0.1) 20px
    );
    animation: shimmer 3s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%) translateY(-100%); }
    100% { transform: translateX(100%) translateY(100%); }
}

.main-header h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
    position: relative;
    z-index: 1;
}

.main-header p {
    margin: 0.5rem 0 0 0;
    font-size: 1.1rem;
    opacity: 0.9;
    position: relative;
    z-index: 1;
}

/* Card styles */
.metric-card, .processing-card, .success-card {
    padding: 1.5rem;
    border-radius: var(--border-radius);
    margin: 1rem 0;
    transition: var(--transition);
    border: 1px solid #e9ecef;
}

.metric-card {
    background: white;
    box-shadow: var(--box-shadow);
    border-left: 4px solid var(--primary-color);
    text-align: center;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.metric-card h2 {
    color: var(--primary-color);
    margin: 0.5rem 0;
    font-size: 2rem;
    font-weight: 600;
}

.metric-card h4 {
    color: var(--dark-color);
    margin: 0;
    font-size: 1.1rem;
    font-weight: 500;
}

.metric-card p {
    color: #6c757d;
    margin: 0.5rem 0 0 0;
    font-size: 0.9rem;
}

.processing-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-left: 4px solid var(--warning-color);
}

.processing-card h4 {
    color: var(--dark-color);
    margin: 0 0 0.5rem 0;
    font-size: 1.2rem;
    font-weight: 600;
}

.success-card {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border-left: 4px solid var(--success-color);
}

/* Video container */
.video-container {
    border-radius: var(--border-radius);
    overflow: hidden;
    box-shadow: var(--box-shadow);
    background: #000;
    position: relative;
    transition: var(--transition);
}

.video-container:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
}

/* Button styles */
.stButton > button {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    color: white;
    border: none;
    border-radius: var(--border-radius);
    padding: 0.75rem 1.5rem;
    font-weight: 500;
    transition: var(--transition);
    box-shadow: var(--box-shadow);
    position: relative;
    overflow: hidden;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.stButton > button:active {
    transform: translateY(0);
}

.stButton > button[data-baseweb="button"][kind="primary"] {
    background: linear-gradient(135deg, var(--success-color) 0%, #20c997 100%);
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    border-radius: 10px;
}

/* Sidebar */
.css-1d391kg {
    background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
}

/* File uploader */
.stFileUploader {
    border: 2px dashed var(--primary-color);
    border-radius: var(--border-radius);
    padding: 2rem;
    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
    text-align: center;
    transition: var(--transition);
}

.stFileUploader:hover {
    border-color: var(--secondary-color);
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: var(--light-color);
    border-radius: var(--border-radius);
    border: 1px solid #dee2e6;
    transition: var(--transition);
}

.stTabs [data-baseweb="tab"]:hover {
    background: #e9ecef;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    color: white;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--light-color);
    border-radius: var(--border-radius);
    border: 1px solid #dee2e6;
    transition: var(--transition);
}

.streamlit-expanderHeader:hover {
    background: #e9ecef;
    box-shadow: var(--box-shadow);
}

/* Metrics display */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}

/* Animation classes */
.fade-in {
    animation: fadeIn 0.5s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.slide-in {
    animation: slideIn 0.5s ease-out;
}

@keyframes slideIn {
    from { transform: translateX(-100%); }
    to { transform: translateX(0); }
}

/* Loading spinner */
.loading-spinner {
    border: 4px solid #f3f3f3;
    border-top: 4px solid var(--primary-color);
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
    margin: 20px auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Responsive design */
@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2rem;
    }
    
    .metric-card {
        margin: 0.5rem 0;
        padding: 1rem;
    }
    
    .stButton > button {
        padding: 0.5rem 1rem;
        width: 100%;
    }
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    .metric-card {
        background: #2d3748;
        color: white;
        border-color: #4a5568;
    }
    
    .processing-card {
        background: #2d3748;
        color: white;
    }
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--secondary-color);
}

/* Notification styles */
.notification {
    padding: 1rem;
    border-radius: var(--border-radius);
    margin: 1rem 0;
    border-left: 4px solid;
    animation: slideIn 0.3s ease-out;
}

.notification.success {
    background: #d4edda;
    border-color: var(--success-color);
    color: #155724;
}

.notification.warning {
    background: #fff3cd;
    border-color: var(--warning-color);
    color: #856404;
}

.notification.error {
    background: #f8d7da;
    border-color: var(--danger-color);
    color: #721c24;
}

.notification.info {
    background: #d1ecf1;
    border-color: #17a2b8;
    color: #0c5460;
}
</style>
"""

# JavaScript pour des interactions avancées
CUSTOM_JS = """
<script>
// Fonction pour améliorer l'expérience utilisateur
function enhanceUI() {
    // Ajouter des classes d'animation aux éléments
    const elements = document.querySelectorAll('.metric-card, .processing-card');
    elements.forEach((el, index) => {
        el.style.animationDelay = `${index * 0.1}s`;
        el.classList.add('fade-in');
    });
    
    // Smooth scroll pour la navigation
    const links = document.querySelectorAll('a[href^="#"]');
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
}

// Exécuter après le chargement
document.addEventListener('DOMContentLoaded', enhanceUI);

// Observer pour les nouveaux éléments
const observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
        if (mutation.addedNodes.length) {
            enhanceUI();
        }
    });
});

observer.observe(document.body, {
    childList: true,
    subtree: true
});
</script>
"""

def apply_custom_styles():
    """Applique les styles personnalisés à l'application"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(CUSTOM_JS, unsafe_allow_html=True)

def get_theme_color(color_name: str) -> str:
    """Retourne une couleur du thème"""
    return THEME_COLORS.get(color_name, '#000000')

def create_notification(message: str, notification_type: str = 'info') -> str:
    """Crée une notification stylisée"""
    return f"""
    <div class="notification {notification_type}">
        {message}
    </div>
    """

def create_metric_card(title: str, value: str, description: str = "", icon: str = "") -> str:
    """Crée une carte de métrique stylisée"""
    return f"""
    <div class="metric-card fade-in">
        <h4>{icon} {title}</h4>
        <h2>{value}</h2>
        {f"<p>{description}</p>" if description else ""}
    </div>
    """

def create_loading_spinner() -> str:
    """Crée un spinner de chargement"""
    return """
    <div class="loading-spinner"></div>
    """
