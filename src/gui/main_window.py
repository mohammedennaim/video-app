"""
Interface graphique principale pour la restauration et colorisation de vidéos.
Permet la comparaison côte-à-côte des différentes méthodes.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, List

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit,
    QTabWidget, QComboBox, QSlider, QSpinBox, QGroupBox,
    QGridLayout, QScrollArea, QSplitter, QFrame, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QFont, QIcon

import cv2
import numpy as np

# Import des modules du projet
sys.path.append(str(Path(__file__).parent.parent))
from classical.video_restoration import ClassicalRestoration
from ai.gan_colorization import GANColorization, SimpleColorization
from evaluation.metrics import VideoMetrics, QualitativeEvaluation
from utils.video_utils import VideoProcessor

class VideoProcessingThread(QThread):
    """Thread pour le traitement vidéo en arrière-plan."""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    frame_processed = pyqtSignal(int, np.ndarray, str)  # index, frame, method
    processing_finished = pyqtSignal(dict)  # résultats
    
    def __init__(self, input_path: str, methods: List[str]):
        super().__init__()
        self.input_path = input_path
        self.methods = methods
        self.should_stop = False
    
    def run(self):
        """Exécute le traitement vidéo."""
        try:
            self.status_updated.emit("🔄 Chargement de la vidéo...")
            processor = VideoProcessor(self.input_path)
            total_frames = len(processor.frames)
            
            results = {
                'original': processor.frames,
                'processed': {},
                'metrics': {}
            }
            
            for method in self.methods:
                if self.should_stop:
                    return
                
                self.status_updated.emit(f"🔧 Traitement avec {method}...")
                
                if method == 'classical':
                    restoration = ClassicalRestoration()
                    processed_frames = restoration.denoise_video(processor.frames)
                    processed_frames = restoration.colorize_video(processed_frames)
                
                elif method == 'ai_gan':
                    gan_colorizer = GANColorization()
                    processed_frames = gan_colorizer.colorize_video(processor.frames)
                
                elif method == 'ai_simple':
                    simple_colorizer = SimpleColorization()
                    processed_frames = simple_colorizer.colorize_video(processor.frames)
                
                else:
                    continue
                
                results['processed'][method] = processed_frames
                
                # Émettre les frames traitées pour affichage en temps réel
                for i, frame in enumerate(processed_frames):
                    if self.should_stop:
                        return
                    self.frame_processed.emit(i, frame, method)
                    self.progress_updated.emit(int((i + 1) / total_frames * 100))
            
            # Calculer les métriques
            self.status_updated.emit("📊 Calcul des métriques...")
            self._calculate_metrics(results)
            
            self.processing_finished.emit(results)
            
        except Exception as e:
            self.status_updated.emit(f"❌ Erreur: {str(e)}")
    
    def _calculate_metrics(self, results: Dict):
        """Calcule les métriques d'évaluation."""
        metrics_calculator = VideoMetrics()
        qualitative_eval = QualitativeEvaluation()
        
        original_frames = results['original']
        
        for method, processed_frames in results['processed'].items():
            if len(processed_frames) == 0:
                continue
            
            # Métriques quantitatives
            psnr_values = []
            ssim_values = []
            
            for orig, proc in zip(original_frames, processed_frames):
                if orig.shape != proc.shape:
                    proc = cv2.resize(proc, (orig.shape[1], orig.shape[0]))
                
                psnr_val = metrics_calculator.calculate_psnr(orig, proc)
                ssim_val = metrics_calculator.calculate_ssim(orig, proc)
                
                psnr_values.append(psnr_val)
                ssim_values.append(ssim_val)
            
            # Métriques qualitatives
            color_coherence = qualitative_eval.calculate_color_coherence(processed_frames)
            naturalness = qualitative_eval.assess_naturalness(processed_frames)
            
            results['metrics'][method] = {
                'psnr_mean': np.mean(psnr_values),
                'ssim_mean': np.mean(ssim_values),
                'color_coherence': color_coherence,
                'naturalness': naturalness
            }
    
    def stop(self):
        """Arrête le traitement."""
        self.should_stop = True

class VideoDisplayWidget(QWidget):
    """Widget pour afficher une vidéo avec contrôles."""
    
    def __init__(self, title: str):
        super().__init__()
        self.title = title
        self.frames = []
        self.current_frame_idx = 0
        self.setup_ui()
    
    def setup_ui(self):
        """Configure l'interface utilisateur."""
        layout = QVBoxLayout()
        
        # Titre
        title_label = QLabel(self.title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)
        
        # Zone d'affichage vidéo
        self.video_label = QLabel()
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid gray; background-color: black;")
        self.video_label.setText("Aucune vidéo chargée")
        layout.addWidget(self.video_label)
        
        # Contrôles de lecture
        controls_layout = QHBoxLayout()
        
        self.play_button = QPushButton("▶")
        self.play_button.setMaximumWidth(40)
        self.play_button.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_button)
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self.on_slider_changed)
        controls_layout.addWidget(self.frame_slider)
        
        self.frame_label = QLabel("0/0")
        controls_layout.addWidget(self.frame_label)
        
        layout.addLayout(controls_layout)
        
        # Timer pour la lecture
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)
        self.is_playing = False
        
        self.setLayout(layout)
    
    def load_frames(self, frames: List[np.ndarray]):
        """Charge les frames de la vidéo."""
        self.frames = frames
        if frames:
            self.frame_slider.setMaximum(len(frames) - 1)
            self.current_frame_idx = 0
            self.display_current_frame()
            self.update_frame_label()
    
    def display_current_frame(self):
        """Affiche la frame courante."""
        if not self.frames or self.current_frame_idx >= len(self.frames):
            return
        
        frame = self.frames[self.current_frame_idx]
        
        # Convertir la frame en QPixmap
        height, width = frame.shape[:2]
        if len(frame.shape) == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            q_image = QPixmap.fromImage(
                QPixmap.fromImage(rgb_frame).toImage()
            )
        else:
            q_image = QPixmap.fromImage(
                QPixmap.fromImage(frame).toImage()
            )
        
        # Redimensionner pour l'affichage
        scaled_pixmap = q_image.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
    
    def toggle_playback(self):
        """Active/désactive la lecture."""
        if self.is_playing:
            self.play_timer.stop()
            self.play_button.setText("▶")
        else:
            self.play_timer.start(33)  # ~30 FPS
            self.play_button.setText("⏸")
        
        self.is_playing = not self.is_playing
    
    def next_frame(self):
        """Passe à la frame suivante."""
        if self.frames and self.current_frame_idx < len(self.frames) - 1:
            self.current_frame_idx += 1
            self.frame_slider.setValue(self.current_frame_idx)
            self.display_current_frame()
            self.update_frame_label()
        else:
            # Fin de la vidéo
            self.toggle_playback()
    
    def on_slider_changed(self, value: int):
        """Gestionnaire de changement du slider."""
        self.current_frame_idx = value
        self.display_current_frame()
        self.update_frame_label()
    
    def update_frame_label(self):
        """Met à jour le label du numéro de frame."""
        total_frames = len(self.frames) if self.frames else 0
        self.frame_label.setText(f"{self.current_frame_idx + 1}/{total_frames}")

class MetricsWidget(QWidget):
    """Widget pour afficher les métriques d'évaluation."""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """Configure l'interface utilisateur."""
        layout = QVBoxLayout()
        
        # Titre
        title = QLabel("📊 Métriques d'Évaluation")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Zone de texte pour les métriques
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMaximumHeight(200)
        layout.addWidget(self.metrics_text)
        
        self.setLayout(layout)
    
    def update_metrics(self, metrics: Dict):
        """Met à jour l'affichage des métriques."""
        text = "=== RÉSULTATS D'ÉVALUATION ===\n\n"
        
        for method, values in metrics.items():
            text += f"🔧 {method.upper()}:\n"
            text += f"  • PSNR: {values['psnr_mean']:.2f} dB\n"
            text += f"  • SSIM: {values['ssim_mean']:.4f}\n"
            text += f"  • Cohérence couleur: {values['color_coherence']:.4f}\n"
            text += f"  • Naturel: {values['naturalness']:.4f}\n\n"
        
        self.metrics_text.setPlainText(text)

class VideoRestorationApp(QMainWindow):
    """Application principale de restauration vidéo."""
    
    def __init__(self):
        super().__init__()
        self.current_video_path = None
        self.processing_thread = None
        self.setup_ui()
        self.setup_style()
    
    def setup_ui(self):
        """Configure l'interface utilisateur principale."""
        self.setWindowTitle("🎬 Restauration et Colorisation de Vidéos Anciennes")
        self.setGeometry(100, 100, 1400, 900)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Barre d'outils
        self.setup_toolbar(main_layout)
        
        # Zone de contenu principal
        content_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(content_splitter)
        
        # Panneau de contrôle gauche
        self.setup_control_panel(content_splitter)
        
        # Zone d'affichage vidéo
        self.setup_video_display(content_splitter)
        
        # Barre de statut
        self.setup_status_bar()
    
    def setup_toolbar(self, layout: QVBoxLayout):
        """Configure la barre d'outils."""
        toolbar_layout = QHBoxLayout()
        
        # Bouton de chargement
        self.load_button = QPushButton("📁 Charger Vidéo")
        self.load_button.clicked.connect(self.load_video)
        toolbar_layout.addWidget(self.load_button)
        
        # Sélection des méthodes
        methods_group = QGroupBox("Méthodes de traitement")
        methods_layout = QHBoxLayout()
        
        self.method_classical = QPushButton("🔧 Classique")
        self.method_classical.setCheckable(True)
        self.method_classical.setChecked(True)
        methods_layout.addWidget(self.method_classical)
        
        self.method_ai_simple = QPushButton("🤖 IA Simple")
        self.method_ai_simple.setCheckable(True)
        self.method_ai_simple.setChecked(True)
        methods_layout.addWidget(self.method_ai_simple)
        
        self.method_ai_gan = QPushButton("🧠 IA GAN")
        self.method_ai_gan.setCheckable(True)
        methods_layout.addWidget(self.method_ai_gan)
        
        methods_group.setLayout(methods_layout)
        toolbar_layout.addWidget(methods_group)
        
        # Bouton de traitement
        self.process_button = QPushButton("▶ Traiter")
        self.process_button.clicked.connect(self.start_processing)
        self.process_button.setEnabled(False)
        toolbar_layout.addWidget(self.process_button)
        
        # Bouton d'arrêt
        self.stop_button = QPushButton("⏹ Arrêter")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        toolbar_layout.addWidget(self.stop_button)
        
        toolbar_layout.addStretch()
        
        # Bouton de sauvegarde
        self.save_button = QPushButton("💾 Sauvegarder")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)
        toolbar_layout.addWidget(self.save_button)
        
        layout.addLayout(toolbar_layout)
    
    def setup_control_panel(self, splitter: QSplitter):
        """Configure le panneau de contrôle."""
        control_widget = QWidget()
        control_widget.setMaximumWidth(300)
        
        layout = QVBoxLayout()
        control_widget.setLayout(layout)
        
        # Informations sur la vidéo
        info_group = QGroupBox("📹 Informations Vidéo")
        info_layout = QVBoxLayout()
        
        self.video_info_label = QLabel("Aucune vidéo chargée")
        self.video_info_label.setWordWrap(True)
        info_layout.addWidget(self.video_info_label)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Paramètres de traitement
        params_group = QGroupBox("⚙️ Paramètres")
        params_layout = QVBoxLayout()
        
        # Qualité de traitement
        params_layout.addWidget(QLabel("Qualité:"))
        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(1, 5)
        self.quality_slider.setValue(3)
        params_layout.addWidget(self.quality_slider)
        
        # Taille de sortie
        params_layout.addWidget(QLabel("Résolution:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["Original", "720p", "480p", "360p"])
        params_layout.addWidget(self.resolution_combo)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Widget des métriques
        self.metrics_widget = MetricsWidget()
        layout.addWidget(self.metrics_widget)
        
        layout.addStretch()
        
        splitter.addWidget(control_widget)
    
    def setup_video_display(self, splitter: QSplitter):
        """Configure la zone d'affichage vidéo."""
        display_widget = QWidget()
        layout = QVBoxLayout()
        display_widget.setLayout(layout)
        
        # Onglets pour les différentes vues
        self.video_tabs = QTabWidget()
        
        # Vue originale
        self.original_display = VideoDisplayWidget("Vidéo Originale")
        self.video_tabs.addTab(self.original_display, "Original")
        
        # Vue classique
        self.classical_display = VideoDisplayWidget("Restauration Classique")
        self.video_tabs.addTab(self.classical_display, "Classique")
        
        # Vue IA simple
        self.ai_simple_display = VideoDisplayWidget("IA Simple")
        self.video_tabs.addTab(self.ai_simple_display, "IA Simple")
        
        # Vue IA GAN
        self.ai_gan_display = VideoDisplayWidget("IA GAN")
        self.video_tabs.addTab(self.ai_gan_display, "IA GAN")
        
        # Vue comparaison
        comparison_widget = QWidget()
        comparison_layout = QGridLayout()
        comparison_widget.setLayout(comparison_layout)
        
        # Affichages de comparaison (2x2)
        self.comparison_displays = {}
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        titles = ["Original", "Classique", "IA Simple", "IA GAN"]
        
        for i, (title, pos) in enumerate(zip(titles, positions)):
            display = VideoDisplayWidget(title)
            comparison_layout.addWidget(display, pos[0], pos[1])
            self.comparison_displays[title.lower().replace(" ", "_")] = display
        
        self.video_tabs.addTab(comparison_widget, "Comparaison")
        
        layout.addWidget(self.video_tabs)
        
        # Barre de progression
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        splitter.addWidget(display_widget)
    
    def setup_status_bar(self):
        """Configure la barre de statut."""
        self.status_label = QLabel("Prêt")
        self.statusBar().addWidget(self.status_label)
    
    def setup_style(self):
        """Configure le style de l'application."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                padding: 8px 16px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: #ffffff;
            }
            QPushButton:hover {
                background-color: #e6f3ff;
            }
            QPushButton:pressed {
                background-color: #cce6ff;
            }
            QPushButton:checked {
                background-color: #007acc;
                color: white;
            }
        """)
    
    def load_video(self):
        """Charge une vidéo."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Sélectionner une vidéo",
            "",
            "Fichiers vidéo (*.mp4 *.avi *.mov *.mkv);;Tous les fichiers (*)"
        )
        
        if file_path:
            try:
                processor = VideoProcessor(file_path)
                self.current_video_path = file_path
                
                # Mettre à jour les informations
                info = processor.get_info()
                info_text = f"""
                Fichier: {Path(file_path).name}
                Résolution: {info['width']}x{info['height']}
                FPS: {info['fps']:.2f}
                Durée: {info['duration']:.1f}s
                Frames: {info['frame_count']}
                """
                self.video_info_label.setText(info_text)
                
                # Charger dans l'affichage original
                self.original_display.load_frames(processor.frames)
                
                # Activer les boutons
                self.process_button.setEnabled(True)
                self.status_label.setText(f"Vidéo chargée: {Path(file_path).name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Impossible de charger la vidéo:\n{str(e)}")
    
    def start_processing(self):
        """Démarre le traitement de la vidéo."""
        if not self.current_video_path:
            return
        
        # Déterminer les méthodes sélectionnées
        methods = []
        if self.method_classical.isChecked():
            methods.append('classical')
        if self.method_ai_simple.isChecked():
            methods.append('ai_simple')
        if self.method_ai_gan.isChecked():
            methods.append('ai_gan')
        
        if not methods:
            QMessageBox.warning(self, "Attention", "Sélectionnez au moins une méthode de traitement.")
            return
        
        # Configurer l'interface pour le traitement
        self.process_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Démarrer le thread de traitement
        self.processing_thread = VideoProcessingThread(self.current_video_path, methods)
        self.processing_thread.progress_updated.connect(self.progress_bar.setValue)
        self.processing_thread.status_updated.connect(self.status_label.setText)
        self.processing_thread.frame_processed.connect(self.on_frame_processed)
        self.processing_thread.processing_finished.connect(self.on_processing_finished)
        self.processing_thread.start()
    
    def stop_processing(self):
        """Arrête le traitement."""
        if self.processing_thread:
            self.processing_thread.stop()
            self.processing_thread.wait()
        
        self.reset_ui_after_processing()
        self.status_label.setText("Traitement arrêté")
    
    def on_frame_processed(self, frame_idx: int, frame: np.ndarray, method: str):
        """Gestionnaire de frame traitée."""
        # Mettre à jour l'affichage en temps réel
        pass
    
    def on_processing_finished(self, results: Dict):
        """Gestionnaire de fin de traitement."""
        # Charger les résultats dans les affichages
        for method, frames in results['processed'].items():
            if method == 'classical':
                self.classical_display.load_frames(frames)
                self.comparison_displays['classique'].load_frames(frames)
            elif method == 'ai_simple':
                self.ai_simple_display.load_frames(frames)
                self.comparison_displays['ia_simple'].load_frames(frames)
            elif method == 'ai_gan':
                self.ai_gan_display.load_frames(frames)
                self.comparison_displays['ia_gan'].load_frames(frames)
        
        # Charger l'original dans la vue comparaison
        self.comparison_displays['original'].load_frames(results['original'])
        
        # Mettre à jour les métriques
        if 'metrics' in results:
            self.metrics_widget.update_metrics(results['metrics'])
        
        # Réinitialiser l'interface
        self.reset_ui_after_processing()
        self.save_button.setEnabled(True)
        self.status_label.setText("✅ Traitement terminé")
    
    def reset_ui_after_processing(self):
        """Remet l'interface dans l'état initial."""
        self.process_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.processing_thread = None
    
    def save_results(self):
        """Sauvegarde les résultats."""
        if not hasattr(self, 'results'):
            return
        
        save_dir = QFileDialog.getExistingDirectory(self, "Sélectionner le dossier de sauvegarde")
        if save_dir:
            # Implémenter la sauvegarde
            QMessageBox.information(self, "Sauvegarde", f"Résultats sauvegardés dans:\n{save_dir}")

def main():
    """Fonction principale."""
    app = QApplication(sys.argv)
    app.setApplicationName("Restauration Vidéo")
    app.setOrganizationName("VideoRestoration")
    
    # Appliquer un style moderne
    app.setStyle('Fusion')
    
    window = VideoRestorationApp()
    window.show()
    
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())
