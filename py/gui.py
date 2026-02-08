from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PySide6.QtGui import *
import sys
import os

class AnalysisWorker(QThread):
    finished = Signal(list)
    error = Signal(str)
    status = Signal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            from processor.video_processor import extract_audio
            from processor.speech_to_text import transcribe_audio_with_timestamps
            from processor.nlp_engine import get_entities_and_nouns

            self.status.emit("Extracting audio & detecting language...")
            audio_path = extract_audio(self.video_path)

            self.status.emit("Transcribing speech...")
            segments, language = transcribe_audio_with_timestamps(audio_path)
            self.detected_language = language

            self.status.emit(f"Analyzing {language.upper()} entities (NER/POS)...")
            for seg in segments:
                seg['entities'] = get_entities_and_nouns(seg['text'])
                seg['selected_wiki'] = None 
                seg['screenshot_path'] = None
                seg['language'] = language
                seg['candidates'] = []

            from processor.nlp_engine import unload_nlp_model
            from processor.speech_to_text import unload_whisper_model
            unload_nlp_model()
            unload_whisper_model()

            self.finished.emit(segments)
        except Exception as e:
            self.error.emit(str(e))

class SearchWorker(QThread):
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, segment_text, entity_name, language):
        super().__init__()
        self.segment_text = segment_text
        self.entity_name = entity_name
        self.language = language

    def run(self):
        try:
            from processor.retrieval_engine import agentic_search
            candidates = agentic_search(self.segment_text, self.entity_name, search_type="all", language=self.language)
            self.finished.emit(candidates)
        except Exception as e:
            self.error.emit(str(e))

class RenderWorker(QThread):
    finished = Signal(str)
    error = Signal(str)
    status = Signal(str)

    def __init__(self, video_path, render_plan):
        super().__init__()
        self.video_path = video_path
        self.render_plan = render_plan

    def run(self):
        try:
            from processor.overlay_engine import render_with_screenshots
            output = render_with_screenshots(self.video_path, self.render_plan)
            self.finished.emit(output)
        except Exception as e:
            self.error.emit(str(e))

class EditorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Video Knowledge Editor")
        self.resize(1200, 800)
        self.segments = []
        self.video_path = ""
        self.current_seg_index = -1

        self.init_ui()

    def init_ui(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1e1e1e; color: #ffffff; font-family: 'Segoe UI'; }
            QListWidget { background-color: #2d2d2d; border: 1px solid #3d3d3d; padding: 5px; }
            QListWidget::item { padding: 10px; border-bottom: 1px solid #3d3d3d; }
            QListWidget::item:selected { background-color: #007acc; }
            QPushButton { background-color: #007acc; border-radius: 4px; padding: 8px 15px; font-weight: bold; border: none; }
            QPushButton:hover { background-color: #1c97ea; }
            QPushButton:disabled { background-color: #444; color: #888; }
            QLabel#Header { font-size: 18px; font-weight: bold; color: #007acc; }
            QTextEdit { background-color: #2d2d2d; border: 1px solid #3d3d3d; color: #eee; }
        """)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.start_page = QWidget()
        start_layout = QVBoxLayout(self.start_page)
        btn_upload = QPushButton("UPLOAD VIDEO TO BEGIN")
        btn_upload.setFixedSize(300, 60)
        btn_upload.clicked.connect(self.upload_video)
        start_layout.addWidget(btn_upload, 0, Qt.AlignCenter)
        self.stack.addWidget(self.start_page)

        self.loading_page = QWidget()
        loading_layout = QVBoxLayout(self.loading_page)
        self.load_status = QLabel("Ready")
        self.load_status.setAlignment(Qt.AlignCenter)
        self.load_status.setStyleSheet("font-size: 20px; font-weight: bold;")
        loading_layout.addWidget(self.load_status)
        self.stack.addWidget(self.loading_page)

        self.editor_page = QWidget()
        editor_layout = QHBoxLayout(self.editor_page)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("Timeline Segments"))
        self.seg_list = QListWidget()
        self.seg_list.itemClicked.connect(self.on_segment_selected)
        left_layout.addWidget(self.seg_list)
        editor_layout.addWidget(left_panel, 1)

        mid_panel = QWidget()
        mid_layout = QVBoxLayout(mid_panel)
        self.seg_text_display = QTextEdit()
        self.seg_text_display.setReadOnly(True)
        self.seg_text_display.setMaximumHeight(100)
        
        self.ent_list = QListWidget()
        self.ent_list.itemClicked.connect(self.on_entity_selected)
        
        self.wiki_list = QListWidget()
        self.wiki_list.itemClicked.connect(self.on_article_selected)

        mid_layout.addWidget(QLabel("Segment Text"))
        mid_layout.addWidget(self.seg_text_display)
        mid_layout.addWidget(QLabel("Detected Entities (NER/POS)"))
        mid_layout.addWidget(self.ent_list)
        mid_layout.addWidget(QLabel("Wikipedia Articles (Choose one)"))
        mid_layout.addWidget(self.wiki_list)
        editor_layout.addWidget(mid_panel, 1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.preview_label = QLabel("No Selection\n\nPick a Wiki article to see the overlay card")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet("border: 2px dashed #444; border-radius: 10px; padding: 20px;")
        
        self.btn_render = QPushButton("FINALIZE & RENDER VIDEO")
        self.btn_render.setFixedHeight(50)
        self.btn_render.clicked.connect(self.start_render)
        
        right_layout.addWidget(QLabel("Knowledge Card Preview"))
        right_layout.addWidget(self.preview_label, 5)
        right_layout.addWidget(self.btn_render, 1)
        editor_layout.addWidget(right_panel, 1)

        self.stack.addWidget(self.editor_page)

    def upload_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.mov *.avi)")
        if file_path:
            self.video_path = file_path
            self.stack.setCurrentIndex(1)
            self.worker = AnalysisWorker(file_path)
            self.worker.status.connect(self.load_status.setText)
            self.worker.finished.connect(self.on_analysis_complete)
            self.worker.error.connect(self.on_error)
            self.worker.start()

    def on_analysis_complete(self, segments):
        self.segments = segments
        self.seg_list.clear()
        for i, seg in enumerate(segments):
            item = QListWidgetItem(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text'][:35]}...")
            self.seg_list.addItem(item)
        self.stack.setCurrentIndex(2)

    def on_segment_selected(self, item):
        self.current_seg_index = self.seg_list.currentRow()
        seg = self.segments[self.current_seg_index]
        self.seg_text_display.setText(seg['text'])
        
        self.ent_list.clear()
        for ent in seg['entities']:
            self.ent_list.addItem(f"{ent['text']} ({ent['label']})")
        
        self.wiki_list.clear()
        if seg['selected_wiki']:
             self.wiki_list.addItem(f"SELECTED: {seg['selected_wiki']}")
             if seg['screenshot_path']:
                 self.update_preview(seg['screenshot_path'])
        else:
             self.preview_label.setText("Select an entity and a Wiki article to preview the card.")
             self.preview_label.setPixmap(QPixmap())
        
    def on_entity_selected(self, item):
        entity_name = item.text().split(" (")[0]
        seg = self.segments[self.current_seg_index]
        
        self.preview_label.setText(f"AI is searching Wiki and News for '{entity_name}'...\n(Loading semantic models if first time)")
        self.wiki_list.clear()
        self.wiki_list.addItem("Searching...")
        
        self.search_worker = SearchWorker(seg['text'], entity_name, seg.get('language', 'en'))
        self.search_worker.finished.connect(self.on_search_finished)
        self.search_worker.error.connect(self.on_error)
        self.search_worker.start()

    def on_search_finished(self, candidates):
        self.wiki_list.clear()
        if not candidates:
            self.wiki_list.addItem("No articles found")
            self.preview_label.setText("No candidates found for this entity.")
            return

        for cand in candidates:
            list_item = QListWidgetItem(cand['title'])
            list_item.setData(Qt.UserRole, cand['url'])
            self.wiki_list.addItem(list_item)
        
        self.preview_label.setText(f"Found {len(candidates)} candidates. Select one to preview the card.")


    def on_article_selected(self, item):
        title = item.text()
        url = item.data(Qt.UserRole)
        if title == "No articles found" or title.startswith("SELECTED:"):
             return
             
        seg = self.segments[self.current_seg_index]
        seg['selected_wiki'] = title
        
        self.preview_label.setText(f"Capturing screenshot from {title}...")
        QApplication.processEvents()
        
        from processor.screenshot_engine import capture_article_screenshot
        path = capture_article_screenshot(url, f"seg_{self.current_seg_index}")
        seg['screenshot_path'] = path
        if path:
            self.update_preview(path)
        else:
            self.preview_label.setText("Failed to capture. Check connection.")

    def update_preview(self, path):
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self.preview_label.setPixmap(pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.preview_label.setText("Image data invalid.")

    def start_render(self):
        render_plan = [seg for seg in self.segments if seg['screenshot_path']]
        
        if not render_plan:
            QMessageBox.warning(self, "No Selections", "Please select at least one Wikipedia article to overlay.")
            return

        self.stack.setCurrentIndex(1)
        self.load_status.setText("RENDERING INTELLIGENCE LAYER...\nPlease wait, encoding video.")
        
        self.render_worker = RenderWorker(self.video_path, render_plan)
        self.render_worker.finished.connect(self.on_render_finished)
        self.render_worker.error.connect(self.on_error)
        self.render_worker.start()

    def on_render_finished(self, output):
        QMessageBox.information(self, "Success", f"Professional knowledge video generated!\n\nSaved to: {output}")
        self.stack.setCurrentIndex(0)

    def on_error(self, message):
        self.stack.setCurrentIndex(2)
        QMessageBox.critical(self, "System Error", f"An operation failed:\n{message}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EditorApp()
    window.show()
    sys.exit(app.exec())
