from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PySide6.QtGui import *
import sys
import os
import csv
import re
import io
from processor.tracker_cloud import track

def format_seconds_to_min_sec(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

class StreamRedirector:
    def __init__(self, log_signal, progress_signal):
        self.log_signal = log_signal
        self.progress_signal = progress_signal
        self.buffer = ""

    def write(self, text):
        if not text: return
        self.log_signal.emit(text.strip())
        
        # Simple percentage parsing for tqdm/huggingface bars
        # Look for things like "100%" or "45%"
        matches = re.findall(r"(\d+)%", text)
        if matches:
            try:
                self.progress_signal.emit(int(matches[-1]))
            except: pass

    def flush(self):
        pass

    def isatty(self):
        return False

class AnalysisWorker(QThread):
    finished = Signal(list)
    error = Signal(str)
    status = Signal(str)
    log = Signal(str)
    progress = Signal(int)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        # Redirect stdout and stderr to capture library logs and tqdm bars
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirector = StreamRedirector(self.log, self.progress)
        sys.stdout = redirector
        sys.stderr = redirector

        try:
            from processor.video_processor import extract_audio
            from processor.speech_to_text import transcribe_audio_with_timestamps
            from processor.nlp_engine import get_entities_and_nouns

            self.status.emit("Extracting audio & detecting language...")
            audio_path = extract_audio(self.video_path)

            self.status.emit("Transcribing speech...")
            segments, language = transcribe_audio_with_timestamps(audio_path)
            self.detected_language = language

            total_segs = len(segments)
            self.status.emit(f"Analyzing {language.upper()} entities...")
            for i, seg in enumerate(segments):
                self.progress.emit(int((i / total_segs) * 100))
                seg['entities'] = get_entities_and_nouns(seg['text'])
                seg['selected_wiki'] = None 
                seg['selected_wiki_url'] = None
                seg['y_offset'] = 0
                seg['screenshot_path'] = None
                seg['language'] = language
                seg['candidates'] = []

            self.progress.emit(100)

            from processor.nlp_engine import unload_nlp_model
            from processor.speech_to_text import unload_whisper_model
            unload_nlp_model()
            unload_whisper_model()

            self.finished.emit(segments)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

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
        track("app_started")

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

        # Selection Page
        self.selection_page = QWidget()
        selection_layout = QVBoxLayout(self.selection_page)
        selection_layout.addStretch(1)
        
        sel_label = QLabel("SELECT PROCESSING MODE")
        sel_label.setAlignment(Qt.AlignCenter)
        sel_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #007acc; margin-bottom: 20px;")
        selection_layout.addWidget(sel_label)

        sub_label = QLabel("This determines the AI models used for transcription and analysis")
        sub_label.setAlignment(Qt.AlignCenter)
        sub_label.setStyleSheet("font-size: 14px; color: #888; margin-bottom: 40px;")
        selection_layout.addWidget(sub_label)

        btn_en = QPushButton("ENGLISH MODE\n(Optimized for English, smaller models)")
        btn_en.setFixedSize(400, 80)
        btn_en.clicked.connect(lambda: self.select_mode("english"))
        selection_layout.addWidget(btn_en, 0, Qt.AlignCenter)
        
        selection_layout.addSpacing(20)

        btn_multi = QPushButton("MULTILINGUAL MODE\n(Supports Hindi, Tamil, etc., larger models)")
        btn_multi.setFixedSize(400, 80)
        btn_multi.clicked.connect(lambda: self.select_mode("multilingual"))
        selection_layout.addWidget(btn_multi, 0, Qt.AlignCenter)

        selection_layout.addStretch(1)
        self.stack.addWidget(self.selection_page)

        self.start_page = QWidget()
        start_layout = QVBoxLayout(self.start_page)
        btn_upload = QPushButton("UPLOAD VIDEO TO BEGIN")
        btn_upload.setFixedSize(300, 60)
        btn_upload.clicked.connect(self.upload_video)
        start_layout.addWidget(btn_upload, 0, Qt.AlignCenter)
        self.stack.addWidget(self.start_page)

        self.loading_page = QWidget()
        loading_layout = QVBoxLayout(self.loading_page)
        loading_layout.addStretch(1)
        
        self.load_status = QLabel("Ready")
        self.load_status.setAlignment(Qt.AlignCenter)
        self.load_status.setStyleSheet("font-size: 22px; font-weight: bold; color: #007acc; margin-bottom: 20px;")
        loading_layout.addWidget(self.load_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(30)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #3d3d3d;
                border-radius: 10px;
                text-align: center;
                background-color: #2d2d2d;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #007acc, stop:1 #1c97ea);
                border-radius: 8px;
            }
        """)
        loading_layout.addWidget(self.progress_bar)

        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("""
            background-color: #0c0c0c;
            color: #00ff41;
            font-family: 'Consolas', 'Courier New';
            font-size: 11px;
            border: 1px solid #3d3d3d;
            border-radius: 5px;
            margin-top: 20px;
        """)
        self.log_console.setMinimumHeight(250)
        loading_layout.addWidget(self.log_console)
        
        loading_layout.addStretch(1)
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
        
        url_layout = QHBoxLayout()
        self.custom_url_input = QLineEdit()
        self.custom_url_input.setPlaceholderText("Or paste any article URL here...")
        self.btn_use_url = QPushButton("CAPTURE URL")
        self.btn_use_url.clicked.connect(self.on_custom_url_submitted)
        url_layout.addWidget(self.custom_url_input)
        url_layout.addWidget(self.btn_use_url)
        mid_layout.addLayout(url_layout)
        
        editor_layout.addWidget(mid_panel, 1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.preview_label = QLabel("No Selection\n\nPick a Wiki article to see the overlay card")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet("border: 2px dashed #444; border-radius: 10px; padding: 20px;")
        
        scroll_group = QWidget()
        scroll_layout = QHBoxLayout(scroll_group)
        # scroll_layout.addWidget(QLabel("Scroll Offset (Y):"))
        # self.scroll_input = QSpinBox()
        # self.scroll_input.setRange(0, 10000)
        # self.scroll_input.setSingleStep(300)
        # self.scroll_input.setSuffix(" px")
        # self.btn_apply_scroll = QPushButton("APPLY SCROLL")
        # self.btn_apply_scroll.clicked.connect(self.on_refresh_with_scroll)
        # scroll_layout.addWidget(self.scroll_input)
        # scroll_layout.addWidget(self.btn_apply_scroll)
        
        self.btn_render = QPushButton("FINALIZE & RENDER VIDEO")
        self.btn_render.setFixedHeight(50)
        self.btn_render.clicked.connect(self.start_render)
        
        right_layout.addWidget(QLabel("Knowledge Card Preview"))
        right_layout.addWidget(self.preview_label, 5)
        right_layout.addWidget(scroll_group)
        right_layout.addWidget(self.btn_render, 1)
        editor_layout.addWidget(right_panel, 1)

        self.stack.addWidget(self.editor_page)

    def select_mode(self, mode):
        from processor.config import set_model_mode
        set_model_mode(mode)
        self.stack.setCurrentIndex(1)

    def upload_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.mov *.avi)")
        if file_path:
            self.video_path = file_path
            self.stack.setCurrentIndex(2)
            self.progress_bar.setValue(0)
            self.log_console.clear()
            self.load_status.setText("Initializing...")
            
            self.worker = AnalysisWorker(file_path)
            self.worker.status.connect(self.load_status.setText)
            self.worker.log.connect(self.append_log)
            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.finished.connect(self.on_analysis_complete)
            self.worker.error.connect(self.on_error)
            self.worker.start()
            track("video_uploaded", {"path": file_path})

    def append_log(self, text):
        self.log_console.appendPlainText(text)
        # Auto-scroll
        self.log_console.verticalScrollBar().setValue(self.log_console.verticalScrollBar().maximum())

    def on_analysis_complete(self, segments):
        self.segments = segments
        self.update_segment_list()
        self.stack.setCurrentIndex(3)
        track("analysis_complete", {"segments_count": len(segments)})

    def update_segment_list(self):
        self.seg_list.clear()
        for i, seg in enumerate(self.segments):
            start_fmt = format_seconds_to_min_sec(seg['start'])
            marker = "⚪"
            if seg.get('screenshot_path'):
                marker = "🟢"
            elif seg.get('entities'):
                marker = "🟠"
                
            item = QListWidgetItem(f"{marker} [{start_fmt}] {seg['text'][:35]}...")
            self.seg_list.addItem(item)

    def on_segment_selected(self, item):
        self.current_seg_index = self.seg_list.currentRow()
        seg = self.segments[self.current_seg_index]
        self.seg_text_display.setText(seg['text'])
        
        self.ent_list.clear()
        for ent in seg['entities']:
            self.ent_list.addItem(f"{ent['text']} ({ent['label']})")
        
        self.wiki_list.clear()
        if seg.get('selected_wiki'):
             self.wiki_list.addItem(f"SELECTED: {seg['selected_wiki']}")
             self.custom_url_input.setText(seg.get('selected_wiki_url', ''))
             # self.scroll_input.setValue(seg.get('y_offset', 0))
             if seg.get('screenshot_path'):
                 self.update_preview(seg['screenshot_path'])
        else:
             self.custom_url_input.clear()
             # self.scroll_input.setValue(0)
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
        if seg.get('selected_wiki') and seg['selected_wiki'] != title:
            track("overlay_overridden", {"old": seg['selected_wiki'], "new": title})
            
        seg['selected_wiki'] = title
        seg['selected_wiki_url'] = url
        self.custom_url_input.setText(url)
        
        self.capture_and_preview(url, title)

    def on_custom_url_submitted(self):
        url = self.custom_url_input.text().strip()
        if not url:
            return
            
        if self.current_seg_index == -1:
            QMessageBox.warning(self, "No Segment", "Please select a segment first.")
            return

        title = url.split("//")[-1].split("/")[0] # Simple title from domain
        seg = self.segments[self.current_seg_index]
        seg['selected_wiki'] = f"[Custom] {title}"
        seg['selected_wiki_url'] = url
        
        self.capture_and_preview(url, seg['selected_wiki'])

    def capture_and_preview(self, url, title):
        seg = self.segments[self.current_seg_index]
        y_offset = 0 # self.scroll_input.value()
        seg['y_offset'] = y_offset
        
        self.preview_label.setText(f"Capturing screenshot from {title}...")
        QApplication.processEvents()
        
        from processor.screenshot_engine import capture_article_screenshot
        path = capture_article_screenshot(url, f"seg_{self.current_seg_index}", y_offset=y_offset)
        seg['screenshot_path'] = path
        if path:
            self.update_preview(path)
            self.update_segment_list()
        else:
            self.preview_label.setText("Failed to capture. Check connection.")

    def on_refresh_with_scroll(self):
        if self.current_seg_index == -1: return
        seg = self.segments[self.current_seg_index]
        url = seg.get('selected_wiki_url')
        if not url:
            url = self.custom_url_input.text().strip()
        
        if url:
            self.capture_and_preview(url, seg.get('selected_wiki', 'Current Page'))

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

        self.stack.setCurrentIndex(2)
        self.load_status.setText("RENDERING INTELLIGENCE LAYER...\nPlease wait, encoding video.")
        
        self.render_worker = RenderWorker(self.video_path, render_plan)
        self.render_worker.finished.connect(self.on_render_finished)
        self.render_worker.error.connect(self.on_error)
        self.render_worker.start()
        track("render_started", {"overlays_count": len(render_plan)})

    def on_render_finished(self, output):
        csv_path = output.rsplit(".", 1)[0] + "_knowledge_links.csv"
        try:
            with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Start Time", "End Time", "Article Title", "URL"])
                for seg in self.segments:
                    if seg.get('screenshot_path'):
                        writer.writerow([
                            format_seconds_to_min_sec(seg['start']),
                            format_seconds_to_min_sec(seg['end']),
                            seg.get('selected_wiki', 'N/A'),
                            seg.get('selected_wiki_url', 'N/A')
                        ])
            msg_add = f"\n\nReference links saved to: {csv_path}"
        except Exception as e:
            msg_add = f"\n\n(Note: CSV export failed: {e})"

        QMessageBox.information(self, "Success", f"Professional knowledge video generated!\n\nSaved to: {output}{msg_add}")
        self.stack.setCurrentIndex(0)
        track("render_finished", {"output_path": output})

    def on_error(self, message):
        self.stack.setCurrentIndex(3)
        QMessageBox.critical(self, "System Error", f"An operation failed:\n{message}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EditorApp()
    window.show()
    sys.exit(app.exec())
