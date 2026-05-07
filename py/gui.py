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
    finished = Signal(list, dict) # segments, global_stats
    error = Signal(str)
    status = Signal(str)
    log = Signal(str)
    progress = Signal(int)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirector = StreamRedirector(self.log, self.progress)
        sys.stdout = redirector
        sys.stderr = redirector

        try:
            from processor.video_processor import extract_audio
            from processor.speech_to_text import transcribe_audio_with_timestamps
            from processor.nlp_engine import get_entities_and_nouns
            from processor.translation_engine import translate_text

            self.status.emit("Extracting audio & detecting language...")
            audio_path = extract_audio(self.video_path)

            self.status.emit("Transcribing speech...")
            print(f"[DEBUG] GUI Worker: calling transcribe_audio_with_timestamps...")
            segments, language = transcribe_audio_with_timestamps(audio_path, video_path=self.video_path)
            print(f"[DEBUG] GUI Worker: Transcription complete. Detected: {language}")
            self.detected_language = language

            INDIAN_LANGS = ['hi', 'mr', 'ta', 'te', 'kn', 'ml', 'bn', 'gu', 'pa', 'as', 'or']
            is_indian = language in INDIAN_LANGS

            # Batch Translate Segment Texts (to avoid rate limits)
            texts_to_translate = [seg['text'] for seg in segments]
            translated_all = []
            
            if is_indian and texts_to_translate:
                # We do it in chunks of 5 sentences to keep text lengths manageable but reduce calls by 5x
                chunk_size = 5
                for i in range(0, len(texts_to_translate), chunk_size):
                    batch = texts_to_translate[i : i+chunk_size]
                    joined = " ###SEP### ".join(batch)
                    translated_batch_raw = translate_text(joined, language, "en")
                    translated_batch = translated_batch_raw.split(" ###SEP### ")
                    # Pad if split failed to return enough parts
                    while len(translated_batch) < len(batch):
                        translated_batch.append(batch[len(translated_batch)])
                    translated_all.extend(translated_batch)
            else:
                translated_all = texts_to_translate

            total_segs = len(segments)
            for i, seg in enumerate(segments):
                self.progress.emit(int((i / max(1, total_segs)) * 100))
                
                original_text = seg['text']
                translated_text = translated_all[i] if i < len(translated_all) else original_text
                
                if is_indian and translated_text != original_text:
                    seg['translated_text'] = translated_text
                    raw_entities = get_entities_and_nouns(translated_text)
                    
                    # Batch translate entities for this segment
                    if raw_entities:
                        ent_names = [e['text'] for e in raw_entities]
                        joined_ents = " ||| ".join(ent_names)
                        local_names_raw = translate_text(joined_ents, "en", language)
                        local_names = local_names_raw.split(" ||| ")
                        
                        dual_entities = []
                        for j, ent in enumerate(raw_entities):
                            l_name = local_names[j].strip() if j < len(local_names) else ent['text']
                            
                            # Add EN entity
                            dual_entities.append({
                                "text": ent['text'],
                                "display_text": f"[EN] {ent['text']}",
                                "language": "en",
                                "label": ent['label']
                            })
                            
                            # Add Local entity if different
                            if l_name != ent['text']:
                                dual_entities.append({
                                    "text": l_name,
                                    "display_text": f"[{language.upper()}] {l_name}",
                                    "language": language,
                                    "label": ent['label']
                                })
                        seg['entities'] = dual_entities
                    else:
                        seg['entities'] = []
                else:
                    # Non-indian or translation failed: just English
                    raw_entities = get_entities_and_nouns(original_text)
                    seg['entities'] = [{
                        "text": e['text'],
                        "display_text": f"[EN] {e['text']}" if is_indian else e['text'],
                        "language": language if not is_indian else "en",
                        "label": e['label']
                    } for e in raw_entities]

                seg['selected_wiki'] = None
                seg['selected_wiki_url'] = None
                seg['y_offset'] = 0
                seg['screenshot_path'] = None
                seg['language'] = language
                seg['candidates'] = []

            # --- FEATURE: GLOBAL ENTITY INTELLIGENCE ---
            from processor.nlp_engine import build_global_entity_stats, compute_global_scores, get_sliding_context, rank_entities_for_segment

            self.status.emit("Building global entity intelligence...")
            global_stats = build_global_entity_stats(segments)
            global_stats = compute_global_scores(global_stats)

            self.status.emit("Ranking entities with context...")
            for i, seg in enumerate(segments):
                context_text = get_sliding_context(segments, i)
                local_entities = seg.get('entities', []) 
                
                # Rank entities using global importance and sliding window context
                final_ranked = rank_entities_for_segment(seg.get('translated_text', seg['text']), local_entities, global_stats, context_text)
                seg['final_entities'] = final_ranked
            # ---------------------------------------------

            self.progress.emit(100)

            from processor.nlp_engine import unload_nlp_model
            from processor.speech_to_text import unload_whisper_model
            unload_nlp_model()
            unload_whisper_model()

            self.finished.emit(segments, global_stats)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

class SearchWorker(QThread):
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, segment_text, entity_name, language, context_text=None, global_entity_scores=None):
        super().__init__()
        self.segment_text = segment_text
        self.entity_name = entity_name
        self.language = language
        self.context_text = context_text
        self.global_entity_scores = global_entity_scores

    def run(self):
        try:
            from processor.retrieval_engine import agentic_search
            candidates = agentic_search(
                self.segment_text, 
                self.entity_name, 
                search_type="all", 
                language=self.language,
                context_text=self.context_text,
                global_entity_scores=self.global_entity_scores
            )
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
            from processor.nlp_engine import unload_nlp_model
            from processor.speech_to_text import unload_whisper_model
            from processor.retrieval_engine import unload_search_model
            unload_nlp_model()
            unload_whisper_model()
            unload_search_model()

            from processor.overlay_engine import render_with_screenshots
            output = render_with_screenshots(self.video_path, self.render_plan)
            self.finished.emit(output)
        except Exception as e:
            self.error.emit(str(e))

class TitleBar(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setFixedHeight(60)
        
        # Explicitly style the title bar container
        self.setObjectName("TitleBar")
        self.setStyleSheet("""
            QWidget#TitleBar {
                background-color: #FFFFFF;
                border-bottom: 2px solid #000000;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
            }
        """)

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(20, 0, 20, 0)
        self.layout.setSpacing(12)

        # Back button
        self.back_btn = QPushButton("← BACK")
        self.back_btn.setFixedSize(85, 32)
        self.back_btn.setCursor(Qt.PointingHandCursor)
        self.back_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #000000;
                font-size: 13px;
                font-weight: 900;
                border: none;
            }
            QPushButton:hover {
                text-decoration: underline;
            }
            QPushButton:disabled {
                color: #CCCCCC;
            }
        """)
        self.back_btn.clicked.connect(self.on_back_clicked)
        self.layout.addWidget(self.back_btn)

        self.title_label = QLabel("VIDEO KNOWLEDGE EDITOR")
        self.title_label.setStyleSheet("""
            font-weight: 900; 
            font-size: 14px; 
            color: #000000; 
            letter-spacing: 1.5px;
            background: transparent;
            border: none;
        """)
        self.layout.addWidget(self.title_label)

        self.layout.addStretch()

        # Window controls
        self.btn_min = QPushButton("—")
        self.btn_max = QPushButton("□")
        self.btn_close = QPushButton("✕")

        for btn in [self.btn_min, self.btn_max, self.btn_close]:
            btn.setFixedSize(45, 45)
            btn.setCursor(Qt.PointingHandCursor)
            
        # Specific styling for close button to ensure it stands out
        self.btn_close.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #000000;
                border: none;
                font-size: 24px;
                font-weight: 900;
            }
            QPushButton:hover {
                background-color: #000000;
                color: #FFFFFF;
            }
        """)
        
        # Consistent styling for other controls
        control_style = """
            QPushButton {
                background-color: transparent;
                color: #000000;
                border: none;
                font-size: 20px;
                font-weight: 900;
            }
            QPushButton:hover {
                background-color: #000000;
                color: #FFFFFF;
            }
        """
        self.btn_min.setStyleSheet(control_style)
        self.btn_max.setStyleSheet(control_style)

        self.btn_min.clicked.connect(self.parent.showMinimized)
        self.btn_max.clicked.connect(self.toggle_maximize)
        self.btn_close.clicked.connect(self.parent.close)

        self.layout.addWidget(self.btn_min)
        self.layout.addWidget(self.btn_max)
        self.layout.addWidget(self.btn_close)

        self.startPos = None
        self.update_back_visibility()

    def toggle_maximize(self):
        if self.parent.isMaximized():
            self.parent.showNormal()
            self.btn_max.setText("▢")
        else:
            self.parent.showMaximized()
            self.btn_max.setText("❐")

    def on_back_clicked(self):
        curr = self.parent.stack.currentIndex()
        target = -1
        
        # Stop any running workers before going back
        if curr == 2: # Loading page
            self.parent.stop_workers()
            target = 1
        elif curr == 1: # Start page
            target = 0
        elif curr == 3: # Editor page
            target = 1
            
        if target != -1:
            self.parent.fade_to_page(target)
            self.parent.stack.setCurrentIndex(target)

    def update_back_visibility(self):
        # Instead of hiding, we disable it on the home page for better B&W visibility
        self.back_btn.setEnabled(self.parent.stack.currentIndex() > 0)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.startPos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self.startPos:
            delta = event.globalPosition().toPoint() - self.startPos
            self.parent.move(self.parent.pos() + delta)
            self.startPos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        self.startPos = None

class EditorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowTitle("Video Knowledge Editor")
        self.resize(1280, 720)  # Slightly larger default for modern screens
        self.segments = []
        self.video_path = ""
        self.current_seg_index = -1

        self.init_ui()
        track("app_started")

    def stop_workers(self):
        """Safely terminate any running background threads."""
        for attr in ['worker', 'render_worker', 'search_worker']:
            if hasattr(self, attr):
                w = getattr(self, attr)
                if w and w.isRunning():
                    w.terminate()
                    w.wait()

    def closeEvent(self, event):
        """Clean up threads before closing."""
        self.stop_workers()
        event.accept()

    def init_ui(self):
        self.setStyleSheet("""
            QMainWindow, QStackedWidget {
                background-color: #FFFFFF;
            }
            QWidget {
                background-color: #FFFFFF;
                color: #000000;
                font-family: -apple-system, 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif;
                font-size: 13px;
            }
            QListWidget {
                background-color: #FAFAFA;
                border: 1px solid #E0E0E0;
                border-radius: 12px;
                padding: 4px 0;
                outline: none;
            }
            QListWidget::item {
                padding: 10px 14px;
                border-bottom: 1px solid #F0F0F0;
                color: #000000;
            }
            QListWidget::item:last-child { border-bottom: none; }
            QListWidget::item:hover { background-color: #F2F2F2; }
            QListWidget::item:selected {
                background-color: #E8E8E8;
                color: #000000;
                border-radius: 8px;
            }
            QPushButton {
                background-color: #000000;
                color: #FFFFFF;
                border: none;
                padding: 9px 20px;
                border-radius: 10px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #222222; }
            QPushButton:pressed { background-color: #444444; }
            QPushButton:disabled { background-color: #C0C0C0; color: #FFFFFF; }
            QLabel {
                color: #000000;
                font-size: 13px;
                background-color: transparent;
            }
            QTextEdit, QPlainTextEdit {
                background-color: #FAFAFA;
                color: #000000;
                border: 1px solid #E0E0E0;
                border-radius: 10px;
                padding: 8px;
                selection-background-color: #000000;
                selection-color: #FFFFFF;
            }
            QLineEdit {
                background-color: #FAFAFA;
                color: #000000;
                border: 1px solid #E0E0E0;
                border-radius: 10px;
                padding: 7px 12px;
                selection-background-color: #000000;
                selection-color: #FFFFFF;
            }
            QLineEdit:focus { border: 1.5px solid #000000; }
            QSpinBox {
                background-color: #FAFAFA;
                color: #000000;
                border: 1px solid #E0E0E0;
                border-radius: 10px;
                padding: 6px 10px;
            }
            QProgressBar {
                border: none;
                border-radius: 6px;
                background-color: #EBEBEB;
                text-align: center;
                color: transparent;
            }
            QProgressBar::chunk {
                background-color: #000000;
                border-radius: 6px;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 8px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #C0C0C0;
                border-radius: 4px;
                min-height: 30px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

        self.main_container = QWidget()
        self.main_container.setObjectName("MainContainer")
        self.main_container.setStyleSheet("""
            QWidget#MainContainer {
                border: 2px solid #000000;
                border-radius: 12px;
                background-color: #FFFFFF;
            }
        """)
        self.main_layout = QVBoxLayout(self.main_container)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.stack = QStackedWidget()

        self.title_bar = TitleBar(self)
        self.main_layout.addWidget(self.title_bar)
        self.main_layout.addWidget(self.stack)

        # Subtle size grip for resizing
        grip_layout = QHBoxLayout()
        grip_layout.setContentsMargins(0, 0, 2, 2)
        grip_layout.addStretch()
        self.size_grip = QSizeGrip(self)
        self.size_grip.setFixedSize(14, 14)
        self.size_grip.setStyleSheet("background: transparent;")
        grip_layout.addWidget(self.size_grip)
        self.main_layout.addLayout(grip_layout)

        self.setCentralWidget(self.main_container)
        self.stack.currentChanged.connect(self.title_bar.update_back_visibility)


        self.selection_page = QWidget()
        selection_layout = QVBoxLayout(self.selection_page)
        selection_layout.addStretch(1)

        sel_label = QLabel("Video Knowledge Editor")
        sel_label.setAlignment(Qt.AlignCenter)
        sel_label.setStyleSheet("""
            font-size: 28px;
            font-weight: 700;
            color: #000000;
            letter-spacing: -0.5px;
            margin-bottom: 8px;
        """)
        selection_layout.addWidget(sel_label)

        sub_label = QLabel("Choose a processing mode to get started")
        sub_label.setAlignment(Qt.AlignCenter)
        sub_label.setStyleSheet("""
            font-size: 15px;
            color: #666666;
            margin-bottom: 40px;
        """)
        selection_layout.addWidget(sub_label)

        btn_en = QPushButton("English Mode")
        btn_en.setFixedSize(340, 52)
        btn_en.setToolTip("Optimized for English — faster, smaller models")
        btn_en.clicked.connect(lambda: self.select_mode("english"))
        selection_layout.addWidget(btn_en, 0, Qt.AlignCenter)

        selection_layout.addSpacing(12)

        btn_multi = QPushButton("Multilingual Mode")
        btn_multi.setFixedSize(340, 52)
        btn_multi.setToolTip("Supports Hindi, Tamil, Telugu and more")
        btn_multi.setStyleSheet("""
            QPushButton {
                background-color: #FFFFFF;
                color: #000000;
                border: 1.5px solid #000000;
                padding: 9px 20px;
                border-radius: 10px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #F0F0F0; }
        """)
        btn_multi.clicked.connect(lambda: self.select_mode("multilingual"))
        selection_layout.addWidget(btn_multi, 0, Qt.AlignCenter)

        selection_layout.addStretch(1)
        self.stack.addWidget(self.selection_page)

        self.start_page = QWidget()
        start_layout = QVBoxLayout(self.start_page)
        start_layout.addStretch(1)
        start_title = QLabel("Open a Video File")
        start_title.setAlignment(Qt.AlignCenter)
        start_title.setStyleSheet("font-size: 22px; font-weight: 700; color: #000000; margin-bottom: 8px;")
        start_layout.addWidget(start_title)
        start_sub = QLabel("Select an MP4, MOV or AVI file to begin analysis")
        start_sub.setAlignment(Qt.AlignCenter)
        start_sub.setStyleSheet("font-size: 14px; color: #666666; margin-bottom: 32px;")
        start_layout.addWidget(start_sub)
        btn_upload = QPushButton("Choose Video...")
        btn_upload.setFixedSize(220, 44)
        btn_upload.clicked.connect(self.upload_video)
        start_layout.addWidget(btn_upload, 0, Qt.AlignCenter)
        start_layout.addStretch(1)
        self.stack.addWidget(self.start_page)

        self.loading_page = QWidget()
        loading_layout = QVBoxLayout(self.loading_page)
        loading_layout.addStretch(1)

        self.load_status = QLabel("Ready")
        self.load_status.setAlignment(Qt.AlignCenter)
        self.load_status.setStyleSheet("""
            font-size: 18px;
            font-weight: 600;
            color: #000000;
            margin-bottom: 12px;
        """)
        loading_layout.addWidget(self.load_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        loading_layout.addWidget(self.progress_bar)

        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("""
            background-color: #FFFFFF;
            color: #111111;
            font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
            font-size: 11px;
            border: 1px solid #E0E0E0;
            border-radius: 12px;
            padding: 12px;
            margin-top: 16px;
        """)
        self.log_console.setMinimumHeight(260)
        loading_layout.addWidget(self.log_console)

        loading_layout.addStretch(1)
        self.stack.addWidget(self.loading_page)

        self.editor_page = QWidget()
        editor_layout = QHBoxLayout(self.editor_page)

        left_panel = QWidget()
        left_panel.setStyleSheet("background-color: #FFFFFF;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 12, 8, 12)
        left_layout.setSpacing(8)
        tl_label = QLabel("Timeline")
        tl_label.setStyleSheet("font-size: 11px; font-weight: 600; color: #666666; letter-spacing: 0.5px; text-transform: uppercase; background: transparent;")
        left_layout.addWidget(tl_label)
        self.seg_list = QListWidget()
        self.seg_list.itemClicked.connect(self.on_segment_selected)
        left_layout.addWidget(self.seg_list)
        editor_layout.addWidget(left_panel, 1)

        mid_panel = QWidget()
        mid_panel.setStyleSheet("background-color: #FFFFFF;")
        mid_layout = QVBoxLayout(mid_panel)
        mid_layout.setContentsMargins(8, 12, 8, 12)
        mid_layout.setSpacing(8)

        self.seg_text_display = QTextEdit()
        self.seg_text_display.setReadOnly(True)
        self.seg_text_display.setMaximumHeight(90)

        self.ent_list = QListWidget()
        self.ent_list.itemClicked.connect(self.on_entity_selected)

        self.wiki_list = QListWidget()
        self.wiki_list.itemClicked.connect(self.on_article_selected)

        def section_label(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #666666; letter-spacing: 0.5px; background: transparent;")
            return lbl

        mid_layout.addWidget(section_label("Segment Text"))
        mid_layout.addWidget(self.seg_text_display)
        mid_layout.addWidget(section_label("Detected Entities"))
        mid_layout.addWidget(self.ent_list)
        mid_layout.addWidget(section_label("Articles"))
        mid_layout.addWidget(self.wiki_list)

        url_layout = QHBoxLayout()
        url_layout.setSpacing(8)
        self.custom_url_input = QLineEdit()
        self.custom_url_input.setPlaceholderText("Paste any article URL...")
        self.btn_use_url = QPushButton("Capture")
        self.btn_use_url.setFixedWidth(90)
        self.btn_use_url.clicked.connect(self.on_custom_url_submitted)
        url_layout.addWidget(self.custom_url_input)
        url_layout.addWidget(self.btn_use_url)
        mid_layout.addLayout(url_layout)

        editor_layout.addWidget(mid_panel, 1)

        right_panel = QWidget()
        right_panel.setStyleSheet("background-color: #FFFFFF;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 12, 8, 12)
        right_layout.setSpacing(8)

        preview_header = QLabel("Knowledge Card")
        preview_header.setStyleSheet("font-size: 11px; font-weight: 600; color: #666666; letter-spacing: 0.5px; background: transparent;")
        right_layout.addWidget(preview_header)

        self.preview_label = QLabel("No Selection\n\nSelect an article to preview the overlay card")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet("""
            background-color: #FAFAFA;
            border: 1.5px solid #E0E0E0;
            border-radius: 12px;
            padding: 24px;
            color: #666666;
            font-size: 13px;
        """)

        scroll_group = QWidget()
        scroll_group.setStyleSheet("background: transparent;")
        scroll_layout = QHBoxLayout(scroll_group)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(8)
        self.y_offset_input = QSpinBox()
        self.y_offset_input.setRange(0, 10000)
        self.y_offset_input.setSingleStep(100)
        self.y_offset_input.setPrefix("Offset: ")
        self.y_offset_input.setFixedHeight(36)

        self.btn_refresh_scroll = QPushButton("Refresh")
        self.btn_refresh_scroll.setFixedHeight(36)
        self.btn_refresh_scroll.setStyleSheet("""
            QPushButton {
                background-color: #FFFFFF;
                color: #000000;
                border: 1.5px solid #000000;
                border-radius: 10px;
                font-weight: 600;
                padding: 0 16px;
            }
            QPushButton:hover { background-color: #F0F0F0; }
        """)
        self.btn_refresh_scroll.clicked.connect(self.on_refresh_with_scroll)
        scroll_layout.addWidget(self.y_offset_input)
        scroll_layout.addWidget(self.btn_refresh_scroll)

        self.btn_render = QPushButton("Render Video")
        self.btn_render.setFixedHeight(44)
        self.btn_render.clicked.connect(self.start_render)

        right_layout.addWidget(self.preview_label, 5)
        right_layout.addWidget(scroll_group)
        right_layout.addWidget(self.btn_render)
        editor_layout.addWidget(right_panel, 1)

        self.stack.addWidget(self.editor_page)

    def select_mode(self, mode):
        from processor.config import set_model_mode
        set_model_mode(mode)
        # Fade transition to the next page for a smoother experience
        self.fade_to_page(1)
        self.stack.setCurrentIndex(1)

    def upload_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.mov *.avi)")
        if file_path:
            self.video_path = file_path
            # Fade transition to processing page
            self.fade_to_page(2)
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

        self.log_console.verticalScrollBar().setValue(self.log_console.verticalScrollBar().maximum())

    def on_analysis_complete(self, segments, global_stats):
        self.segments = segments
        self.global_stats = global_stats # Store global stats for later search boosting
        self.update_segment_list()
        self.stack.setCurrentIndex(3)
        track("analysis_complete", {"segments_count": len(segments)})

    def update_segment_list(self):
        self.seg_list.clear()
        for i, seg in enumerate(self.segments):
            start_fmt = format_seconds_to_min_sec(seg['start'])
            # Use simple colored bullet markers instead of emojis
            if seg.get('screenshot_path'):
                marker = "●"  # green bullet for captured screenshots
                color = "#4caf50"
            elif seg.get('entities'):
                marker = "●"  # orange bullet for detected entities
                color = "#ff9800"
            else:
                marker = "●"  # gray bullet for plain segments
                color = "#9e9e9e"
            item = QListWidgetItem(f"{marker} [{start_fmt}] {seg['text'][:35]}...")
            # Apply color via HTML formatting
            item.setForeground(QColor(color))
            self.seg_list.addItem(item)

    def on_segment_selected(self, item):
        self.current_seg_index = self.seg_list.currentRow()
        seg = self.segments[self.current_seg_index]
        if seg.get('translated_text'):
            display_text = f"{seg['text']}\n\n[Translation]: {seg['translated_text']}"
        else:
            display_text = seg['text']
        self.seg_text_display.setText(display_text)

        self.ent_list.clear()
        for ent in seg['entities']:
            display_ent = ent.get('display_text', ent['text'])
            self.ent_list.addItem(f"{display_ent} - {ent['label']}")

        self.wiki_list.clear()
        if seg.get('selected_wiki'):
             self.wiki_list.addItem(f"SELECTED: {seg['selected_wiki']}")
             self.custom_url_input.setText(seg.get('selected_wiki_url', ''))

             if seg.get('screenshot_path'):
                 self.update_preview(seg['screenshot_path'])
        else:
             self.custom_url_input.clear()

             self.preview_label.setText("Select an entity and a Wiki article to preview the card.")
             self.preview_label.setPixmap(QPixmap())

    def on_entity_selected(self, item):
        row = self.ent_list.currentRow()
        seg = self.segments[self.current_seg_index]
        ent = seg['entities'][row]
        
        entity_name = ent['text']
        search_language = ent.get('language', seg.get('language', 'en'))
        
        self.preview_label.setText(f"AI is searching {search_language.upper()} Wiki for '{entity_name}'...")
        self.wiki_list.clear()
        self.wiki_list.addItem("Searching...")

        search_text = seg.get('translated_text', seg['text'])
        
        # FEATURE 1: PREPARE CONTEXT & GLOBAL SCORES
        from processor.nlp_engine import get_sliding_context
        context_text = get_sliding_context(self.segments, self.current_seg_index)
        global_scores = {k: v['score'] for k, v in self.global_stats.items()} if hasattr(self, 'global_stats') else None

        if hasattr(self, 'search_worker') and self.search_worker.isRunning():
            self.search_worker.terminate()
            self.search_worker.wait()

        self.search_worker = SearchWorker(search_text, entity_name, search_language,
                                          context_text=context_text,
                                          global_entity_scores=global_scores)
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

        title = url.split("//")[-1].split("/")[0]
        seg = self.segments[self.current_seg_index]
        seg['selected_wiki'] = f"[Custom] {title}"
        seg['selected_wiki_url'] = url

        self.capture_and_preview(url, seg['selected_wiki'])

    def capture_and_preview(self, url, title):
        seg = self.segments[self.current_seg_index]
        y_offset = self.y_offset_input.value()
        seg['y_offset'] = y_offset

        self.preview_label.setText(f"Capturing screenshot from {title}...")
        QApplication.processEvents()

        # Unload heavy models to free up RAM for Chromium (prevent OOM crash)
        from processor.nlp_engine import unload_nlp_model
        from processor.speech_to_text import unload_whisper_model
        from processor.retrieval_engine import unload_search_model
        unload_nlp_model()
        unload_whisper_model()
        unload_search_model()

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

        # Fade transition to rendering page
        self.fade_to_page(2)
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
        # Return to the editor page with a fade effect on error
        self.fade_to_page(3)
        self.stack.setCurrentIndex(3)
        QMessageBox.critical(self, "System Error", f"An operation failed:\n{message}")

    def fade_to_page(self, index):
        """Fade animation when switching pages in the QStackedWidget."""
        from PySide6.QtCore import QPropertyAnimation, QEasingCurve
        current = self.stack.currentWidget()
        next_widget = self.stack.widget(index)
        if not current or not next_widget:
            return
        fade_out = QPropertyAnimation(current, b"windowOpacity")
        fade_out.setDuration(200)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)
        fade_out.setEasingCurve(QEasingCurve.InOutQuad)
        fade_in = QPropertyAnimation(next_widget, b"windowOpacity")
        fade_in.setDuration(200)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.setEasingCurve(QEasingCurve.InOutQuad)
        fade_out.finished.connect(fade_in.start)
        fade_out.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Enable high DPI scaling for crisp visuals
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    window = EditorApp()
    window.show()
    sys.exit(app.exec())
