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
            QMainWindow, QWidget { background-color: #1e1e1e; color: #ffffff; }
            QListWidget { background-color: #252526; border: 1px solid #333; }
            QListWidget::item { padding: 10px; border-bottom: 1px solid #333; }
            QListWidget::item:selected { background-color: #094771; color: white; }
            QPushButton { background-color: #0e639c; color: white; padding: 8px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #1177bb; }
            QPushButton:disabled { background-color: #3a3d41; color: #888; }
            QLabel { font-size: 13px; }
            QTextEdit { background-color: #3c3c3c; color: white; border: 1px solid #555; }
        """)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)


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

        btn_multi = QPushButton("MULTILINGUAL MODE\n(Supports Hindi, Tamil, etc., smaller models)")
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
                border: 1px solid #555;
                border-radius: 10px;
                text-align: center;
                background-color: #2d2d2d;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0e639c, stop:1 #4db3ff);
                border-radius: 8px;
            }
        """)
        loading_layout.addWidget(self.progress_bar)

        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("""
            background-color: #1e1e1e;
            color: #d4d4d4;
            font-family: 'Consolas', 'Courier New';
            font-size: 11px;
            border: 1px solid #333;
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
        self.y_offset_input = QSpinBox()
        self.y_offset_input.setRange(0, 10000)
        self.y_offset_input.setSingleStep(100)
        self.y_offset_input.setPrefix("V-Offset: ")
        self.y_offset_input.setStyleSheet("background-color: #3c3c3c; height: 30px;")
        
        self.btn_refresh_scroll = QPushButton("REFRESH VIEW")
        self.btn_refresh_scroll.clicked.connect(self.on_refresh_with_scroll)
        
        scroll_layout.addWidget(self.y_offset_input)
        scroll_layout.addWidget(self.btn_refresh_scroll)










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
