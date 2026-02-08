import sys
import types
import av
# Enhanced Monkeypatch for PyAV 13.0.0+ 
if not hasattr(av, 'subtitles'):
    sub_mod = types.ModuleType("subtitles")
    av.subtitles = sub_mod
    sys.modules["av.subtitles"] = sub_mod
    
    stream_mod = types.ModuleType("stream")
    # Define SubtitleStream to satisfy imports
    SubtitleStream = type("SubtitleStream", (), {})
    stream_mod.SubtitleStream = SubtitleStream
    sub_mod.SubtitleStream = SubtitleStream
    
    sub_mod.stream = stream_mod
    sys.modules["av.subtitles.stream"] = stream_mod

from PySide6.QtWidgets import QApplication
from gui import EditorApp

app = QApplication(sys.argv)

window = EditorApp()
window.show()

sys.exit(app.exec())