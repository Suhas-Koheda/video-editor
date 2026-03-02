import sys
import types
import av

if not hasattr(av, 'subtitles'):
    sub_mod = types.ModuleType("subtitles")
    av.subtitles = sub_mod
    sys.modules["av.subtitles"] = sub_mod

    stream_mod = types.ModuleType("stream")

    SubtitleStream = type("SubtitleStream", (), {})
    stream_mod.SubtitleStream = SubtitleStream
    sub_mod.SubtitleStream = SubtitleStream

    sub_mod.stream = stream_mod
    sys.modules["av.subtitles.stream"] = stream_mod

from gui import EditorApp
from processor.tracker_cloud import ph
from PySide6.QtWidgets import QApplication
app = QApplication(sys.argv)

window = EditorApp()
window.show()

exit_code = app.exec()
if ph:
    ph.shutdown()
sys.exit(exit_code)
