import sys
from PySide6.QtWidgets import QApplication
from gui import EditorApp

app = QApplication(sys.argv)

window = EditorApp()
window.show()

sys.exit(app.exec())