import sys

from PyQt5.QtWidgets import QApplication

from src.gui.main_window import MainWindow

if __name__ == "__main__":
    # The Wayland warning can be ignored or bypassed by using:
    # QT_QPA_PLATFORM=xcb python3 main.py
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1000, 600)
    window.show()
    sys.exit(app.exec_())
