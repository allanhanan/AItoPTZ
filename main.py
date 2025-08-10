# main.py
"""
Main entry point for the AItoPTZ application.
Initializes the Qt Application and the main GUI window.
"""
import os
import sys
from PyQt5.QtWidgets import QApplication
from app.gui import MainWindow

# This line may be necessary on some Linux systems to specify the Qt platform plugin path,
# especially when running within virtual environments like Conda.
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/home/allan/.conda/envs/AItoPTZ/lib/python3.9/site-packages/PyQt5/Qt5/plugins/platforms"


def main():
    """Initializes and runs the Qt application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()