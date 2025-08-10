# app/camera_feed.py

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
import cv2


class CameraFeed(QThread):
    """
    Captures frames from the webcam in a background thread and emits both:
    - A QImage for displaying in the GUI.
    - The raw BGR frame (NumPy array) for processing (e.g., detection).
    """
    image_updated = pyqtSignal(QImage)   # Signal to send QImage for GUI
    frame_updated = pyqtSignal(object)   # Signal to send raw frame (BGR ndarray)

    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)
        self.cap = cv2.VideoCapture(camera_index)
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue

            # Emit the raw frame for processing
            self.frame_updated.emit(frame.copy())

            # Convert to RGB for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_updated.emit(qt_image)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
        if self.cap.isOpened():
            self.cap.release()
