"""
Lillith Display - PyQt5 GUI Launcher
Pop-up window for launching and controlling Lillith V4.1 system.
Separate from terminal and VS Code.
Includes camera video preview, microphone EQ visualization, and screen capture.
"""

import sys
import subprocess
from pathlib import Path
import threading
import numpy as np
import cv2
import sounddevice as sd
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, QThread, pyqtSignal


# Utility functions for enumerating devices
def get_available_cameras():
    """Enumerate available cameras on Windows."""
    cameras = []
    for i in range(10):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cameras.append((i, f"Camera {i}"))
                cap.release()
        except Exception:
            pass
    return cameras if cameras else [(0, "Camera 0")]


def get_available_screens():
    """Enumerate available screens/displays on Windows using Windows API."""
    screens = []
    try:
        from screeninfo import get_monitors
        monitors = get_monitors()
        for idx, monitor in enumerate(monitors):
            screens.append((idx + 1000, f"Screen {idx + 1} ({monitor.width}x{monitor.height})"))
    except ImportError:
        screens = [(1000, "Screen 1")]
    return screens


def get_available_mics():
    """Enumerate available microphones on Windows."""
    try:
        devices = sd.query_devices()
        mics = []
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                mics.append((i, dev['name']))
        return mics if mics else [(0, "Mic 0")]
    except Exception:
        return [(0, "Mic 0")]


# GUI Components
class ProcessConsole(QtWidgets.QPlainTextEdit):
    """Read-only console for displaying backend output."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        font = QtGui.QFont("Consolas", 10)
        self.setFont(font)

    def append_text(self, text: str):
        """Append text to console and scroll to end."""
        self.moveCursor(QtGui.QTextCursor.End)
        self.insertPlainText(text)
        self.moveCursor(QtGui.QTextCursor.End)


class CameraThread(QThread):
    """Thread for capturing camera frames or screen captures."""
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = True
        self.is_screen = camera_index >= 1000

    def run(self):
        """Capture frames from camera or screen."""
        if self.is_screen:
            self.capture_screen()
        else:
            self.capture_camera()

    def capture_camera(self):
        """Capture from physical camera."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.running:
            ret, frame = cap.read()
            if ret:
                self.frame_signal.emit(frame)
            else:
                break

        cap.release()

    def capture_screen(self):
        """Capture from screen/display using PIL."""
        screen_idx = self.camera_index - 1000
        try:
            from PIL import ImageGrab
            from screeninfo import get_monitors
            monitors = get_monitors()
            if screen_idx >= len(monitors):
                screen_idx = 0
            monitor = monitors[screen_idx]
            bbox = (monitor.x, monitor.y, monitor.x + monitor.width, monitor.y + monitor.height)

            while self.running:
                pil_image = ImageGrab.grab(bbox=bbox)
                frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (640, 480))
                self.frame_signal.emit(frame)
                QtCore.QThread.msleep(33)
        except Exception as e:
            print(f"Screen capture error: {e}")

    def stop(self):
        """Stop camera/screen thread."""
        self.running = False


class AudioThread(QThread):
    """Thread for capturing audio and computing spectrum."""
    spectrum_signal = pyqtSignal(np.ndarray)

    def __init__(self, mic_index=0):
        super().__init__()
        self.mic_index = mic_index
        self.running = True
        self.block_size = 2048
        self.sample_rate = 44100

    def audio_callback(self, indata, frames, time_info, status):
        """Audio stream callback."""
        if status:
            return

        audio_data = indata[:, 0].astype(np.float32)
        spectrum = np.abs(np.fft.rfft(audio_data))[:256]
        self.spectrum_signal.emit(spectrum)

    def run(self):
        """Start audio stream."""
        with sd.InputStream(device=self.mic_index, channels=1, callback=self.audio_callback,
                            blocksize=self.block_size, samplerate=self.sample_rate):
            while self.running:
                QtCore.QThread.msleep(10)

    def stop(self):
        """Stop audio thread."""
        self.running = False


# Main GUI Class
class DisplayApp(QtWidgets.QMainWindow):
    """Main application window."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lillith Display")
        self.setGeometry(100, 100, 800, 600)

        # Dropdowns
        self.camera_dropdown = QtWidgets.QComboBox(self)
        self.camera_dropdown.addItems([name for _, name in get_available_cameras()])

        self.mic_dropdown = QtWidgets.QComboBox(self)
        self.mic_dropdown.addItems([name for _, name in get_available_mics()])

        self.screen_dropdown = QtWidgets.QComboBox(self)
        self.screen_dropdown.addItems([name for _, name in get_available_screens()])

        # Buttons
        self.launch_button = QtWidgets.QPushButton("Launch", self)
        self.save_button = QtWidgets.QPushButton("Save State", self)
        self.shutdown_button = QtWidgets.QPushButton("Shutdown", self)
        self.kill_button = QtWidgets.QPushButton("Kill", self)

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.camera_dropdown)
        layout.addWidget(self.mic_dropdown)
        layout.addWidget(self.screen_dropdown)
        layout.addWidget(self.launch_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.shutdown_button)
        layout.addWidget(self.kill_button)

        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DisplayApp()
    window.show()
    sys.exit(app.exec_())
