# app/gui.py
import platform
import subprocess
import re
from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QWidget, QApplication, QDialog, QGridLayout, QInputDialog,
    QAction, QGroupBox, QComboBox, QSizePolicy, QShortcut
)
from PyQt5.QtGui import QPixmap, QKeySequence
from PyQt5.QtCore import Qt, QEvent, QTimer
from PyQt5.QtGui import QPainter, QPen, QColor, QImage

import qdarkstyle
import cv2

from app.camera_feed import CameraFeed
from app.controller import Controller
from app import config


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PTZ Camera Tracker")
        self.resize(800, 700)
        QApplication.instance().setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

        self.video_label = QLabel()
        self.video_label.setFixedSize(800, 450)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000;")
        self.setCentralWidget(self.video_label)

        self.statusBar().showMessage("Ready")
        self.controller = Controller()

        self.setFocusPolicy(Qt.StrongFocus)
        self.video_label.setFocusPolicy(Qt.NoFocus)

        self._pressed_keys = set()
        self._ptz_keymap = {}
        self._rebuild_ptz_keymap()

        # --- NEW: State for tracking key hold duration ---
        self._key_hold_frames = {} # Stores frame count for each held key

        # This loop is KEPT as requested for polling key presses
        self._ptz_timer = QTimer()
        self._ptz_timer.timeout.connect(self._ptz_update_loop)
        self._ptz_timer.start(33)  # ~30 FPS

        self._camera_started = False
        self.preset_buttons = []
        self.preset_names = {}
        self.setting_preset = False

        self._build_control_panel()
        self._create_menus()
        self._init_keybindings()
        self._load_presets()

        self._last_mouse_pos = None
        self.video_label.setMouseTracking(True)
        self.video_label.installEventFilter(self)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._camera_started:
            saved_index = config.CONFIG.get('camera_index', 0)
            self._start_camera(saved_index)
            self._camera_started = True
        self.setFocus()

    def _store_and_process_frame(self, frame):
        self.controller.last_frame = frame
        self.controller.process_frame(frame)

    def update_image(self, qt_img):
        pixmap = QPixmap.fromImage(qt_img)
        painter = QPainter(pixmap)
        pen_inactive = QPen(QColor(255, 255, 255, 128))
        pen_selected = QPen(QColor(0, 255, 0), 3)

        boxes, selected_idx = self.controller.get_overlay_state()

        for idx, (x, y, w, h) in enumerate(boxes):
            pen = pen_selected if idx == selected_idx else pen_inactive
            painter.setPen(pen)
            painter.drawRect(x, y, w, h)

        painter.end()
        self.video_label.setPixmap(pixmap)

    def _build_control_panel(self):
        panel = QGroupBox("Controls")
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QVBoxLayout()
        camera_select_row = QHBoxLayout()
        camera_select_row.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self._change_camera)
        camera_select_row.addWidget(self.camera_combo)
        layout.addLayout(camera_select_row)
        self._populate_camera_list()
        pan_tilt_layout = QHBoxLayout()
        btn_tilt_up = self._make_button("Tilt ▲", "tilt_up", lambda: self.controller.manual_tilt(1))
        btn_pan_left = self._make_button("◀ Pan", "pan_left", lambda: self.controller.manual_pan(-1))
        btn_home = self._make_button("Home", "home", self.controller.stop_tracking)
        btn_pan_right = self._make_button("Pan ▶", "pan_right", lambda: self.controller.manual_pan(1))
        btn_tilt_down = self._make_button("Tilt ▼", "tilt_down", lambda: self.controller.manual_tilt(-1))
        pan_tilt_layout.addWidget(btn_tilt_up)
        pan_tilt_layout.addWidget(btn_pan_left)
        pan_tilt_layout.addWidget(btn_home)
        pan_tilt_layout.addWidget(btn_pan_right)
        pan_tilt_layout.addWidget(btn_tilt_down)
        layout.addLayout(pan_tilt_layout)
        zoom_layout = QHBoxLayout()
        btn_zoom_in = self._make_button("Zoom +", "zoom_in", lambda: self.controller.manual_zoom(1))
        btn_zoom_out = self._make_button("Zoom -", "zoom_out", lambda: self.controller.manual_zoom(-1))
        zoom_layout.addStretch()
        zoom_layout.addWidget(btn_zoom_in)
        zoom_layout.addWidget(btn_zoom_out)
        zoom_layout.addStretch()
        layout.addLayout(zoom_layout)
        sel_layout = QHBoxLayout()
        btn_cycle_left = self._make_button("◀ Select", "cycle_left", self.controller.cycle_left)
        btn_track = self._make_button("Track ▶", "track", self.controller.select_target)
        btn_cycle_right = self._make_button("Select ▶", "cycle_right", self.controller.cycle_right)
        sel_layout.addWidget(btn_cycle_left)
        sel_layout.addWidget(btn_track)
        sel_layout.addWidget(btn_cycle_right)
        layout.addLayout(sel_layout)
        mode_layout = QHBoxLayout()
        self.mode_combo = QComboBox()
        for mode in ("face", "torso", "full_body"):
            self.mode_combo.addItem(mode.capitalize())
        self.mode_combo.setCurrentText(self.controller.track_mode.capitalize())
        self.mode_combo.currentTextChanged.connect(
            lambda txt: setattr(self.controller, "track_mode", txt.lower())
        )
        mode_layout.addStretch()
        mode_layout.addWidget(QLabel("Mode:"))
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)
        preset_layout = QVBoxLayout()
        preset_controls = QHBoxLayout()
        btn_add = self._make_button("+ Add Preset", None, self._add_preset_button)
        btn_set = self._make_button("Set Preset", None, self._enter_set_preset_mode)
        preset_controls.addWidget(btn_add)
        preset_controls.addWidget(btn_set)
        preset_layout.addLayout(preset_controls)
        self.preset_btn_row = QHBoxLayout()
        preset_layout.addLayout(self.preset_btn_row)
        layout.addLayout(preset_layout)
        panel.setLayout(layout)
        container = QWidget()
        outer = QVBoxLayout()
        outer.addWidget(self.video_label)
        outer.addWidget(panel)
        container.setLayout(outer)
        self.setCentralWidget(container)

    def _make_button(self, text, keybind_name, callback):
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(callback)
        if keybind_name:
            key = config.CONFIG['keybinds'].get(keybind_name, "")
            btn.setToolTip(f"{text} (Key: {key})")
        btn.setStyleSheet("padding: 6px; font-size: 14px;")
        return btn

    def _create_menus(self):
        menubar = self.menuBar()
        settings_menu = menubar.addMenu("Settings")
        kb_action = QAction("Keybindings...", self)
        kb_action.triggered.connect(self._show_keybind_dialog)
        settings_menu.addAction(kb_action)

    def _init_keybindings(self):
        self._shortcuts = {}
        def bind(action, func):
            key = config.CONFIG['keybinds'].get(action)
            if key:
                shortcut = QShortcut(QKeySequence(key), self)
                shortcut.activated.connect(func)
                self._shortcuts[action] = shortcut
        bind('cycle_left', self.controller.cycle_left)
        bind('cycle_right', self.controller.cycle_right)
        bind('track', self.controller.select_target)
        bind('home', self.controller.stop_tracking)
        for i in range(1, 10):
            shortcut = QShortcut(QKeySequence(str(i)), self)
            shortcut.activated.connect(lambda i=i: self._activate_preset_by_number(i))

    def eventFilter(self, source, event):
        if source is self.video_label and event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
            self.controller.select_person(event.x(), event.y())
            return True
        return super().eventFilter(source, event)

    def _rebuild_ptz_keymap(self):
        name_to_qt = {"Left": Qt.Key_Left, "Right": Qt.Key_Right, "Up": Qt.Key_Up, "Down": Qt.Key_Down, **{chr(c): getattr(Qt, f"Key_{chr(c)}") for c in range(ord('A'), ord('Z')+1)}}
        self._ptz_keymap.clear()
        for action in ['pan_left', 'pan_right', 'tilt_up', 'tilt_down', 'zoom_in', 'zoom_out']:
            key_str = config.CONFIG['keybinds'].get(action, "").capitalize()
            qt_key = name_to_qt.get(key_str)
            if qt_key: self._ptz_keymap[action] = qt_key

    # --- UPDATED key events to track hold duration ---
    def keyPressEvent(self, event):
        if not event.isAutoRepeat():
            key = event.key()
            self._pressed_keys.add(key)
            # Start counting how long the key is held
            if key in self._ptz_keymap.values():
                self._key_hold_frames[key] = 0

    def keyReleaseEvent(self, event):
        if not event.isAutoRepeat():
            key = event.key()
            self._pressed_keys.discard(key)
            # Reset the counter when the key is released
            if key in self._key_hold_frames:
                del self._key_hold_frames[key]

        
    # --- THE MAIN FIX: Acceleration logic in the update loop ---
    def _ptz_update_loop(self):
        if not self.controller.ptz:
            return

        # --- Acceleration Parameters (can be moved to config.py) ---
        base_step = 1.0  # The size of a single, quick tap
        acceleration_factor = 0.1 # How quickly the speed ramps up
        max_speed_multiplier = 8.0 # Max speed for a long press

        # A helper function to calculate the accelerating step
        def calculate_step(key_name, direction):
            key_code = self._ptz_keymap.get(key_name)
            if key_code in self._pressed_keys:
                # Increment the hold counter
                self._key_hold_frames[key_code] = self._key_hold_frames.get(key_code, 0) + 1
                
                # Calculate speed based on how long the key has been held
                hold_duration = self._key_hold_frames[key_code]
                speed_multiplier = min(1 + (hold_duration * acceleration_factor), max_speed_multiplier)
                
                return direction * base_step * speed_multiplier
            return 0

        # Calculate the step for each direction
        pan_delta = calculate_step('pan_left', -1) + calculate_step('pan_right', 1)
        tilt_delta = calculate_step('tilt_up', 1) + calculate_step('tilt_down', -1)
        zoom_delta = calculate_step('zoom_in', 1) + calculate_step('zoom_out', -1)

        # Send commands to the controller if there's movement
        if pan_delta != 0: self.controller.manual_pan(pan_delta)
        if tilt_delta != 0: self.controller.manual_tilt(tilt_delta)
        if zoom_delta != 0: self.controller.manual_zoom(zoom_delta)

    def closeEvent(self, event):
        if hasattr(self, 'camera'): self.camera.stop()
        if self.controller.ptz: self.controller.ptz.stop()
        event.accept()



    def _show_keybind_dialog(self):
        """OBS-style dialog to edit keybindings."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Edit Keybindings")
        layout = QGridLayout(dlg)
        keys = config.CONFIG['keybinds']
        row = 0
        for action, label in [
            ('pan_left', 'Pan Left'),
            ('pan_right', 'Pan Right'),
            ('tilt_up', 'Tilt Up'),
            ('tilt_down', 'Tilt Down'),
            ('zoom_in', 'Zoom In'),
            ('zoom_out', 'Zoom Out'),
            ('cycle_left', 'Cycle Left'),
            ('cycle_right', 'Cycle Right'),
            ('track', 'Track'),
            ('home', 'Home')
        ]:
            layout.addWidget(QLabel(label), row, 0)
            edit = QPushButton(keys[action])
            edit.clicked.connect(lambda _, a=action, b=edit: self._change_key(a, b))
            layout.addWidget(edit, row, 1)
            row += 1
        close = QPushButton("Close")
        close.clicked.connect(dlg.accept)
        layout.addWidget(close, row, 0, 1, 2)
        dlg.exec_()

    def _change_key(self, action, button):
        text, ok = QInputDialog.getText(self, "Change Key", f"New key for {action}:")
        if ok and text:
            config.CONFIG['keybinds'][action] = text
            button.setText(text)
            config.save_config()
            self._rebuild_ptz_keymap()


    def _rebuild_ptz_keymap(self):
        """Map config key names to Qt.Key_* codes."""
        name_to_qt = {
            "A": Qt.Key_A,
            "B": Qt.Key_B,
            "C": Qt.Key_C,
            "D": Qt.Key_D,
            "E": Qt.Key_E,
            "F": Qt.Key_F,
            "G": Qt.Key_G,
            "H": Qt.Key_H,
            "I": Qt.Key_I,
            "J": Qt.Key_J,
            "K": Qt.Key_K,
            "L": Qt.Key_L,
            "M": Qt.Key_M,
            "N": Qt.Key_N,
            "O": Qt.Key_O,
            "P": Qt.Key_P,
            "Q": Qt.Key_Q,
            "R": Qt.Key_R,
            "S": Qt.Key_S,
            "T": Qt.Key_T,
            "U": Qt.Key_U,
            "V": Qt.Key_V,
            "W": Qt.Key_W,
            "X": Qt.Key_X,
            "Y": Qt.Key_Y,
            "Z": Qt.Key_Z,
            "Left": Qt.Key_Left,
            "Right": Qt.Key_Right,
            "Up": Qt.Key_Up,
            "Down": Qt.Key_Down,
            "Space": Qt.Key_Space,
            # add more if needed
        }

        self._ptz_keymap.clear()
        for action in ['pan_left', 'pan_right', 'tilt_up', 'tilt_down', 'zoom_in', 'zoom_out']:
            key_str = config.CONFIG['keybinds'].get(action, "")
            key_str = key_str.strip().upper()
            qt_key = name_to_qt.get(key_str)
            if qt_key:
                self._ptz_keymap[action] = qt_key
            else:
                print(f"[Keybind] Warning: Unknown key '{key_str}' for {action}")

    def keyPressEvent(self, event):
        self._pressed_keys.add(event.key())

    def keyReleaseEvent(self, event):
        self._pressed_keys.discard(event.key())

    # ----------------------------------------
    # Mouse-driven PTZ (drag & wheel)
    # ----------------------------------------
    def eventFilter(self, source, event):
        # 1.  RIGHT-click still selects a person  (existing code)
        if source is self.video_label and event.type() == QEvent.MouseButtonPress \
                and event.button() == Qt.RightButton:
            self.controller.select_person(event.x(), event.y())
            return True

        # 2.  LEFT-button drag  → manual pan / tilt
        if source is self.video_label and event.type() == QEvent.MouseButtonPress \
                and event.button() == Qt.LeftButton:
            self._last_mouse_pos = event.pos()
            return True

        if source is self.video_label and event.type() == QEvent.MouseMove \
                and self._last_mouse_pos is not None:
            dx = event.x() - self._last_mouse_pos.x()   # +x → mouse right
            dy = event.y() - self._last_mouse_pos.y()   # +y → mouse down
            gain = 0.1                                 # drag sensitivity

            if dx:
                self.controller.manual_pan(-dx * gain)  # camera pans opposite
            if dy:
                self.controller.manual_tilt(dy * gain)

            self._last_mouse_pos = event.pos()
            return True

        if source is self.video_label and event.type() == QEvent.MouseButtonRelease \
                and event.button() == Qt.LeftButton:
            self._last_mouse_pos = None
            return True

        # 3.  Mouse wheel  → manual zoom
        if source is self.video_label and event.type() == QEvent.Wheel:
            step = 1 if event.angleDelta().y() > 0 else -1
            self.controller.manual_zoom(step)
            return True

        # fall back to the default handler
        return super().eventFilter(source, event)


    def _populate_camera_list(self):
        """Populate dropdown with available cameras and pre-select saved one."""
        # prevent firing change events while we rebuild
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()

        saved_index = config.CONFIG.get('camera_index', 0)
        found_index = -1
        system = platform.system()

        if system == "Linux":
            try:
                output = subprocess.check_output(["v4l2-ctl", "--list-devices"], encoding="utf-8")
                current_name = None
                for line in output.splitlines():
                    if not line.startswith("\t"):
                        current_name = line.strip()
                    else:
                        dev = line.strip()
                        m = re.search(r"/dev/video(\d+)", dev)
                        if m:
                            idx = int(m.group(1))
                            label = f"{current_name} ({dev})"
                            self.camera_combo.addItem(label, idx)
                            if idx == saved_index:
                                found_index = self.camera_combo.count() - 1
            except Exception:
                self._fallback_camera_probe()

        elif system == "Windows":
            try:
                import wmi # type: ignore
                w = wmi.WMI()
                # detect which indices actually open
                available = {i for i in range(10) if cv2.VideoCapture(i).isOpened()}
                for cam in w.Win32_PnPEntity():
                    if any(k in cam.Name for k in ("Camera", "Webcam")):
                        # assume sequential indices
                        for idx in sorted(available):
                            self.camera_combo.addItem(cam.Name, idx)
                            if idx == saved_index:
                                found_index = self.camera_combo.count() - 1
                            available.remove(idx)
                            break
                if self.camera_combo.count() == 0:
                    self._fallback_camera_probe()
            except Exception:
                self._fallback_camera_probe()

        else:
            self._fallback_camera_probe()

        # ensure we always have a valid selection
        if found_index < 0:
            found_index = 0
        self.camera_combo.setCurrentIndex(found_index)
        self.camera_combo.blockSignals(False)


    def _change_camera(self):
        """Called when the user picks a new camera from the dropdown."""
        idx = self.camera_combo.currentData()
        if idx is None or idx == config.CONFIG.get("camera_index"):
            return

        print(f"[Camera] Switching to index {idx}")
        config.CONFIG["camera_index"] = idx
        config.save_config()

        # stop existing feed (if any)
        if hasattr(self, "camera"):
            self.camera.stop()

        # start the new one
        self._start_camera(idx)
        # Update PTZ controller dynamically
        self.controller.set_camera_index(idx)



    def _start_camera(self, preferred_index: int):
        """
        Try preferred_index first, then fall back through 0–4.
        Releases each VideoCapture fully before moving on.
        """
        tried = [preferred_index] + [i for i in range(5) if i != preferred_index]
        for attempt in tried:
            print(f"[Camera] Trying camera index {attempt}")
            cap = cv2.VideoCapture(attempt, cv2.CAP_V4L)  # force V4L on Linux
            if not cap.isOpened():
                cap.release()
                err_msg = f"Camera {preferred_index} unavailable"
                print(f"[Camera] {err_msg}")
                self.statusBar().showMessage(err_msg)
                continue

            cap.release()
            # successful—hook up our CameraFeed thread
            self.camera = CameraFeed(camera_index=attempt)
            self.camera.image_updated.connect(self.update_image)
            self.camera.frame_updated.connect(self._store_and_process_frame)
            self.camera.start()

            # persist working index
            config.CONFIG["camera_index"] = attempt
            config.save_config()
            print(f"[Camera] Started on index {attempt}")
            self.statusBar().showMessage("Ready")
            self.controller.set_camera_index(attempt)
            return

        print("[Camera] All camera indices failed.")


    def _fallback_camera_probe(self):
        """Simple fallback for camera detection."""
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.camera_combo.addItem(f"Camera {i}", i)
                cap.release()


    def _show_bounds_dialog(self):
        """OBS-style dialog to set PTZ bounds."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Set Movement Bounds")
        layout = QGridLayout(dlg)
        bounds = config.CONFIG['bounds']
        row = 0
        for key, label in [
            ('pan_min', 'Left Pan Bound'),
            ('pan_max', 'Right Pan Bound'),
            ('tilt_min', 'Down Tilt Bound'),
            ('tilt_max', 'Up Tilt Bound')
        ]:
            layout.addWidget(QLabel(label), row, 0)
            value = QLabel(str(bounds[key]))
            layout.addWidget(value, row, 1)
            btn = QPushButton("Set Current")
            btn.clicked.connect(lambda _, k=key, v=value: self._set_bound(k, v))
            layout.addWidget(btn, row, 2)
            row += 1
        close = QPushButton("Close")
        close.clicked.connect(dlg.accept)
        layout.addWidget(close, row, 0, 1, 3)
        dlg.exec_()

    def _set_bound(self, key, label):
        if 'pan' in key:
            val = self.controller.pan
        else:
            val = self.controller.tilt
        config.CONFIG['bounds'][key] = val
        label.setText(f"{val:.1f}")
        config.save_config()


    def _add_preset_button(self):
        name, ok = QInputDialog.getText(self, "Preset Name", "Enter name for this preset:")
        if not ok or not name:
            return

        pid = str(len(self.preset_buttons) + 1)

        # Save name persistently
        config.CONFIG.setdefault("preset_names", {})[pid] = name
        config.save_config()

        # Create UI
        wrapper = QWidget()
        wrapper_layout = QHBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)

        btn = QPushButton(name)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet("padding: 6px; font-size: 13px;")
        btn.clicked.connect(lambda _, pid=pid: self._preset_clicked(pid))

        delete_btn = QPushButton("✕")
        delete_btn.setFixedSize(20, 20)
        delete_btn.setStyleSheet("color: red; font-size: 12px; padding: 0;")
        delete_btn.clicked.connect(lambda _, pid=pid, wrapper=wrapper: self._delete_preset_button(pid, wrapper))

        wrapper_layout.addWidget(btn)
        wrapper_layout.addWidget(delete_btn)

        self.preset_buttons.append((pid, wrapper))
        self.preset_names[pid] = name
        self.preset_btn_row.addWidget(wrapper)



    def _enter_set_preset_mode(self):
        self.setting_preset = True
        self.statusBar().showMessage("Click a preset button to save current camera position.")

    def _preset_clicked(self, name: str):
        if self.setting_preset:
            self.controller.set_preset(name)
            self.statusBar().showMessage(f"Preset '{name}' updated.")
            self.setting_preset = False
        else:
            self.controller.goto_preset(name)
            self.statusBar().showMessage(f"Moved to preset '{name}'")

    def _load_presets(self):
        presets = config.CONFIG.get("presets", {})
        names = config.CONFIG.get("preset_names", {})

        for pid in presets:
            name = names.get(pid, f"Preset {pid}")
            self.preset_names[pid] = name

            wrapper = QWidget()
            wrapper_layout = QHBoxLayout(wrapper)
            wrapper_layout.setContentsMargins(0, 0, 0, 0)

            btn = QPushButton(name)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet("padding: 6px; font-size: 13px;")
            btn.clicked.connect(lambda _, pid=pid: self._preset_clicked(pid))

            delete_btn = QPushButton("✕")
            delete_btn.setFixedSize(20, 20)
            delete_btn.setStyleSheet("color: red; font-size: 12px; padding: 0;")
            delete_btn.clicked.connect(lambda _, pid=pid, wrapper=wrapper: self._delete_preset_button(pid, wrapper))

            wrapper_layout.addWidget(btn)
            wrapper_layout.addWidget(delete_btn)

            self.preset_buttons.append((pid, wrapper))
            self.preset_btn_row.addWidget(wrapper)


    def _delete_preset_button(self, pid, widget):
        self.preset_names.pop(pid, None)
        self.preset_buttons = [(i, w) for i, w in self.preset_buttons if i != pid]
        widget.setParent(None)

        # Remove from config and save
        if pid in config.CONFIG.get("presets", {}):
            del config.CONFIG["presets"][pid]
        if pid in config.CONFIG.get("preset_names", {}):
            del config.CONFIG["preset_names"][pid]
        config.save_config()

    def _activate_preset_by_number(self, number):
        pid = str(number)
        if any(p[0] == pid for p in self.preset_buttons):
            self._preset_clicked(pid)


    def _ptz_update_loop(self):
        if not self.controller.ptz:
            return

        keys = self._pressed_keys

        if self._ptz_keymap.get('pan_left') in keys:
            self.controller.manual_pan(-1)
        if self._ptz_keymap.get('pan_right') in keys:
            self.controller.manual_pan(1)
        if self._ptz_keymap.get('tilt_up') in keys:
            self.controller.manual_tilt(1)
        if self._ptz_keymap.get('tilt_down') in keys:
            self.controller.manual_tilt(-1)
        if self._ptz_keymap.get('zoom_in') in keys:
            self.controller.manual_zoom(1)
        if self._ptz_keymap.get('zoom_out') in keys:
            self.controller.manual_zoom(-1)


    def closeEvent(self, event):
        self.camera.stop()
        event.accept()
