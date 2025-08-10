# app/controller.py

import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from app import config
from app.ptz_control import PTZController
from app.vision import VisionSystem

class Controller(QObject):
    annotated_frame = pyqtSignal(np.ndarray)
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.ptz = None
        self.cfg = config.CONFIG
        self.tracked_ids = []
        self.current_index = 0
        self._frame_count = 0
        self.tracking_timeout_threshold = 10
        self.frames_since_last_detection = 0
        self.last_tracked_box = None
        self.last_tilt_step = 0.0  # For smoothing
        
        # PID gains
        self.Kp_pan = self.cfg.get('pid_Kp_pan', 1.0)
        self.Kd_pan = self.cfg.get('pid_Kd_pan', 0.1)
        self.Kp_tilt = self.cfg.get('pid_Kp_tilt', 1.2)
        self.Kd_tilt = self.cfg.get('pid_Kd_tilt', 0.1)
        
        # Default scales
        self.tracking_pan_scale = 1000
        self.tracking_tilt_scale = 1000
        self.tracking_zoom_scale = 100
        self.manual_pan_scale = 2000
        self.manual_tilt_scale = 2000
        self.manual_zoom_scale = 200
        
        self.vision = VisionSystem(
            person_model=self.cfg['yolo_model'],
            face_model=self.cfg['face_model']
        )
        
        self.last_frame = None

    def _reset_tracking_state(self):
        print("[Control] Resetting tracking state.")
        self.last_tilt_step = 0.0

    def set_mode(self, mode: str):
        self.vision.set_mode(mode)
        print(f"Track mode set to {mode}")

    def select_person(self, x: int, y: int):
        if self.last_frame is not None:
            self.vision.select_person(self.last_frame, x, y)
            if self.vision.target_id is not None:
                print(f"Selected target ID {self.vision.target_id}")
                self._reset_tracking_state()

    def stop_tracking(self):
        if self.vision.target_id is not None:
            print("Tracking stopped.")
            self.vision.target_id = None
            self._reset_tracking_state()
        if self.ptz:
            self.ptz.go_home()

    def manual_pan(self, delta: float):
        if self.vision.target_id is not None:
            print("Manual control initiated, stopping auto-tracking.")
            self.vision.target_id = None
        if self.ptz:
            zoom_scale = self._get_zoom_scale_factor()
            scaled_delta = delta * self.manual_pan_scale * zoom_scale
            self.ptz.pan(scaled_delta)

    def manual_tilt(self, delta: float):
        if self.vision.target_id is not None:
            print("Manual control initiated, stopping auto-tracking.")
            self.vision.target_id = None
        if self.ptz:
            zoom_scale = self._get_zoom_scale_factor()
            scaled_delta = delta * self.manual_tilt_scale * zoom_scale
            self.ptz.tilt(scaled_delta)

    def manual_zoom(self, delta: float):
        if self.vision.target_id is not None:
            print("Manual control initiated, stopping auto-tracking.")
            self.vision.target_id = None
        if self.ptz:
            self.ptz.zoom(delta * self.manual_zoom_scale)

    def _get_zoom_scale_factor(self):
        if not self.ptz:
            return 1.0
        current_zoom = self.ptz.get_zoom()
        zoom_min = self.ptz._zoom_min
        zoom_max = self.ptz._zoom_max
        zoom_normalized = (current_zoom - zoom_min) / (zoom_max - zoom_min) if zoom_max != zoom_min else 0
        scale_factor = 1.0 - (zoom_normalized * 0.9)
        return max(scale_factor, 0.1)

    def _calculate_adaptive_scales(self) -> None:
        if not self.ptz:
            return
        
        pan_range = self.ptz._pan_max - self.ptz._pan_min
        tilt_range = self.ptz._tilt_max - self.ptz._tilt_min
        zoom_range = self.ptz._zoom_max - self.ptz._zoom_min
        
        self.manual_pan_scale = pan_range * 0.035
        self.manual_tilt_scale = tilt_range * 0.105
        self.manual_zoom_scale = zoom_range * 0.15
        self.tracking_pan_scale = pan_range * 0.0315
        
        # Adjust tilt scale by mode
        if self.vision.mode == 'face':
            self.tracking_tilt_scale = tilt_range * 10.4  # Aggressive for face
        else:
            self.tracking_tilt_scale = tilt_range * 0.3  # Standard
        
        self.tracking_zoom_scale = zoom_range * 0.15
        
        print(f"[Control] Dynamic scales:")
        print(f"  Pan range: {pan_range}, Manual: {self.manual_pan_scale:.0f}, Tracking: {self.tracking_pan_scale:.0f}")
        print(f"  Tilt range: {tilt_range}, Manual: {self.manual_tilt_scale:.0f}, Tracking: {self.tracking_tilt_scale:.0f}")
        print(f"  Zoom range: {zoom_range}, Manual: {self.manual_zoom_scale:.0f}, Tracking: {self.tracking_zoom_scale:.0f}")

    def process_frame(self, frame: np.ndarray):
        self._frame_count += 1
        self.last_frame = frame.copy()

        if self.ptz and (self._frame_count % 5 == 0):
            self.ptz.update()

        annotated, box, ids = self.vision.process_frame(frame)
        self.tracked_ids = ids
        self.annotated_frame.emit(annotated)

        if self.vision.target_id is not None and box is not None and self.ptz is not None:
            self.frames_since_last_detection = 0
            
            h, w = frame.shape[:2]
            cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
            target_x, target_y = (w / 2, h / 2.2) if self.vision.mode == 'face' else (w / 2, h / 2)
            
            dx, dy = (target_x - cx) / (w / 2), (target_y - cy) / (h / 2)
            
            # Use person box for zoom in face mode
            if self.vision.mode == 'face' and hasattr(self.vision, '_person_box_for_zoom') and self.vision._person_box_for_zoom is not None:
                zoom_box = self.vision._person_box_for_zoom
                box_height_fraction = (zoom_box[3] - zoom_box[1]) / h
            else:
                box_height_fraction = (box[3] - box[1]) / h
            
            # Sensitivity calculation
            min_sensitivity = 0.1
            max_sensitivity = 1.0
            distance_sensitivity = max(min_sensitivity, max_sensitivity - box_height_fraction)
            
            
            # Debug
            if self._frame_count % 15 == 0:
                print(f"[DEBUG] Frame: {w}x{h}, Target: ({target_x:.0f},{target_y:.0f}), Person: ({cx:.0f},{cy:.0f})")
                print(f"[DEBUG] dx={dx:.3f}, dy={dy:.3f}, box_frac={box_height_fraction:.2f}, distance_sens={distance_sensitivity:.2f}")
            
            self.last_tracked_box = box
            
            # Movement urgency
            abs_dx, abs_dy = abs(dx), abs(dy)
            pan_urgency_exp = 1.5
            tilt_urgency_exp = 1.5
            pan_urgency = min(abs_dx ** pan_urgency_exp, 1.0)
            tilt_urgency = min(abs_dy ** tilt_urgency_exp, 0.5)
            
            base_pan_multiplier = 25
            base_tilt_multiplier = 12
            pan_base_gain = 0.2
            tilt_base_gain = 0.15
            
            base_pan_gain = pan_base_gain * (1.0 + pan_urgency * base_pan_multiplier) * distance_sensitivity
            base_tilt_gain = tilt_base_gain * (1.0 + tilt_urgency * base_tilt_multiplier) * distance_sensitivity
            
            pan_step = -base_pan_gain * dx
            tilt_step = base_tilt_gain * dy

            
            pan_base_step = 0.03
            tilt_base_step = 0.02
            pan_max_multiplier = 0.5
            tilt_max_multiplier = 0.25
            
            max_pan_step = (pan_base_step + (pan_urgency * pan_max_multiplier)) * distance_sensitivity
            max_tilt_step = (tilt_base_step + (tilt_urgency * tilt_max_multiplier)) * distance_sensitivity
            
            pan_step = max(min(pan_step, max_pan_step), -max_pan_step)
            tilt_step = max(min(tilt_step, max_tilt_step), -max_tilt_step)
            
            # Smoothing for tilt
            tilt_step = tilt_step * 0.8 + self.last_tilt_step * 0.2
            self.last_tilt_step = tilt_step
            
            if self._frame_count % 15 == 0:
                print(f"[DEBUG] Urgency: pan={pan_urgency:.2f}, tilt={tilt_urgency:.2f}")
                print(f"[DEBUG] Gains: pan={base_pan_gain:.3f}, tilt={base_tilt_gain:.3f}")
                print(f"[DEBUG] Steps: pan={pan_step:.4f}, tilt={tilt_step:.4f}")
            
            zoom_scale = self._get_zoom_scale_factor()
            
            # Zoom control
            target_sizes = {
                'full_body': 0.7,
                'torso': 0.5, 
                'face': 1.0  # Adjusted for face zoom without overshoot
            }
            target_size = target_sizes[self.vision.mode]
            
            size_error = target_size - box_height_fraction
            
            zoom_scale_factor = 1000
            max_zoom_command = 800
            zoom_deadzone = 0.005

            if self.vision.mode == 'face':
                # More aggressive tilt for face mode
                base_tilt_gain *= 4.0  # Quadruple the tilt responsiveness
                max_tilt_step *= 3.0   # Allow larger tilt steps
                
                # Less smoothing for faster response
                tilt_step = tilt_step * 0.95 + self.last_tilt_step * 0.05
            else:
                # Standard smoothing for other modes
                tilt_step = tilt_step * 0.8 + self.last_tilt_step * 0.2
            
            zoom_command = round(size_error * zoom_scale_factor)
            zoom_command = max(min(zoom_command, max_zoom_command), -max_zoom_command)
            
            if self._frame_count % 15 == 0:
                print(f"[DEBUG] Box size: {box_height_fraction:.2f}, Target: {target_size:.2f}, Error: {size_error:.3f}")
                print(f"[DEBUG] Zoom command: {zoom_command}")
            
            pan_tilt_deadzone = 0.001
            
            if self._frame_count % 5 == 0:
                if abs(size_error) > zoom_deadzone and self.frames_since_last_detection == 0:
                    if self._frame_count % 15 == 0:
                        print(f"[DEBUG] Sending zoom command: {zoom_command}")
                    self.ptz.zoom(zoom_command)
                else:
                    if self._frame_count % 15 == 0:
                        if self.frames_since_last_detection > 0:
                            print(f"[DEBUG] Skipping zoom - lost target for {self.frames_since_last_detection} frames")
                        else:
                            print(f"[DEBUG] Zoom error {size_error:.3f} within deadzone {zoom_deadzone}")
                
                if abs(pan_step) > pan_tilt_deadzone:
                    pan_command = round(pan_step * self.tracking_pan_scale * zoom_scale)
                    if self._frame_count % 15 == 0:
                        print(f"[DEBUG] Pan command: {pan_command}")
                    self.ptz.pan(pan_command)
                    
                if abs(tilt_step) > pan_tilt_deadzone:
                    tilt_command = round(tilt_step * self.tracking_tilt_scale * zoom_scale)
                    if self._frame_count % 15 == 0:
                        print(f"[DEBUG] Tilt command: {tilt_command}")
                    self.ptz.tilt(tilt_command)

        elif self.vision.target_id is not None and box is None:
            self.frames_since_last_detection += 1
            
            if self.frames_since_last_detection > self.tracking_timeout_threshold:
                print(f"[Control] Lost target for {self.tracking_timeout_threshold} frames, stopping tracking")
                self._reset_tracking_state()
                self.vision.target_id = None
            else:
                if self.last_tracked_box is not None and self._frame_count % 5 == 0:
                    print(f"[Control] Target lost for {self.frames_since_last_detection} frames, maintaining position")

    def cycle_left(self):
        if self.tracked_ids:
            self.current_index = (self.current_index - 1) % len(self.tracked_ids)
            tid = self.tracked_ids[self.current_index]
            print(f"Cycled left to ID {tid}")

    def cycle_right(self):
        if self.tracked_ids:
            self.current_index = (self.current_index + 1) % len(self.tracked_ids)
            tid = self.tracked_ids[self.current_index]
            print(f"Cycled right to ID {tid}")

    def select_target(self):
        if self.tracked_ids:
            tid = self.tracked_ids[self.current_index]
            self.vision.target_id = tid
            self._reset_tracking_state()
            print(f"Tracking started on ID {tid}")

    @property
    def track_mode(self):
        return self.vision.mode

    @track_mode.setter
    def track_mode(self, mode):
        self.vision.set_mode(mode)

    def get_overlay_state(self):
        boxes = []
        selected_idx = -1
        frame = self.last_frame

        if frame is None or not self.tracked_ids:
            return boxes, selected_idx

        current_id = self.vision.target_id or (
            self.tracked_ids[self.current_index] if self.current_index < len(self.tracked_ids) else None
        )

        results = self.vision.model.track(frame, classes=[0], persist=True, verbose=False)[0]

        if results.boxes is None or results.boxes.id is None or results.boxes.xyxy is None:
            return [], -1

        yolo_boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        yolo_ids = results.boxes.id.cpu().numpy().astype(int)

        for idx, (bbox, tid) in enumerate(zip(yolo_boxes, yolo_ids)):
            x1, y1, x2, y2 = bbox
            boxes.append((x1, y1, x2 - x1, y2 - y1))
            if tid == current_id:
                selected_idx = idx

        return boxes, selected_idx

    def set_preset(self, name: str):
        if self.ptz is None:
            print("[Preset] Cannot save preset: PTZ controller not initialized.")
            return

        pan = self.ptz.get_pan()
        tilt = self.ptz.get_tilt()
        zoom = self.ptz.get_zoom()

        self.cfg['presets'][name] = {"name": name, "pan": pan, "tilt": tilt, "zoom": zoom}
        config.save_config()
        print(f"[Preset] Saved '{name}' at pan={pan}, tilt={tilt}, zoom={zoom}")

    def set_camera_index(self, camera_index: int):
        try:
            device_path = f"/dev/video{camera_index}"
            if self.ptz:
                self.ptz.stop()
            
            self.ptz = PTZController(device_path)
            self._calculate_adaptive_scales()
            
            # Load presets
            config_presets = self.cfg.get('presets', {})
            for preset_name, preset_data in config_presets.items():
                if isinstance(preset_data, dict) and all(k in preset_data for k in ['pan', 'tilt', 'zoom']):
                    self.ptz._presets[preset_name] = (
                        preset_data['pan'], 
                        preset_data['tilt'], 
                        preset_data['zoom']
                    )
                    print(f"[PTZ] Loaded preset '{preset_name}': {preset_data}")
            
            print(f"[PTZ] Switched to camera device {device_path}")
        except Exception as e:
            print(f"[PTZ] Failed to switch PTZ controller: {e}")
            self.ptz = None

    def goto_preset(self, name: str):
        if self.ptz is None:
            print("[Preset] Cannot move to preset: PTZ controller not initialized.")
            return

        config_preset = self.cfg['presets'].get(name)
        if not config_preset:
            print(f"[Preset] Not found: '{name}'")
            return

        self.vision.target_id = None
        self._reset_tracking_state()

        with self.ptz._state_lock:
            self.ptz._pan_target = config_preset['pan']
            self.ptz._tilt_target = config_preset['tilt']
            self.ptz._zoom_target = config_preset['zoom']

        print(f"[Preset] Moving to '{name}': pan={config_preset['pan']}, tilt={config_preset['tilt']}, zoom={config_preset['zoom']}")
