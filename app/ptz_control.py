import subprocess
import platform
import re
import time
import threading
from queue import Queue, Empty
from app import config


class PTZController:
    """
    Controls a PTZ (Pan-Tilt-Zoom) camera on Linux using v4l2-ctl.
    All hardware communication (subprocess calls) is handled in a separate
    thread to prevent blocking the main application.
    """
    def __init__(self, device_identifier=None):
        """
        Initializes the PTZ controller and starts the background worker thread.
        """
        self.osname = platform.system().lower()
        if self.osname != "linux":
            raise NotImplementedError(f"[PTZ] PTZ control is only supported on Linux. OS detected: {self.osname}")

        self.cfg = config.CONFIG
        self._presets = {}
        self._ctrl_ranges = {}
        self.device = device_identifier if device_identifier else "/dev/video0"

        # --- Threading and Command Queue ---
        self.command_queue = Queue()
        self.stop_event = threading.Event()
        self._state_lock = threading.Lock()

        # --- Internal state for current and target PTZ values ---
        self._pan, self._tilt, self._zoom = 0.0, 0.0, 0.0
        self._pan_target, self._tilt_target, self._zoom_target = 0.0, 0.0, 0.0

        # Easing settings
        self.smoothing_pan = self.cfg.get("ptz_easing_pan", 0.10)
        self.smoothing_tilt = self.cfg.get("ptz_easing_tilt", 0.15)
        self.smoothing_zoom = self.cfg.get("ptz_easing_zoom", 0.15)

        if not self._load_control_ranges():
            raise RuntimeError("[PTZ] No PTZ controls found on device.")

        self._pan_min, self._pan_max = self._get_ctrl_range("pan_absolute")
        self._tilt_min, self._tilt_max = self._get_ctrl_range("tilt_absolute")
        self._zoom_min, self._zoom_max = self._get_ctrl_range("zoom_absolute")

        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self._sync_from_device()

    def _worker_loop(self):
        """
        The main loop for the worker thread.
        It waits for commands from the queue and executes blocking subprocess calls.
        """
        while not self.stop_event.is_set():
            try:
                cmd, args = self.command_queue.get(timeout=0.1)
                if cmd == 'set':
                    ctrl_name, value = args
                    self._execute_v4l2_set(ctrl_name, value)
                elif cmd == 'get_and_update_state':
                    self._execute_sync_from_device()
            except Empty:
                continue
            except Exception as e:
                print(f"[PTZ Worker Error] {e}")

    def _execute_v4l2_set(self, ctrl_name, value):
        """Executes the blocking v4l2-ctl set command. ONLY called by worker."""
        cmin, cmax = self._get_ctrl_range(ctrl_name)
        clamped_val = max(min(value, cmax), cmin)
        try:
            subprocess.run(
                ["v4l2-ctl", "--device", self.device, "--set-ctrl", f"{ctrl_name}={int(clamped_val)}"],
                check=True, capture_output=True, text=True
            )
        except Exception as e:
            print(f"[PTZ] Failed to set {ctrl_name}: {e}")
        return clamped_val

    def _execute_sync_from_device(self):
        """Executes blocking get commands and updates state. ONLY called by worker."""
        def _get(ctrl_name, fallback_value):
            try:
                result = subprocess.run(
                    ["v4l2-ctl", "--device", self.device, "--get-ctrl", ctrl_name],
                    capture_output=True, text=True, check=True
                )
                return float(result.stdout.strip().split(":")[1].strip())
            except Exception:
                return fallback_value
        pan = _get("pan_absolute", self._pan)
        tilt = _get("tilt_absolute", self._tilt)
        zoom = _get("zoom_absolute", self._zoom)
        with self._state_lock:
            self._pan, self._pan_target = pan, pan
            self._tilt, self._tilt_target = tilt, tilt
            self._zoom, self._zoom_target = zoom, zoom
        print(f"[PTZ] Synced from hardware: pan={self._pan}, tilt={self._tilt}, zoom={self._zoom}")

    def stop(self):
        """Stops the worker thread gracefully."""
        self.stop_event.set()
        self.worker_thread.join(timeout=2)

    def _load_control_ranges(self):
        try:
            result = subprocess.run(
                ["v4l2-ctl", "--device", self.device, "--list-ctrls"],
                capture_output=True, text=True, check=True
            )
            for line in result.stdout.splitlines():
                m = re.search(r"(\w+).*min=(-?\d+)\s+max=(-?\d+)", line)
                if m:
                    self._ctrl_ranges[m.group(1)] = (int(m.group(2)), int(m.group(3)))
            return bool(self._ctrl_ranges)
        except Exception as e:
            print(f"[PTZ] Failed to load control ranges: {e}")
            return False

    def _get_ctrl_range(self, ctrl_name):
        return self._ctrl_ranges.get(ctrl_name, (-10000, 10000))
    
    def get_physical_ranges(self):
        """Query actual physical movement ranges from v4l2-ctl."""
        try:
            # Get detailed control info with step and default values
            result = subprocess.run(
                ["v4l2-ctl", "--device", self.device, "--list-ctrls-menus"],
                capture_output=True, text=True, check=True
            )
            
            pan_info = tilt_info = zoom_info = None
            
            for line in result.stdout.splitlines():
                if "pan_absolute" in line:
                    # Extract step size if available
                    step_match = re.search(r"step=(\d+)", line)
                    pan_step = int(step_match.group(1)) if step_match else 1
                    pan_info = {'step': pan_step, 'range': self._pan_max - self._pan_min}
                    
                elif "tilt_absolute" in line:
                    step_match = re.search(r"step=(\d+)", line)
                    tilt_step = int(step_match.group(1)) if step_match else 1
                    tilt_info = {'step': tilt_step, 'range': self._tilt_max - self._tilt_min}
                    
                elif "zoom_absolute" in line:
                    step_match = re.search(r"step=(\d+)", line)
                    zoom_step = int(step_match.group(1)) if step_match else 1
                    zoom_info = {'step': zoom_step, 'range': self._zoom_max - self._zoom_min}
            
            # Try to get actual device capabilities
            caps_result = subprocess.run(
                ["v4l2-ctl", "--device", self.device, "--all"],
                capture_output=True, text=True, check=True
            )
            
            # Look for PTZ capability info that might indicate physical ranges
            physical_pan_range = 612000.0  # Default assumption
            physical_tilt_range = 324000.0  # Default assumption
            
            # Some cameras report physical ranges in capabilities
            for line in caps_result.stdout.splitlines():
                if "pan" in line.lower() and "degree" in line.lower():
                    degree_match = re.search(r"(\d+\.?\d*)\s*degree", line)
                    if degree_match:
                        physical_pan_range = float(degree_match.group(1)) * 2  # Usually ±X degrees
                elif "tilt" in line.lower() and "degree" in line.lower():
                    degree_match = re.search(r"(\d+\.?\d*)\s*degree", line)
                    if degree_match:
                        physical_tilt_range = float(degree_match.group(1)) * 2  # Usually ±X degrees
            
            return {
                'pan_degrees_per_unit': physical_pan_range / pan_info['range'] if pan_info else 0.001,
                'tilt_degrees_per_unit': physical_tilt_range / tilt_info['range'] if tilt_info else 0.001,
                'pan_step': pan_info['step'] if pan_info else 1,
                'tilt_step': tilt_info['step'] if tilt_info else 1,
                'zoom_step': zoom_info['step'] if zoom_info else 1,
                'physical_pan_range': physical_pan_range,
                'physical_tilt_range': physical_tilt_range
            }
            
        except Exception as e:
            print(f"[PTZ] Failed to query physical ranges: {e}")
            # Fallback to reasonable defaults
            return {
                'pan_degrees_per_unit': 360.0 / (self._pan_max - self._pan_min),
                'tilt_degrees_per_unit': 180.0 / (self._tilt_max - self._tilt_min),
                'pan_step': 1,
                'tilt_step': 1,
                'zoom_step': 1,
                'physical_pan_range': 360.0,
                'physical_tilt_range': 180.0
            }


    def _v4l2_ctl_set(self, ctrl_name, value):
        self.command_queue.put(('set', (ctrl_name, value)))
        with self._state_lock:
            if 'pan' in ctrl_name: self._pan = value
            elif 'tilt' in ctrl_name: self._tilt = value
            elif 'zoom' in ctrl_name: self._zoom = value
        return value

    def _sync_from_device(self):
        self.command_queue.put(('get_and_update_state', None))

    def get_pan(self):
        with self._state_lock: return self._pan
    def get_tilt(self):
        with self._state_lock: return self._tilt
    def get_zoom(self):
        with self._state_lock: return self._zoom

    def set_pan(self, value):
        clamped = max(min(value, self._pan_max), self._pan_min)
        return self._v4l2_ctl_set("pan_absolute", clamped)

    def set_tilt(self, value):
        clamped = max(min(value, self._tilt_max), self._tilt_min)
        return self._v4l2_ctl_set("tilt_absolute", clamped)

    def set_zoom(self, value):
        clamped = max(min(value, self._zoom_max), self._zoom_min)
        return self._v4l2_ctl_set("zoom_absolute", clamped)

    def pan(self, delta):
        with self._state_lock:
            self._pan_target = self._pan + delta
            self._pan_target = max(min(self._pan_target, self._pan_max), self._pan_min)
            print(f"[PTZ] New pan target set to {self._pan_target}")

    def tilt(self, delta):
        with self._state_lock:
            self._tilt_target = self._tilt + delta
            self._tilt_target = max(min(self._tilt_target, self._tilt_max), self._tilt_min)
            print(f"[PTZ] New tilt target set to {self._tilt_target}")

    def zoom(self, delta):
        with self._state_lock:
            self._zoom_target = self._zoom + delta
            self._zoom_target = max(min(self._zoom_target, self._zoom_max), self._zoom_min)
            print(f"[PTZ] New zoom target set to {self._zoom_target}")

    def update(self):
        def ease(current, target, smoothing):
            # If close enough, snap to target and stop
            if abs(target - current) < 5:  # Dead zone
                return target
            return current + (target - current) * smoothing

        with self._state_lock:
            current_pan, pan_target = self._pan, self._pan_target
            current_tilt, tilt_target = self._tilt, self._tilt_target
            current_zoom, zoom_target = self._zoom, self._zoom_target

        # Calculate new positions
        new_pan = ease(current_pan, pan_target, self.smoothing_pan)
        new_tilt = ease(current_tilt, tilt_target, self.smoothing_tilt)
        new_zoom = ease(current_zoom, zoom_target, self.smoothing_zoom)
        
        # Only send commands if something actually changed
        commands_sent = False
        
        if abs(new_pan - current_pan) > 2:  # Meaningful change threshold
            self.set_pan(new_pan)
            commands_sent = True
        
        if abs(new_tilt - current_tilt) > 2:
            self.set_tilt(new_tilt)
            commands_sent = True
            
        if abs(new_zoom - current_zoom) > 2:
            self.set_zoom(new_zoom)
            commands_sent = True
        
        # Debug: show when we stop sending commands
        if not commands_sent and hasattr(self, '_was_moving'):
            if self._was_moving:
                print("[PTZ] Reached target - stopping updates")
            self._was_moving = False
        elif commands_sent:
            self._was_moving = True


    def set_preset(self, name):
        self._presets[name] = (self.get_pan(), self.get_tilt(), self.get_zoom())
        print(f"[PTZ] Preset saved '{name}': pan={self._presets[name][0]}, tilt={self._presets[name][1]}, zoom={self._presets[name][2]}")

    def goto_preset(self, name):
        if name not in self._presets:
            print(f"[PTZ] Preset '{name}' not found")
            return
        pan, tilt, zoom = self._presets[name]
        with self._state_lock:
            self._pan_target = pan
            self._tilt_target = tilt
            self._zoom_target = zoom
        print(f"[PTZ] Moving to preset '{name}': pan={pan}, tilt={tilt}, zoom={zoom}")

    def go_home(self):
        with self._state_lock:
            self._pan_target = 0.0
            self._tilt_target = 0.0
            self._zoom_target = 0.0
        print("[PTZ] Going home to pan=0, tilt=0, zoom=0")
