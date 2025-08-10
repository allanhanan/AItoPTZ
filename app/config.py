# app/config.py

import json
import os

# Path to config file (in project root)
CONFIG_FILE = os.path.join(os.path.dirname(__file__), os.pardir, 'config.json')

# Default configuration
DEFAULT_CONFIG = {
    "camera_index": 0,
    "presets": {},

    "bounds": {
        "pan_min": -45.0,   # degrees
        "pan_max": 45.0,
        "tilt_min": -30.0,
        "tilt_max": 30.0,
        "zoom_min": 1.0,    # arbitrary zoom units
        "zoom_max": 5.0
    },

    "keybinds": {
        "pan_left": "A",
        "pan_right": "D",
        "tilt_up": "W",
        "tilt_down": "S",
        "zoom_in": "I",
        "zoom_out": "K",
        "cycle_left": "L",
        "cycle_right": "R",
        "track": "Return",
        "home": "Backspace"
    },

    # Tracking + PTZ speed settings
    "yolo_model": "yolov8n.pt",
    "face_model": "yolov8n-face.pt",
    "face_tolerance": 0.6,

    "ptz_speed_pan": 1.0,     # pan speed multiplier (delta scaling)
    "ptz_speed_tilt": 1.0,    # tilt speed multiplier (delta scaling)
    "ptz_speed_zoom": 1.0,    # zoom speed multiplier (delta scaling)
    "ptz_deadzone": 0.05,     # movement deadzone

    # NEW: Independent easing for smooth gimbal feel
    "ptz_easing_pan": 0.10,   # slower pan easing
    "ptz_easing_tilt": 0.15,  # medium tilt easing
    "ptz_easing_zoom": 0.25,   # faster zoom easing

    "manual_movement_percent": 0.02,    # 2% of range per manual step
    "tracking_movement_percent": 0.01,  # 1% of range per tracking step  
    "zoom_movement_percent": 0.05,      # 5% of range per zoom step
    
    # Range-based scaling multipliers for fine-tuning
    "pan_scale_multiplier": 1.0,        # Additional multiplier for pan movements
    "tilt_scale_multiplier": 1.0,       # Additional multiplier for tilt movements
    "zoom_scale_multiplier": 1.0, 
}

# Load or create config
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                cfg = json.load(f)
        except Exception:
            cfg = DEFAULT_CONFIG.copy()
    else:
        cfg = DEFAULT_CONFIG.copy()

    # Merge missing keys from default
    for key in DEFAULT_CONFIG:
        if key not in cfg:
            cfg[key] = DEFAULT_CONFIG[key]
        elif isinstance(DEFAULT_CONFIG[key], dict):
            for subkey in DEFAULT_CONFIG[key]:
                if subkey not in cfg[key]:
                    cfg[key][subkey] = DEFAULT_CONFIG[key][subkey]

    return cfg

CONFIG = load_config()

def save_config():
    """Save the current CONFIG dictionary to the JSON file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(CONFIG, f, indent=2)
    except Exception as e:
        print(f"Error saving config: {e}")
