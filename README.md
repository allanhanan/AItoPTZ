# AItoPTZ

AItoPTZ is a PTZ (Pan-Tilt-Zoom) camera controller app for Linux, with AI-based tracking as a key feature. It provides manual and automated control over UCV PTZ cameras, using YOLO models for person and face detection to track subjects.

This app is currently in early development (v0.1.0 beta) and is Linux-only. Contributions are welcome to expand features, add Windows support(coming soon), improve stability, or fix bugs. See the Contributing section below for details.

<img width="600" height=auto alt="image" src="https://github.com/user-attachments/assets/b5ab6252-27c8-4c63-a28c-08d5e4a61eb3" />


## Features

- **PTZ Control**: Uses v4l2_ctl for smooth pan, tilt, and zoom with easing and adaptive scaling based on camera hardware.
- **GUI Interface**: PyQt5-based UI for camera selection, manual controls, preset management, and visual feedback.
- **Input Controls**: Keyboard shortcuts, mouse drag for pan/tilt, wheel for zoom, and click to interact.
- **Presets**: Save and recall positions via buttons or number keys.
- **AI Tracking**: Uses YOLOv8 for person and face detection with modes for full body, torso, or face focus. Enables automated tracking while maintaining manual override.
- **Configurable**: config.py and JSON file for keybindings, easing parameters, model paths, and more.


## Requirements

- Python 3.8 or higher
- Linux OS with V4L2 support (tested on Ubuntu)
- PTZ camera compatible with V4L2 (e.g., USB PTZ webcams)
- Dependencies: PyQt5, opencv-python, ultralytics, qdarkstyle
- For AI features: YOLO models yolov8n.pt (person) and yolov8n-face.pt (face) – download from Ultralytics and place in project root or update config.json.

## Installation

1. Clone the repository:
   ```bash
   git clone <REPO_URL>
   cd AItoPTZ
   ```

2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install PyQt5 opencv-python ultralytics qdarkstyle
   ```

4. For AI tracking: Download YOLO models and edit `config.json` if needed.

## Usage

Run the app:
```bash
python main.py
```

The app launches a GUI with video feed and controls.

### GUI Controls

- **Camera Selection**: Dropdown to choose V4L2 device.
- **Pan/Tilt Buttons**: "Tilt ▲", "◀ Pan", "Home", "Pan ▶", "Tilt ▼" for manual movement. "Home" resets position.
- **Zoom Buttons**: "Zoom +" and "Zoom -" for zoom adjustment.
- **Selection Buttons**: "◀ Select" and "Select ▶" cycle through detected objects (if AI enabled), "Track ▶" starts automated tracking.
- **Mode Dropdown**: "Face", "Torso", "Full Body" for AI tracking modes.
- **Preset Management**: "+ Add Preset" creates named buttons. "Set Preset" overwrites existing with current position. Click button to recall (or set in set mode). "✕" deletes.

### Mouse Controls

- **Right-Click**: Select object for tracking (if AI enabled).
- **Left-Click Drag**: Pan (horizontal) or tilt (vertical).
- **Wheel**: Zoom in/out.

### Keyboard Controls

Configurable via Settings > Keybindings (defaults in config.json):
- Pan Left/Right, Tilt Up/Down, Zoom In/Out: Manual movement (hold for acceleration).
- Cycle Left/Right, Track, Home: Object selection and tracking commands.
- Numbers 1-9: Recall presets.

The app is beta, logs to console for debugging.

## Configuration

Edit `config.json` for:  (mostly automatic from config.py)
- `"camera_index"`: Default device.
- `"keybinds"`: Action-to-key mappings.
- `"ptz_easing_*"`: Movement smoothing (0.0-1.0).
- `"yolo_model"` and `"face_model"`: Paths for AI.
- `"presets"` and `"preset_names"`: Stored positions (GUI-managed).

## Project Structure

- `app/camera_feed.py`: Camera capture thread.
- `app/config.py`: JSON config handling.
- `app/controller.py`: Tracking logic and PTZ integration.
- `app/gui.py`: GUI components and events.
- `app/ptz_control.py`: V4L2 PTZ interface.
- `app/vision.py`: YOLO detection.
- `main.py`: Entry point.
- `config.json`: Config file.

## Development Status

v0.1.0 beta – core PTZ control works, AI tracking is functional but needs tuning. Limitations: Linux-only, others I havent really checked ngl theres a lot.

Future plans: Windows support, CLI mode, advanced AI options.

## Contributing

Contributions welcome for any improvements.

Areas: Platform support, bug fixes, UI enhancements, AI improvements.

## License

GNU GENERAL PUBLIC LICENSE Version 3
## Version

v0.1.0 beta
