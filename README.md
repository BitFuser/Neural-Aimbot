# 🎯 Neural Aimbot (`aiaim.py`)

An advanced Python-based neural network aimbot that uses object detection to identify targets on screen and assist with mouse aiming. Built using PyTorch, OpenCV, and YOLOv5 for high-speed inference and precision.

> ⚠️ **For educational and research purposes only.** Unauthorized use in online games may violate terms of service and lead to bans or legal consequences.

---

## 🚀 Features

- 🧠 **Neural Network Target Detection** (YOLOv5)
- 🖱️ **Automatic Mouse Movement Toward Targets**
- 🎯 **Target Lock Visualization with Overlay**
- 📷 **Optional Data Collection Mode** (`collect_data`)
- 🔄 **Realtime Toggle**:  
  - `F1` to enable/disable aimbot  
  - `F2` to quit instantly
- ⚡ CUDA acceleration support (if available)

---

## 🧱 Requirements

- Python 3.8+
- Windows OS
- CUDA-enabled GPU (optional but recommended)
- Dependencies:
  - `torch`
  - `opencv-python`
  - `mss`
  - `numpy`
  - `pynput`
  - `termcolor`
  - `pywin32`
  - `ultralytics` (for loading YOLOv5 models)

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 📁 File Structure

```
.
├── aiaim.py             # Main aimbot logic
├── yolov5x.pt           # YOLOv5 trained model file
├── config.json          # Sensitivity configuration
├── data/                # (Optional) screenshots if collecting data
```

---

## ⚙️ Initial Setup

On first run, if `config.json` is missing, you’ll be prompted to input:

- X/Y sensitivity
- Targeting sensitivity

These values are used to tune the mouse movement for your game.

---

## 🧠 How It Works

1. Captures a square portion of the screen centered on the crosshair.
2. Runs object detection using YOLOv5.
3. Identifies and targets the closest object to the center.
4. Moves the mouse toward the detected head position, simulating subtle, human-like aiming.

---

## 🖥️ Controls

| Key         | Function              |
|-------------|------------------------|
| `F1`        | Toggle aimbot on/off   |
| `F2`        | Exit script gracefully |
| `Q` (in window) | Quit OpenCV window      |

---

## 📸 Data Collection Mode

To capture screenshots of frames with visible targets (for retraining):

```bash
python aiaim.py collect_data
```

Images will be saved to the `data/` directory.

---

## 🛑 Disclaimer

This tool is **strictly for educational and research purposes** in areas such as:

- Computer vision
- Real-time object detection
- Mouse control automation

**Do not use in online multiplayer games** or violate any game’s terms of service.
