import os
import sys
import json
import time
import uuid
import math
import ctypes
import threading
import numpy as np
import cv2
import mss
import torch
import win32api
from pynput import keyboard
from termcolor import colored
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# Paths for config and model file in current folder
CONFIG_PATH = BASE_DIR / "config.json"
MODEL_PATH = BASE_DIR / "yolov5x.pt"
DATA_DIR = BASE_DIR / "data"



SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
CENTER_X = SCREEN_WIDTH // 2
CENTER_Y = SCREEN_HEIGHT // 2

# Input structures for mouse control
PUL = ctypes.POINTER(ctypes.c_ulong)

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("mi", MouseInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Aimbot class
class NeuralAimbot:
    def __init__(self, collect_data=False, mouse_delay=0.0001, debug=False):
        self.collect_data = collect_data
        self.mouse_delay = mouse_delay
        self.debug = debug
        self.active = True
        self.screen = mss.mss()
        self.box_size = 500
        self.extra = ctypes.c_ulong(0)
        self.ii_ = Input_I()
        self.pixel_increment = 1

        # Load sensitivity settings
        with open(CONFIG_PATH) as f:
            self.sens_config = json.load(f)

        # Load model
        print("[INFO] Loading neural network model...")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=False)
        self.model.conf = 0.77
        self.model.iou = 0.77

        if torch.cuda.is_available():
            print(colored("[INFO] CUDA acceleration enabled", "green"))
        else:
            print(colored("[WARNING] CUDA acceleration unavailable. Performance may be degraded.", "red"))

        print("\n[INFO] Press 'F1' to toggle aimbot\n[INFO] Press 'F2' to quit")

    def toggle_active(self):
        self.active = not self.active
        status = "ENABLED" if self.active else "DISABLED"
        color = "green" if self.active else "red"
        print(f"[INFO] Aimbot is now {colored(status, color)}")

    def is_targeting(self):
        return win32api.GetKeyState(0x02) < 0

    def is_target_locked(self, x, y):
        threshold = 5
        return (CENTER_X - threshold <= x <= CENTER_X + threshold and
                CENTER_Y - threshold <= y <= CENTER_Y + threshold)

    def move_mouse(self, x, y):
        if not self.is_targeting():
            return

        scale = self.sens_config["targeting_scale"]
        dx = (x - CENTER_X) * scale / self.pixel_increment
        dy = (y - CENTER_Y) * scale / self.pixel_increment
        length = int(math.hypot(dx, dy))
        if length == 0:
            return

        unit_dx = dx / length * self.pixel_increment
        unit_dy = dy / length * self.pixel_increment
        sum_dx = sum_dy = 0

        for i in range(length):
            move_x = round(unit_dx * i - sum_dx)
            move_y = round(unit_dy * i - sum_dy)
            sum_dx += move_x
            sum_dy += move_y
            self.ii_.mi = MouseInput(move_x, move_y, 0, 0x0001, 0, ctypes.pointer(self.extra))
            input_obj = Input(ctypes.c_ulong(0), self.ii_)
            ctypes.windll.user32.SendInput(1, ctypes.byref(input_obj), ctypes.sizeof(input_obj))
            time.sleep(self.mouse_delay)

    def run(self):
        print("[INFO] Starting screen capture...")
        half_width = SCREEN_WIDTH // 2
        half_height = SCREEN_HEIGHT // 2
        detection_box = {'left': half_width - self.box_size // 2,
                         'top': half_height - self.box_size // 2,
                         'width': self.box_size,
                         'height': self.box_size}
        collect_timer = 0

        while True:
            start_time = time.perf_counter()
            frame = np.array(self.screen.grab(detection_box))
            if self.collect_data:
                orig_frame = frame.copy()
            results = self.model(frame)

            if results.xyxy[0].size(0) > 0:
                closest = None
                min_dist = float('inf')
                for *box, conf, cls in results.xyxy[0]:
                    x1, y1, x2, y2 = map(int, box)
                    height = y2 - y1
                    head_x = (x1 + x2) // 2
                    head_y = int((y1 + y2) / 2 - height / 2.7)
                    crosshair_dist = math.hypot(head_x - self.box_size // 2, head_y - self.box_size // 2)

                    if crosshair_dist < min_dist:
                        min_dist = crosshair_dist
                        closest = (head_x, head_y, x1, y1, x2, y2, conf.item())

                if closest:
                    head_x, head_y, x1, y1, x2, y2, conf = closest
                    abs_x = head_x + detection_box['left']
                    abs_y = head_y + detection_box['top']
                    cv2.circle(frame, (head_x, head_y), 5, (0, 255, 0), -1)
                    cv2.line(frame, (head_x, head_y), (self.box_size // 2, self.box_size // 2), (0, 255, 255), 2)
                    label = "LOCKED" if self.is_target_locked(abs_x, abs_y) else "TARGETING"
                    cv2.putText(frame, label, (x1 + 40, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0) if label == "LOCKED" else (255, 0, 0), 2)
                    if self.active:
                        self.move_mouse(abs_x, abs_y)

            if self.collect_data and time.perf_counter() - collect_timer > 1 and self.is_targeting() and self.active:
                cv2.imwrite(f"{DATA_DIR}/{uuid.uuid4()}.jpg", orig_frame)
                collect_timer = time.perf_counter()

            fps = int(1 / (time.perf_counter() - start_time))
            cv2.putText(frame, f"FPS: {fps}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow("Aimbot Output", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.screen.close()
        cv2.destroyAllWindows()

# Setup function
def setup():
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    print("[INFO] In-game X and Y axis sensitivity should be the same.")

    def prompt(message):
        while True:
            try:
                return float(input(message))
            except ValueError:
                print("[ERROR] Invalid input. Please enter a numeric value.")

    xy_sens = prompt("Enter X and Y axis sensitivity: ")
    targeting_sens = prompt("Enter targeting sensitivity: ")

    settings = {
        "xy_sens": xy_sens,
        "targeting_sens": targeting_sens,
        "xy_scale": 10 / xy_sens,
        "targeting_scale": 1000 / (targeting_sens * xy_sens)
    }

    with open(CONFIG_PATH, 'w') as f:
        json.dump(settings, f)
    print("[INFO] Sensitivity configuration saved.")

# Main function
def main():
    if not os.path.exists(CONFIG_PATH):
        print("[INFO] Sensitivity configuration not found. Running setup...")
        setup()
    if "collect_data" in sys.argv and not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    aimbot = NeuralAimbot(collect_data="collect_data" in sys.argv)

    def on_release(key):
        if key == keyboard.Key.f1:
            aimbot.toggle_active()
        elif key == keyboard.Key.f2:
            print("\n[INFO] Exiting...")
            os._exit(0)

    listener = keyboard.Listener(on_release=on_release)
    listener.start()
    aimbot.run()

if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    main()
