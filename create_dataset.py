import os
import pickle
import urllib.request
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

# -----------------------------
# Download hand landmarker model if not present
# -----------------------------
MODEL_PATH = 'hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Download complete.")

# -----------------------------
# Setup HandLandmarker (IMAGE mode for static images)
# -----------------------------
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionTaskRunningMode.IMAGE,
    num_hands=1
)
landmarker = HandLandmarker.create_from_options(options)

DATA_DIR = './data'

data = []
labels = []
skipped = 0

all_dirs = sorted(os.listdir(DATA_DIR))
print(f"Found classes: {all_dirs}\n")

for dir_ in all_dirs:
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue

    img_files = os.listdir(dir_path)
    print(f"Processing class '{dir_}' — {len(img_files)} images...")

    for img_path in img_files:
        full_path = os.path.join(dir_path, img_path)

        img = cv2.imread(full_path)
        if img is None:
            skipped += 1
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        results = landmarker.detect(mp_image)

        if results.hand_landmarks:
            hand = results.hand_landmarks[0]

            x_list = [lm.x for lm in hand]
            y_list = [lm.y for lm in hand]

            data_aux = []
            for x, y in zip(x_list, y_list):
                data_aux.append(x - min(x_list))
                data_aux.append(y - min(y_list))

            data.append(data_aux)
            labels.append(dir_)
        else:
            skipped += 1

print(f"\nDone! Total samples: {len(data)}, Skipped (no hand detected): {skipped}")

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Saved to data.pickle")