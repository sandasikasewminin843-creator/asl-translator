import pickle
import cv2
import numpy as np
import os
import urllib.request

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

# -----------------------------
# Load trained Random Forest model
# -----------------------------
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

FEATURE_VECTOR_LENGTH = model.n_features_in_

# -----------------------------
# Labels mapping (A-Z + SPACE)
# -----------------------------
labels_dict = {i: ch for i, ch in enumerate(
    ['A','B','C','D','E','F','G','H','I','J','K','L','M',
     'N','O','P','R','S','T','U','V','W','X','Y','SPACE']
)}

# -----------------------------
# Download the hand landmarker model if not present
# -----------------------------
MODEL_PATH = 'hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Download complete.")

# -----------------------------
# Mediapipe HandLandmarker setup
# -----------------------------
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionTaskRunningMode.VIDEO,
    num_hands=1
)
landmarker = HandLandmarker.create_from_options(options)

# -----------------------------
# Start camera (index 1 = Iriun)
# -----------------------------
cam_index = 1
cap = cv2.VideoCapture(cam_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera opened successfully. Press ESC to quit.")

def get_hand_features(hand_landmarks):
    x_list = [lm.x for lm in hand_landmarks]
    y_list = [lm.y for lm in hand_landmarks]

    data_aux = []
    for x, y in zip(x_list, y_list):
        data_aux.append(x - min(x_list))
        data_aux.append(y - min(y_list))

    if len(data_aux) < FEATURE_VECTOR_LENGTH:
        data_aux += [0] * (FEATURE_VECTOR_LENGTH - len(data_aux))
    else:
        data_aux = data_aux[:FEATURE_VECTOR_LENGTH]

    return data_aux, x_list, y_list

# -----------------------------
# Main loop
# -----------------------------
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("WARNING: Failed to grab frame, retrying...")
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if timestamp_ms <= 0:
        timestamp_ms = frame_idx * 33

    results = landmarker.detect_for_video(mp_image, timestamp_ms)
    frame_idx += 1

    if results.hand_landmarks:
        hand = results.hand_landmarks[0]
        data_aux, x_list, y_list = get_hand_features(hand)

        prediction = model.predict([np.array(data_aux)])
        predicted_character = str(prediction[0])

        # Display SPACE as a visible label on screen
        display_text = '[SPACE]' if predicted_character == 'SPACE' else predicted_character

        H, W, _ = frame.shape
        x1 = int(min(x_list) * W) - 10
        y1 = int(min(y_list) * H) - 10
        x2 = int(max(x_list) * W) + 10
        y2 = int(max(y_list) * H) + 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
        cv2.putText(frame, display_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('ASL Detector - Phone Camera', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
