import pickle
import cv2
import numpy as np
import os
import urllib.request
import time
from flask import Flask, Response, jsonify, send_from_directory, request
from flask_cors import CORS

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

app = Flask(__name__, static_folder='.')
CORS(app)

# -----------------------------
# Load trained Random Forest model
# -----------------------------
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
FEATURE_VECTOR_LENGTH = model.n_features_in_

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
# Mediapipe — VIDEO mode for live stream
# -----------------------------
video_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionTaskRunningMode.VIDEO,
    num_hands=1
)
landmarker = HandLandmarker.create_from_options(video_options)

# Mediapipe — IMAGE mode for uploaded images
image_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionTaskRunningMode.IMAGE,
    num_hands=1
)
image_landmarker = HandLandmarker.create_from_options(image_options)

# -----------------------------
# Camera state
# -----------------------------
cap = None
camera_active = False
current_cam_index = -1
frame_idx = 0
current_prediction = {'letter': ''}

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

def open_camera(index):
    global cap, camera_active, current_cam_index, frame_idx
    if cap is not None:
        cap.release()
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if cap.isOpened():
        camera_active = True
        current_cam_index = index
        frame_idx = 0
        return True
    return False

def scan_cameras():
    available = []
    for i in range(6):
        c = cv2.VideoCapture(i)
        if c.isOpened():
            ret, _ = c.read()
            c.release()
            if ret:
                available.append(i)
    return available

def generate_frames():
    global frame_idx, current_prediction, camera_active
    while True:
        if not camera_active or cap is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, 'Camera Off', (200, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (160, 145, 110), 2)
            _, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int(time.monotonic() * 1000)

        results = landmarker.detect_for_video(mp_image, timestamp_ms)
        frame_idx += 1

        if results.hand_landmarks:
            hand = results.hand_landmarks[0]
            data_aux, x_list, y_list = get_hand_features(hand)

            prediction = model.predict([np.array(data_aux)])
            predicted_character = str(prediction[0])

            current_prediction['letter'] = ' ' if predicted_character == 'SPACE' else predicted_character
            display_text = '[SPACE]' if predicted_character == 'SPACE' else predicted_character

            H, W, _ = frame.shape
            x1 = int(min(x_list) * W) - 10
            y1 = int(min(y_list) * H) - 10
            x2 = int(max(x_list) * W) + 10
            y2 = int(max(y_list) * H) + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (99, 180, 140), 3)
            cv2.putText(frame, display_text, (x1, y1 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (99, 180, 140), 3, cv2.LINE_AA)
        else:
            current_prediction['letter'] = ''

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction')
def prediction():
    return jsonify(current_prediction)

@app.route('/cameras')
def cameras():
    return jsonify({'cameras': scan_cameras()})

@app.route('/camera/on', methods=['POST'])
def camera_on():
    data = request.json or {}
    index = int(data.get('index', 0))
    success = open_camera(index)
    return jsonify({'success': success, 'index': index})

@app.route('/camera/off', methods=['POST'])
def camera_off():
    global camera_active, cap
    camera_active = False
    if cap:
        cap.release()
        cap = None
    current_prediction['letter'] = ''
    return jsonify({'success': True})

@app.route('/predict_image', methods=['POST'])
def predict_image():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image provided'}), 400

    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Could not decode image'}), 400

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = image_landmarker.detect(mp_image)

    if not results.hand_landmarks:
        return jsonify({'letter': None, 'message': 'No hand detected in image'})

    hand = results.hand_landmarks[0]
    data_aux, _, _ = get_hand_features(hand)
    prediction = model.predict([np.array(data_aux)])
    predicted_character = str(prediction[0])

    display = '[SPACE]' if predicted_character == 'SPACE' else predicted_character
    letter = ' ' if predicted_character == 'SPACE' else predicted_character
    return jsonify({'letter': letter, 'display': display})

if __name__ == '__main__':
    print("Starting ASL Web App at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
