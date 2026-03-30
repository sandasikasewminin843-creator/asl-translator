import pickle
import cv2
import numpy as np
import os
import urllib.request
import base64
from flask import Flask, Response, jsonify, send_from_directory, request
from flask_cors import CORS

# Tell MediaPipe to use CPU only, no OpenGL needed
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

app = Flask(__name__, static_folder='.')
CORS(app, origins=[
    "https://sandasikasewminin843-creator.github.io",
    "http://localhost:5000",
    "http://127.0.0.1:5000"
], supports_credentials=True)

# ── Model ──
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
FEATURE_LEN = model.n_features_in_

# ── Download hand landmarker if needed ──
MODEL_PATH = 'hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Done.")

# ── Lazy initialization of MediaPipe ──
_image_landmarker = None

def get_landmarker():
    global _image_landmarker
    if _image_landmarker is None:
        image_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_hands=1
        )
        _image_landmarker = HandLandmarker.create_from_options(image_options)
    return _image_landmarker

def get_features(hand):
    x = [lm.x for lm in hand]
    y = [lm.y for lm in hand]
    aux = []
    for xi, yi in zip(x, y):
        aux.append(xi - min(x))
        aux.append(yi - min(y))
    if len(aux) < FEATURE_LEN:
        aux += [0] * (FEATURE_LEN - len(aux))
    return aux[:FEATURE_LEN]

# ── Routes ──
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    data = request.json
    if not data or 'frame' not in data:
        return jsonify({'error': 'No frame provided'}), 400
    try:
        img_data = data['frame'].split(',')[1] if ',' in data['frame'] else data['frame']
        img_bytes = base64.b64decode(img_data)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'letter': ''})

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = get_landmarker().detect(mp_image)

        if not results.hand_landmarks:
            return jsonify({'letter': ''})

        hand = results.hand_landmarks[0]
        feats = get_features(hand)
        pred = model.predict([np.array(feats)])
        letter = str(pred[0])
        letter = ' ' if letter == 'SPACE' else letter
        return jsonify({'letter': letter})
    except Exception as e:
        return jsonify({'error': str(e), 'letter': ''}), 500

@app.route('/predict_image', methods=['POST'])
def predict_image():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image provided'}), 400
    try:
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Could not decode image'}), 400

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = get_landmarker().detect(mp_image)

        if not results.hand_landmarks:
            return jsonify({'letter': None, 'message': 'No hand detected in image'})

        hand = results.hand_landmarks[0]
        feats = get_features(hand)
        pred = model.predict([np.array(feats)])
        letter = str(pred[0])
        display = '[SPACE]' if letter == 'SPACE' else letter
        actual  = ' '      if letter == 'SPACE' else letter
        return jsonify({'letter': actual, 'display': display})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
