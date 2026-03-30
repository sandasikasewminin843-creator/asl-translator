import pickle
import cv2
import numpy as np
import os
import urllib.request
import base64
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS

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

# ── MediaPipe hands (legacy solutions API) ──
import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

def get_features(hand_landmarks):
    x = [lm.x for lm in hand_landmarks.landmark]
    y = [lm.y for lm in hand_landmarks.landmark]
    aux = []
    for xi, yi in zip(x, y):
        aux.append(xi - min(x))
        aux.append(yi - min(y))
    if len(aux) < FEATURE_LEN:
        aux += [0] * (FEATURE_LEN - len(aux))
    return aux[:FEATURE_LEN]

def predict_from_img(img):
    """Run hand detection and return predicted letter or empty string."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        return ''
    hand = results.multi_hand_landmarks[0]
    feats = get_features(hand)
    pred = model.predict([np.array(feats)])
    letter = str(pred[0])
    return ' ' if letter == 'SPACE' else letter

# ── Routes ──
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    data = request.json
    if not data or 'frame' not in data:
        return jsonify({'letter': ''})
    try:
        raw = data['frame']
        img_data = raw.split(',')[1] if ',' in raw else raw
        img_bytes = base64.b64decode(img_data)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'letter': ''})
        letter = predict_from_img(img)
        return jsonify({'letter': letter})
    except Exception as e:
        print(f"predict_frame error: {e}")
        return jsonify({'letter': '', 'error': str(e)})

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
        letter = predict_from_img(img)
        if not letter:
            return jsonify({'letter': None, 'message': 'No hand detected'})
        display = '[SPACE]' if letter == ' ' else letter
        return jsonify({'letter': letter, 'display': display})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
