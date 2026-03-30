import pickle
import numpy as np
import os
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS

app = Flask(__name__, static_folder='.')
CORS(app, origins=[
    "https://sandasikasewminin843-creator.github.io",
    "http://localhost:5000",
    "http://127.0.0.1:5000"
], supports_credentials=True)

# ── Load model ──
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
FEATURE_LEN = model.n_features_in_

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receive hand landmarks from browser MediaPipe JS,
    return predicted letter. No image processing needed.
    """
    data = request.json
    if not data or 'landmarks' not in data:
        return jsonify({'letter': ''})
    try:
        landmarks = data['landmarks']  # list of {x, y} dicts
        x_list = [lm['x'] for lm in landmarks]
        y_list = [lm['y'] for lm in landmarks]

        aux = []
        for x, y in zip(x_list, y_list):
            aux.append(x - min(x_list))
            aux.append(y - min(y_list))

        if len(aux) < FEATURE_LEN:
            aux += [0] * (FEATURE_LEN - len(aux))
        aux = aux[:FEATURE_LEN]

        pred = model.predict([np.array(aux)])
        letter = str(pred[0])
        letter = ' ' if letter == 'SPACE' else letter
        return jsonify({'letter': letter})
    except Exception as e:
        print(f"predict error: {e}")
        return jsonify({'letter': '', 'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
