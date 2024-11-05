from flask import Flask, request, jsonify
import sys
from fastai.vision.all import *
from pathlib import Path, PosixPath
from flask_cors import CORS

import pathlib

# Define a dummy WindowsPath that behaves like PosixPath
class WindowsPath(PosixPath):
    pass

# Assign the dummy WindowsPath to pathlib
pathlib.WindowsPath = WindowsPath

# Define a dummy get_y function that returns None since it's required by fastai but not used
def get_y(x):
    pass 

# Adjust the __main__ module to point to your app module to look for get_y here
sys.modules['__main__'] = sys.modules[__name__]

app = Flask(__name__)
CORS(app) 

learner = load_learner(str(Path('./SkinScanAI_DensNet121_88.22.pkl')))

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename provided.'}), 400
    try:
        img = PILImage.create(file.stream)
        pred, pred_idx, probs = learner.predict(img)
        response = {
            'prediction': str(pred),
            'probability': float(probs[pred_idx]),
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
