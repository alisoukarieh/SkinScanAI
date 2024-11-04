from flask import Flask, request, jsonify
import sys
import os
import torch
from fastai.vision.all import *
from fastai.data.block import ColReader
from pathlib import Path
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Load the model when the app starts
get_y = ColReader('dx')
learner = load_learner('./SkinScanAI_DensNet121_88.22.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename provided.'}), 400
    try:
        # Read the image file
        img = PILImage.create(file.stream)
        # Make prediction
        pred, pred_idx, probs = learner.predict(img)
        # Prepare response
        response = {
            'prediction': str(pred),
            'probability': float(probs[pred_idx]),
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the app on host '0.0.0.0' to make it accessible over the network if needed
    app.run(host='0.0.0.0', port=5000, debug=True)
