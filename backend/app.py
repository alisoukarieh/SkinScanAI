from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import PurePath
import os
import torch

from torch import nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)
CORS(app) 

# Define the transformer once
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the model once
model_path = './model_acc_0.87.pth'
model = models.densenet121()
model.classifier = nn.Linear(model.classifier.in_features, 2)
model.load_state_dict(torch.load(model_path))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename provided.'}), 400
    try:
        image = Image.open(file).convert('RGB')
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            _, prediction = torch.max(output, 1)
        classes = ['Not Cancer', 'Cancer']
        predicted_class = classes[prediction.item()]
        return jsonify({'prediction': predicted_class}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)