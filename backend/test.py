import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

def predict(image_path, model_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    model = models.densenet121()
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        _, prediction = torch.max(output, 1)
    classes = ['Not Cancer', 'Cancer']
    predicted_class = classes[prediction.item()]
    probability = probabilities[0][prediction.item()].item()
    print(f'Prediction: {predicted_class}, Probability: {probability:.4f}')

predict('akiec.jpg', 'model_acc_0.87.pth')