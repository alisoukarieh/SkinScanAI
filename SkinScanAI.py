import torch
import torch.nn as nn
import torch.nn.functional as F 
from PIL import Image
from torchvision import transforms
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Define the model class (same as during training)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # Adjust based on input size
        self.fc2 = nn.Linear(128, len(label_encoder.classes_))  # Number of classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 2: Load the model
model = torch.load('SkinScanAI.pth')
model.eval()  # Set the model to evaluation mode

# Step 3: Define the same transformations you used during training
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to the size used in training
    transforms.ToTensor(),          # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Same normalization
])

# Step 4: Load and preprocess the image
image_path = './Dataset/Images/ISIC_0024306.jpg'  # Replace with your image path
image = Image.open(image_path)
image = transform(image)
image = image.unsqueeze(0)  # Add batch dimension

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image = image.to(device)
model.to(device)

# Step 5: Perform inference
with torch.no_grad():  # Disable gradient calculations for inference
    output = model(image)

# Step 6: Get the predicted class
_, predicted_class = torch.max(output, 1)

# Step 7: Decode the class label
csv_file = './Dataset/HAM10000_metadata'  # Path to your CSV file with labels
data = pd.read_csv(csv_file)

label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['dx'])  # Assuming 'dx' column contains labels

# Decode the predicted label
label = label_encoder.inverse_transform([predicted_class.item()])
print(f'Predicted label: {label[0]}')
