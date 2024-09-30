import torch
from model import CNNModel  # Import the model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load the trained model
model = CNNModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Test Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_dataset = datasets.ImageFolder(root='path_to_test_data', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluation
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_dataloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model: {100 * correct / total}%')
