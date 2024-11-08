import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

class SkinTumorDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        # Map dx labels to cancer (1) or not cancer (0)
        cancer_labels = ['mel', 'bcc', 'akiec']
        self.annotations['label'] = self.annotations['dx'].apply(lambda x: 1 if x in cancer_labels else 0)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx]['image_id'] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = self.annotations.iloc[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label
    
def validate(model, dataloader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_accuracy = correct / total
    return val_loss / len(dataloader), val_accuracy

def validate(model, dataloader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_accuracy = correct / total
    return val_loss / len(dataloader), val_accuracy

def test(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_accuracy = correct / total
    return test_accuracy, all_preds, all_labels

def train_model(num_epochs=10):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load dataset
    dataset = SkinTumorDataset(csv_file='Dataset/HAM10000_metadata.csv', root_dir='Dataset/Images', transform=transform)
    
    # Split dataset
    train_size = int(0.75 * len(dataset))
    val_size = int(0.20 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Handle class imbalance on the training data
    train_labels = [label for _, label in train_dataset]
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for _, label in train_dataset]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load model
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))
        
        # Validation
        val_loss, val_accuracy = validate(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_accuracy:.4f}')
    
    # Testing
    test_accuracy, all_preds, all_labels = test(model, test_loader)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    
    # Save model
    model_filename = f'model_acc_{test_accuracy:.2f}.pth'
    torch.save(model.state_dict(), model_filename)
    
    # Save logs
    if not os.path.exists('logs'):
        os.makedirs('logs')
    # Loss plot
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.savefig('logs/loss_plot.png')
    plt.close()
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True)
    plt.savefig('logs/confusion_matrix.png')
    plt.close()

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

#train_model(num_epochs=10)
# cancer_labels = ['mel', 'bcc', 'akiec']
# predict('test_images/test.jpg', 'model_acc_0.87.pth')