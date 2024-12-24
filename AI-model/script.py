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
    
def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)  
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_accuracy = correct / total
    return val_loss / len(dataloader), val_accuracy


def test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)  
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())   
            all_labels.extend(labels.cpu().numpy())  
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    test_accuracy = correct / total
    return test_accuracy, all_preds, all_labels


def train_model(num_epochs=10):
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

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
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)
    model = model.to(device)  # Move model to device
    
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
            images, labels = images.to(device), labels.to(device)  # Move to device
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Val Acc: {val_accuracy:.4f}')
    
    # Testing
    test_accuracy, all_preds, all_labels = test(model, test_loader, device)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    
    # Save model
    if not os.path.exists('models'):
        os.makedirs('models')
    model_filename = f'models/model_acc_{test_accuracy:.2f}.pth'
    torch.save(model.state_dict(), model_filename)
    print(f'Model saved to {model_filename}')
    
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
    sns.heatmap(cm, annot=True, fmt='d')
    plt.savefig('logs/confusion_matrix.png')
    plt.close()

def compare_models(num_epochs=5):
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Define transformations (reuse if already defined)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load dataset (reuse your existing SkinTumorDataset class)
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

    # Define models to compare
    model_names = ['densenet121', 'resnet18', 'alexnet', 'resnet50', 'vgg16']
    results = []

    # Ensure logs and models directories exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('models'):
        os.makedirs('models')  

    for model_name in model_names:
        print(f'\nTraining model: {model_name}')
        # Initialize model
        if model_name == 'densenet121':
            model = models.densenet121(pretrained=True)
            num_features = model.classifier.in_features
            model.classifier = nn.Linear(num_features, 2)
        elif model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 2)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 2)
        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            num_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_features, 2)
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            num_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_features, 2)
        else:
            print(f"Model {model_name} is not supported.")
            continue

        model = model.to(device)

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
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)

            # Validation using updated validate function
            val_loss, val_accuracy = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

        # Testing using updated test function
        test_accuracy, all_preds, all_labels = test(model, test_loader, device)
        print(f'Test Accuracy for {model_name}: {test_accuracy:.4f}')

        # Save the trained model with accuracy in the filename
        model_filename = f'models/{model_name}_acc_{test_accuracy:.2f}.pth'
        torch.save(model.state_dict(), model_filename)
        print(f'Model saved to {model_filename}')

        # Save loss plots
        plt.figure()
        plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
        plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
        plt.title(f'Loss Plot for {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        loss_plot_path = f'logs/{model_name}_loss_plot.png'
        plt.savefig(loss_plot_path)
        plt.close()

        # Save confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Cancer', 'Cancer'], yticklabels=['Not Cancer', 'Cancer'])
        plt.title(f'Confusion Matrix for {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        cm_plot_path = f'logs/{model_name}_confusion_matrix.png'
        plt.savefig(cm_plot_path)
        plt.close()

        # Append results
        results.append({
            'Model': model_name,
            'Test Accuracy': test_accuracy,
            'Final Train Loss': train_losses[-1],
            'Final Val Loss': val_losses[-1],
            'Final Val Accuracy': val_accuracies[-1],
            'Loss Plot': loss_plot_path,
            'Confusion Matrix': cm_plot_path,
            'Model Path': model_filename  # Include model path in the results
        })

    # Create comparison DataFrame
    results_df = pd.DataFrame(results)
    comparison_csv_path = 'logs/model_comparison.csv'
    results_df.to_csv(comparison_csv_path, index=False)
    print(f'\nModel comparison saved to {comparison_csv_path}')

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

# train_model(num_epochs=10)
# cancer_labels = ['mel', 'bcc', 'akiec']
# predict('test_images/test.jpg', 'model_acc_0.87.pth')
compare_models(num_epochs=3)
