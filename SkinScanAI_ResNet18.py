import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
import os
from PIL import Image
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# =========================
# Data Preparation Classes
# =========================

class SkinLesionDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_mapping, sex_mapping, localization_mapping, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Use provided mappings
        self.label_mapping = label_mapping
        self.annotations['label'] = self.annotations['dx'].map(self.label_mapping)

        self.sex_mapping = sex_mapping
        self.localization_mapping = localization_mapping

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index]['image_id']
        img_path = os.path.join(self.img_dir, img_id + '.jpg')  # Adjust extension if needed

        image = Image.open(img_path).convert('RGB')
        label = self.annotations.iloc[index]['label']

        if self.transform:
            image = self.transform(image)

        # Get metadata
        age = self.annotations.iloc[index]['age']
        sex = self.annotations.iloc[index]['sex']
        localization = self.annotations.iloc[index]['localization']

        # Encode metadata
        sex_encoded = self.sex_mapping.get(str(sex).lower(), -1)  # -1 for unknown
        localization_encoded = self.localization_mapping.get(str(localization).lower(), -1)

        # Handle missing age
        if pd.isnull(age):
            age = -1

        metadata = torch.tensor([age, sex_encoded, localization_encoded], dtype=torch.float)

        return image, metadata, label

# =========================
# Model Definition
# =========================

class CombinedModel(nn.Module):
    def __init__(self, num_classes):
        super(CombinedModel, self).__init__()
        self.image_model = models.resnet18(pretrained=True)

        # Modify the last layer to output features instead of classification
        num_ftrs = self.image_model.fc.in_features
        self.image_model.fc = nn.Identity()

        # Metadata features
        self.metadata_fc = nn.Sequential(
            nn.Linear(3, 64),  # Assuming 3 metadata features: age, sex, localization
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Combine image and metadata features
        self.classifier = nn.Linear(num_ftrs + 64, num_classes)

    def forward(self, x_image, x_metadata):
        x_image = self.image_model(x_image) * 1
        x_metadata = self.metadata_fc(x_metadata) * 0.65

        # Concatenate image and metadata features
        x = torch.cat((x_image, x_metadata), dim=1)
        x = self.classifier(x)
        return x

# =========================
# Training Function
# =========================

def train(csv_file, img_dir, num_epochs=10, batch_size=32):
    # Read the CSV file
    annotations = pd.read_csv(csv_file)

    # Map labels to integers
    label_mapping = {label: idx for idx, label in enumerate(annotations['dx'].unique())}

    # Map categorical metadata to numerical values
    sex_mapping = {'male': 0, 'female': 1}
    localization_mapping = {loc.lower(): idx for idx, loc in enumerate(annotations['localization'].unique())}

    # Save mappings for later use
    with open('label_mapping.pkl', 'wb') as f:
        pickle.dump(label_mapping, f)
    with open('sex_mapping.pkl', 'wb') as f:
        pickle.dump(sex_mapping, f)
    with open('localization_mapping.pkl', 'wb') as f:
        pickle.dump(localization_mapping, f)

    # Split the data into training and validation sets
    train_df, val_df = train_test_split(annotations, test_size=0.2, random_state=42, stratify=annotations['dx'])

    # Save train and validation dataframes to CSV
    train_df.to_csv('train.csv', index=False)
    val_df.to_csv('val.csv', index=False)

    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create datasets
    train_dataset = SkinLesionDataset(
        csv_file='train.csv',
        img_dir=img_dir,
        label_mapping=label_mapping,
        sex_mapping=sex_mapping,
        localization_mapping=localization_mapping,
        transform=train_transform
    )

    val_dataset = SkinLesionDataset(
        csv_file='val.csv',
        img_dir=img_dir,
        label_mapping=label_mapping,
        sex_mapping=sex_mapping,
        localization_mapping=localization_mapping,
        transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    num_classes = len(label_mapping)
    model = CombinedModel(num_classes)

    # Check for available device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0

        for images, metadata, labels in train_loader:
            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device).long()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct.double() / len(train_loader.dataset)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for images, metadata, labels in val_loader:
                images = images.to(device)
                metadata = metadata.to(device)
                labels = labels.to(device).long()

                outputs = model(images, metadata)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_correct.double() / len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {epoch_loss:.4f}  Train Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'SkinScanAI_ResNet18.pth')
    print('Training complete. Model saved as skin_lesion_model.pth.')

# =========================
# Prediction Function
# =========================

def predict_image(image_path, age, sex, localization):
    # Load the label and metadata mappings
    with open('label_mapping.pkl', 'rb') as f:
        label_mapping = pickle.load(f)
    idx_to_label = {idx: label for label, idx in label_mapping.items()}

    with open('sex_mapping.pkl', 'rb') as f:
        sex_mapping = pickle.load(f)

    with open('localization_mapping.pkl', 'rb') as f:
        localization_mapping = pickle.load(f)

    num_classes = len(label_mapping)
    model = CombinedModel(num_classes)

    # Load the trained model weights
    model.load_state_dict(torch.load('SkinScanAI_ResNet18.pth', map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode

    # Define the transformation (same as validation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Encode metadata
    sex_encoded = sex_mapping.get(str(sex).lower(), -1)  # -1 for unknown
    localization_encoded = localization_mapping.get(str(localization).lower(), -1)
    if age is None or pd.isnull(age):
        age = -1
    metadata = torch.tensor([[age, sex_encoded, localization_encoded]], dtype=torch.float)

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Forward pass
    with torch.no_grad():
        outputs = model(image, metadata)
        # Apply softmax to get probabilities
        probabilities = torch.softmax(outputs, dim=1)
        # Get the predicted class and its probability
        predicted_prob, predicted = torch.max(probabilities, 1)
        class_index = predicted.item()
        class_label = idx_to_label.get(class_index, 'Unknown')
        confidence = predicted_prob.item()

    print(f'Predicted Tumor Type: {class_label}, Confidence: {confidence:.4f}')
    return class_label, confidence


#train("Images/Dataset/HAM10000_metadata", "Images/Dataset/Images", num_epochs=30, batch_size=32)
#predict_image("Images/test_images/test.jpg", 60, "male", "back")
predict_image("Images/test_images/akiec.jpg", 59, "male", "hand")

