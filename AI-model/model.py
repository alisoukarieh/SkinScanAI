import argparse
import pandas as pd
from fastai.vision.all import *
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Process the CSV file with a mapping to 2 main classes
def process_metadata(metadata):
    mapping = {
        'nv': 'healthy',       # Benign nevus
        'mel': 'cancer',       # Melanoma
        'bkl': 'healthy',      # Benign keratosis
        'df': 'healthy',       # Dermatofibroma
        'vasc': 'healthy',     # Benign vascular lesion
        'bcc': 'cancer',       # Basal cell carcinoma
        'akiec': 'cancer',     # Actinic keratosis
        'sain': 'healthy',
    }
    metadata['dx'] = metadata['dx'].map(mapping)
    metadata = metadata[metadata['dx'].notnull()]
    return metadata

# Modified function to plot and save the evolution of losses per epoch
def plot_loss(learn, model_name):
    epochs = range(len(learn.recorder.values))
    train_losses = [x[0] for x in learn.recorder.values]
    valid_losses = [x[1] for x in learn.recorder.values]

    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, valid_losses, label="Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f"Loss over Epochs for {model_name}")
    plt.legend()
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    plt.savefig(logs_dir / f'{model_name}_loss_plot.png')
    plt.close()

# Function to get DataLoaders
def get_data_loaders(metadata_path, images_path, batch_tfms, valid_pct=0.2, subset_size=None, seed=42, num_workers=0):
    metadata = pd.read_csv(metadata_path)
    metadata = process_metadata(metadata)

    # Create 'image_path' column
    metadata['image_path'] = metadata['image_id'].apply(lambda x: images_path / f'{x}.jpg')

    # Remove rows where image file does not exist
    metadata = metadata[metadata['image_path'].apply(lambda x: x.exists())]

    # If subset_size is specified, sample a subset using stratified sampling
    if subset_size is not None and subset_size < len(metadata):
        metadata, _ = train_test_split(
            metadata,
            train_size=subset_size,
            random_state=seed,
            stratify=metadata['dx']
        )
        metadata.reset_index(drop=True, inplace=True)

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=ColReader('image_path'),
        get_y=ColReader('dx'),
        splitter=RandomSplitter(valid_pct=valid_pct, seed=seed),
        item_tfms=Resize(224),
        batch_tfms=batch_tfms
    )

    dls = dblock.dataloaders(metadata, bs=64, num_workers=num_workers)
    return dls, metadata

# Function to get test DataLoader
def get_test_dataloader(metadata_path, images_path, batch_tfms=None, num_workers=0):
    metadata = pd.read_csv(metadata_path)
    metadata = process_metadata(metadata)

    # Create 'image_path' column
    metadata['image_path'] = metadata['image_id'].apply(lambda x: images_path / f'{x}.jpg')

    # Remove rows where image file does not exist
    metadata = metadata[metadata['image_path'].apply(lambda x: x.exists())]

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=ColReader('image_path'),
        get_y=ColReader('dx'),
        splitter=IndexSplitter([]),  # No validation set
        item_tfms=Resize(224),
        batch_tfms=batch_tfms
    )

    dls = dblock.dataloaders(metadata, bs=64, num_workers=num_workers)
    return dls.train

# Function to test predictions and compute accuracy
def test_model(learn):
    test_metadata_path = Path('Images/testing_dataset/testing_dataset_metadata.csv')
    test_images_path = Path('Images/testing_dataset/Images')

    test_dl = get_test_dataloader(test_metadata_path, test_images_path)

    # Get predictions and true labels
    preds, targs = learn.get_preds(dl=test_dl)

    # Convert predictions to class labels
    pred_classes = preds.argmax(dim=1)
    true_classes = targs

    # Get class labels
    classes = learn.dls.vocab

    # Calculate overall accuracy
    accuracy = accuracy_score(true_classes, pred_classes) * 100
    balanced_acc = balanced_accuracy_score(true_classes, pred_classes) * 100
    print(f"Overall Test Accuracy: {accuracy:.2f}%")
    print(f"Balanced Test Accuracy: {balanced_acc:.2f}%")

    # Classification report
    report = classification_report(true_classes, pred_classes, target_names=classes)
    print("\nClassification Report:")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(true_classes, pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(logs_dir / f'confusion_matrix_{timestamp}.png')
    plt.close()

    # Return metrics
    return accuracy, balanced_acc, report

# Function to predict a single image
def predict_image(learn, image_path):
    img = PILImage.create(image_path)
    pred, pred_idx, probs = learn.predict(img)
    print(f"Image: {image_path.name}")
    print(f"Prediction: {pred}")
    print(f"Probability: {probs[pred_idx]:.4f}")

def compute_class_weights(metadata):
    class_counts = metadata['dx'].value_counts()
    total_samples = len(metadata)
    class_weights = total_samples / (len(class_counts) * class_counts)
    # Ensure weights are in the same order as the classes in the DataLoaders
    class_to_idx = {v: k for k, v in enumerate(sorted(class_counts.index))}
    weights = torch.tensor([class_weights[class_name] for class_name in learn.dls.vocab])
    return weights

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    metadata_path = Path('Images/training_dataset/training_dataset_metadata.csv')
    images_path = Path('Images/training_dataset/Images')

    dls, metadata = get_data_loaders(metadata_path, images_path, batch_tfms=aug_transforms())

    # Compute class weights
    class_counts = metadata['dx'].value_counts()
    total_samples = len(metadata)
    class_weights = total_samples / (len(class_counts) * class_counts)
    # Map weights to classes in the order of dls.vocab
    weights = torch.tensor([class_weights[class_name] for class_name in dls.vocab]).float().to(device)

    # Create learner with weighted loss function
    learn = vision_learner(dls, densenet121, metrics=error_rate)
    learn.model.to(device)
    learn.loss_func = CrossEntropyLoss(weight=weights)

    learn.fine_tune(5)

    # Test the model and get test accuracy
    test_accuracy, balanced_acc, classification_rep = test_model(learn)

    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f'model_densenet121_5epochs_{balanced_acc:.2f}bacc_{timestamp}.pkl'
    model_path = models_dir / model_filename
    learn.export(model_path)

    # Save the loss plot
    plot_loss(learn, f'densenet121_{timestamp}')

    # Save the classification report
    logs_dir = Path('logs')
    with open(logs_dir / f'classification_report_{timestamp}.txt', 'w') as f:
        f.write(classification_rep)

def compare_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    metadata_path = Path('Images/training_dataset/training_dataset_metadata.csv')
    images_path = Path('Images/training_dataset/Images')

    subset_size = 1000  # Adjust as needed

    dls, metadata = get_data_loaders(metadata_path, images_path, batch_tfms=aug_transforms(), subset_size=subset_size)

    # Compute class weights
    class_counts = metadata['dx'].value_counts()
    total_samples = len(metadata)
    class_weights = total_samples / (len(class_counts) * class_counts)
    weights = torch.tensor([class_weights[class_name] for class_name in dls.vocab]).float().to(device)

    models_to_test = [
        ('resnet18', resnet18),
        ('resnet34', resnet34),
        ('resnet50', resnet50),
        ('densenet121', densenet121),
        ('vgg16', vgg16),
    ]

    results = {}
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)

    # Prepare a single file to write all classification reports
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    classification_reports_path = logs_dir / f'classification_reports_{timestamp}.txt'
    with open(classification_reports_path, 'w') as reports_file:
        for model_name, model_func in models_to_test:
            print(f"\nTraining model: {model_name}")
            learn = vision_learner(dls, model_func, metrics=error_rate)
            learn.model.to(device)
            learn.loss_func = CrossEntropyLoss(weight=weights)
            learn.fine_tune(10)

            # Test the model and get test accuracy
            test_accuracy, balanced_acc, classification_rep = test_model(learn)

            model_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f'{model_name}_{balanced_acc:.2f}bacc_{model_timestamp}.pkl'
            model_path = models_dir / model_filename
            learn.export(model_path)

            plot_loss(learn, f'{model_name}_{model_timestamp}')

            val_loss, val_error = learn.validate()
            print(f"Validation Loss: {val_loss:.4f}, Error Rate: {val_error:.4f}")

            # Write the classification report to the combined file
            reports_file.write(f"Classification Report for {model_name}:\n")
            reports_file.write(classification_rep)
            reports_file.write("\n" + "="*80 + "\n")

            # Include test accuracy in results
            results[model_name] = {
                'model_path': str(model_path),
                'val_loss': val_loss,
                'error_rate': val_error,
                'test_accuracy': test_accuracy,
                'balanced_accuracy': balanced_acc,
                'timestamp': model_timestamp,
            }

    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv(logs_dir / 'model_comparison_results.csv')
    print("\nModel comparison completed. Results saved in 'logs/model_comparison_results.csv'.")
    print(f"All classification reports saved in '{classification_reports_path}'.")

def main():
    parser = argparse.ArgumentParser(description='Skin Cancer Classification Script')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true', help='Train the model')
    group.add_argument('--test', action='store_true', help='Test the model')
    group.add_argument('--predict', action='store_true', help='Predict a single image')
    group.add_argument('--compare_models', action='store_true', help='Compare different models')
    parser.add_argument('--image_path', type=str, help='Path to the image for prediction')
    parser.add_argument('--model_path', type=str, default='models/model_densenet121_5epochs.pkl', help='Path to the model file')

    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.test:
        learn = load_learner(args.model_path)
        test_accuracy, balanced_acc, classification_rep = test_model(learn)
    elif args.predict:
        if not args.image_path:
            print("Please provide the --image_path argument when using --predict.")
            exit(1)
        learn = load_learner(args.model_path)
        image_path = Path(args.image_path)
        predict_image(learn, image_path)
    elif args.compare_models:
        compare_models()

if __name__ == '__main__':
    main()

# Sample commands:
# python model.py --train
# python model.py --test --model_path path/to/your/model.pkl
# python model.py --predict --image_path path/to/your/image.jpg --model_path path/to/your/model.pkl
# python model.py --compare_models
