import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from model import MultimodalClassifier
from dataset import ChestXrayDataset
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Set mode: 'development' or 'production'
mode = 'development'  # Change to 'production' for actual data

# Hyperparameters
batch_size = 8
learning_rate = 0.001
num_epochs = 10
test_split_ratio = 0.2

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.show()

def train_model(train_loader, test_loader, model, loss_fn, optimizer, num_epochs):
    all_train_true_labels, all_train_pred_labels = [], []
    all_test_true_labels, all_test_pred_labels = [], []

    for epoch in range(num_epochs):
        model.train()
        train_true_labels, train_pred_labels = [], []

        for combined_features, labels in train_loader:
            combined_features, labels = combined_features.to(device), labels.to(device).unsqueeze(1)

            outputs = model(combined_features)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = (outputs > 0.5).float()
            train_true_labels.extend(labels.cpu().numpy())
            train_pred_labels.extend(preds.cpu().numpy())

        all_train_true_labels.extend(train_true_labels)
        all_train_pred_labels.extend(train_pred_labels)

        # Testing phase
        model.eval()
        with torch.no_grad():
            for combined_features, labels in test_loader:
                combined_features, labels = combined_features.to(device), labels.to(device).unsqueeze(1)

                outputs = model(combined_features)
                preds = (outputs > 0.5).float()

                all_test_true_labels.extend(labels.cpu().numpy())
                all_test_pred_labels.extend(preds.cpu().numpy())

    # Compute confusion matrices after training
    train_cm = confusion_matrix(all_train_true_labels, all_train_pred_labels)
    test_cm = confusion_matrix(all_test_true_labels, all_test_pred_labels)

    # Display results
    print('Final Training Confusion Matrix:')
    print(train_cm)
    print('Final Testing Confusion Matrix:')
    print(test_cm)

    # Optionally plot confusion matrices
    plot_confusion_matrix(train_cm, 'Final Train Confusion Matrix')
    plot_confusion_matrix(test_cm, 'Final Test Confusion Matrix')


# Paths to local data (update these paths as per your local setup)
train_normal_img_dir = os.path.expanduser('~/Downloads/chest_xray/train/NORMAL')
train_pneumonia_img_dir = os.path.expanduser('~/Downloads/chest_xray/train/PNEUMONIA')
train_normal_csv = os.path.expanduser('~/Downloads/synthesized_dataset/train/Normal.csv')
train_pneumonia_csv = os.path.expanduser('~/Downloads/synthesized_dataset/train/Pneumonia.csv')

# Load datasets based on mode
if mode == 'development':
    # Dummy data
    dummy_data = [
        (np.random.rand(3, 224, 224), 0),  # Dummy image tensor and label
        (np.random.rand(3, 224, 224), 1),
        (np.random.rand(3, 224, 224), 0),
        (np.random.rand(3, 224, 224), 1),
        (np.random.rand(3, 224, 224), 0),
        (np.random.rand(3, 224, 224), 1),
    ]
    # Custom dataset class for dummy data
    class DummyDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_tensor, label = self.data[idx]
            return torch.tensor(img_tensor, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    train_dataset = DummyDataset(dummy_data)
    test_dataset = DummyDataset(dummy_data)  # Use the same for testing in development
else:
    # Load datasets for production
    train_normal_dataset = ChestXrayDataset(img_dir=train_normal_img_dir, csv_file=train_normal_csv)
    train_pneumonia_dataset = ChestXrayDataset(img_dir=train_pneumonia_img_dir, csv_file=train_pneumonia_csv)
    full_dataset = train_normal_dataset + train_pneumonia_dataset

    # Split dataset into training and testing sets
    train_size = int((1 - test_split_ratio) * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = MultimodalClassifier().to(device)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train and evaluate the model
train_model(train_loader, test_loader, model, loss_fn, optimizer, num_epochs)
