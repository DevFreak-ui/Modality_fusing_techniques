import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from model import MultimodalClassifier
from dataset import ChestXrayDataset
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    for epoch in range(num_epochs):
        # Training phase
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

        # Compute confusion matrix and metrics for training data
        train_cm = confusion_matrix(train_true_labels, train_pred_labels)
        train_accuracy = (train_cm[0,0] + train_cm[1,1]) / train_cm.sum()
        train_precision = precision_score(train_true_labels, train_pred_labels)
        train_recall = recall_score(train_true_labels, train_pred_labels)

        # Testing phase
        model.eval()
        test_true_labels, test_pred_labels = [], []

        with torch.no_grad():
            for combined_features, labels in test_loader:
                combined_features, labels = combined_features.to(device), labels.to(device).unsqueeze(1)

                outputs = model(combined_features)
                preds = (outputs > 0.5).float()

                test_true_labels.extend(labels.cpu().numpy())
                test_pred_labels.extend(preds.cpu().numpy())

        # Compute confusion matrix and metrics for test data
        test_cm = confusion_matrix(test_true_labels, test_pred_labels)
        test_accuracy = (test_cm[0,0] + test_cm[1,1]) / test_cm.sum()
        test_precision = precision_score(test_true_labels, test_pred_labels)
        test_recall = recall_score(test_true_labels, test_pred_labels)

        # Display results
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}')
        print('Training Confusion Matrix:')
        print(train_cm)
        print(f'Test Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}')
        print('Testing Confusion Matrix:')
        print(test_cm)

        # Optionally plot confusion matrices
        plot_confusion_matrix(train_cm, f'Train Confusion Matrix - Epoch {epoch+1}')
        plot_confusion_matrix(test_cm, f'Test Confusion Matrix - Epoch {epoch+1}')

# Paths to local data (update these paths as per your local setup)
train_normal_img_dir = os.path.expanduser('~/Downloads/chest_xray/train/NORMAL')
train_pneumonia_img_dir = os.path.expanduser('~/Downloads/chest_xray/train/PNEUMONIA')
train_normal_csv = os.path.expanduser('~/Downloads/synthesized_dataset/train/Normal.csv')
train_pneumonia_csv = os.path.expanduser('~/Downloads/synthesized_dataset/train/Pneumonia.csv')

# Load datasets
train_normal_dataset = ChestXrayDataset(img_dir=train_normal_img_dir, csv_file=train_normal_csv)
train_pneumonia_dataset = ChestXrayDataset(img_dir=train_pneumonia_img_dir, csv_file=train_pneumonia_csv)

# Combine datasets
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
