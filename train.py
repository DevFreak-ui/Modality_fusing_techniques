import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from model import MultimodalClassifier
from dataset import ChestXrayDataset
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths to local data (update these paths as per your local setup)
train_normal_img_dir = os.path.expanduser('~/Downloads/chest_xray/train/NORMAL')
train_pneumonia_img_dir = os.path.expanduser('~/Downloads/chest_xray/train/PNEUMONIA')
train_normal_csv = os.path.expanduser('~/Downloads/synthesized_dataset/train/Normal.csv')
train_pneumonia_csv = os.path.expanduser('~/Downloads/synthesized_dataset/train/Pneumonia.csv')

# Hyperparameters
batch_size = 8
learning_rate = 0.001
num_epochs = 1
test_split_ratio = 0.2

def plot_confusion_matrix(cm, title, ax):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'], ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title(title)

def train_model(train_loader, test_loader, model, loss_fn, optimizer, num_epochs, print_every=5):
    all_train_true_labels, all_train_pred_labels = [], []
    all_test_true_labels, all_test_pred_labels = [], []

    for epoch in range(num_epochs):
        model.train()
        train_true_labels, train_pred_labels = [], []
        train_loss = 0
        batch_loss = 0
        batch_count = 0 

        # Wrap the training loop with tqdm
        for batch_idx, (combined_features, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)):
            combined_features, labels = combined_features.to(device), labels.to(device).unsqueeze(1)

            outputs = model(combined_features)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = (outputs > 0.5).float()
            train_true_labels.extend(labels.cpu().numpy())
            train_pred_labels.extend(preds.cpu().numpy())

            train_loss += loss.item()
            batch_loss += loss.item()
            batch_count += 1

            # Print training loss every 'print_every' batches
            if (batch_idx + 1) % print_every == 0:
                avg_batch_loss = batch_loss / batch_count 
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1} | Batch Loss: {avg_batch_loss:.6f}")
                batch_loss = 0
                batch_count = 0 

        all_train_true_labels.extend(train_true_labels)
        all_train_pred_labels.extend(train_pred_labels)

        # Compute average training loss for the epoch
        avg_train_loss = train_loss / len(train_loader)  

        # Testing phase
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for combined_features, labels in tqdm(test_loader, desc="Testing", leave=False):
                combined_features, labels = combined_features.to(device), labels.to(device).unsqueeze(1)

                outputs = model(combined_features)
                loss = loss_fn(outputs, labels)
                test_loss += loss.item()

                preds = (outputs > 0.5).float()
                all_test_true_labels.extend(labels.cpu().numpy())
                all_test_pred_labels.extend(preds.cpu().numpy())

        # Compute average testing loss
        avg_test_loss = test_loss / len(test_loader)  
        
        # Calculate accuracy
        accuracy = accuracy_score(all_test_true_labels, all_test_pred_labels)

        # Print training and testing loss after every two epochs
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} | Training Loss: {avg_train_loss:.6f} | Testing Loss: {avg_test_loss:.6f} | Accuracy: {accuracy:.4f}")  # {{ edit_3 }} Print accuracy

    # Compute confusion matrices after training
    train_cm = confusion_matrix(all_train_true_labels, all_train_pred_labels)
    test_cm = confusion_matrix(all_test_true_labels, all_test_pred_labels)

    # Calculate metrics
    precision = precision_score(all_test_true_labels, all_test_pred_labels, zero_division=0)
    recall = recall_score(all_test_true_labels, all_test_pred_labels, zero_division=0)
    f1 = f1_score(all_test_true_labels, all_test_pred_labels, zero_division=0)

    print(f"\nPrecision: {precision:.6f} \nRecall: {recall:.6f} \nF1 Score: {f1:.6f} \nAccuracy: {accuracy:.4f}") 

    # Confusion matrices
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    plot_confusion_matrix(train_cm, 'Final Train Confusion Matrix', axs[0])
    plot_confusion_matrix(test_cm, 'Final Test Confusion Matrix', axs[1])
    plt.show()

# Load datasets based on mode
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
