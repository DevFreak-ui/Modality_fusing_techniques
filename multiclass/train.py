import os
import torch
from torch.utils.data import DataLoader
from dataset import CustomMedicalDataset
from model import MultiModalClassifier
from torch import nn, optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
import pandas as pd

# Define paths to CSV files
print("Defining paths to CSV files...")
base_dir = os.path.join(os.path.dirname(__file__), 'newly_synthesized')
train_csv_files = [
    os.path.join(base_dir, 't_train', 'tuberculosis.csv'),
    os.path.join(base_dir, 't_train', 'normal.csv'),
    os.path.join(base_dir, 'p_train', 'pneumonia.csv'),
    os.path.join(base_dir, 'p_train', 'normal.csv')
]

# Define test CSV files
test_csv_files = [
    os.path.join(base_dir, 't_train', 'tb_test.csv'),
    os.path.join(base_dir, 'p_train', 'pneumonia_test.csv')
]

# Define image directories
print("Defining image directories...")
base_img_dir = os.path.join('C:\\', 'Users', 'devfr', 'Downloads')
image_dirs = {
    'normal': [
        os.path.join(base_img_dir, 'TB', 'Normal'),
        os.path.join(base_img_dir, 'chest_xray', 'train', 'NORMAL')
    ],
    'tuberculosis': [os.path.join(base_img_dir, 'TB', 'Tuberculosis')],
    'pneumonia': [os.path.join(base_img_dir, 'chest_xray', 'train', 'Pneumonia')]
}

# Define transforms
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create training dataset
print("Creating training dataset...")
train_dataset = CustomMedicalDataset(train_csv_files, image_dirs, transform=train_transforms)
print(f"Training dataset created with {len(train_dataset)} samples.")

# Create testing dataset
print("Creating testing dataset...")
test_dataset = CustomMedicalDataset(test_csv_files, image_dirs, transform=test_transforms)
print(f"Testing dataset created with {len(test_dataset)} samples.")

# DataLoader
batch_size = 8
print("Creating DataLoaders...")

def collate_fn(batch):
    images, texts, labels = [], [], []
    for image, text, label in batch:
        images.append(image)
        texts.append(text)
        labels.append(label)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, texts, labels

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
print(f"DataLoaders created with batch size {batch_size}.")

# Initialize model, loss function, and optimizer with regularization
print("Initializing model, loss function, and optimizer...")
num_classes = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = MultiModalClassifier(num_classes, device=device)
model.to(device)

criterion = nn.CrossEntropyLoss()

# Set weight decay for L2 regularization
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
print("Model, criterion, and optimizer initialized.")

# Training loop without early stopping
num_epochs = 10
train_losses = []
val_losses = []

print("Starting training loop...")
for epoch in range(num_epochs):
    print(f"Epoch [{epoch+1}/{num_epochs}] starting...")
    model.train()
    total_loss = 0
    for batch_idx, (images, texts, labels) in enumerate(train_loader):
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Print loss after every 10 batches
        if (batch_idx + 1) % 10 == 0:
            avg_batch_loss = total_loss / (batch_idx + 1)
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {avg_batch_loss:.4f}")
    
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] completed. Training Loss: {avg_loss:.4f}")
    
    # Validation step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, texts, labels in test_loader:
            labels = labels.to(device)
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(test_loader)
    val_losses.append(avg_val_loss)
    print(f"Validation Loss: {avg_val_loss:.4f}")

print("Training completed.")

# Save the final model
torch.save(model.state_dict(), 'final_model.pth')

# Evaluation on test set
print("Evaluating on test set...")
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, texts, labels in test_loader:
        labels = labels.to(device)
        outputs = model(images, texts)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")

# Classification report
cm_labels = ['Normal', 'Tuberculosis', 'Pneumonia']
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=cm_labels, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2])
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=cm_labels, yticklabels=cm_labels, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
