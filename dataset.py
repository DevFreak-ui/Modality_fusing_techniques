import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from image_encoder import ImageEncoder
from text_encoder import text_encoder

class ChestXrayDataset(Dataset):
    """
    Custom Dataset class for loading chest X-ray images and corresponding reports.
    
    Args:
        img_dir (str): Directory containing image files.
        csv_file (str): Path to the CSV file with image filenames and labels.
    """
    
    def __init__(self, img_dir, csv_file):
        self.img_dir = os.path.abspath(img_dir)
        self.data = pd.read_csv(os.path.abspath(csv_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            combined_features (torch.Tensor): Concatenated image and text embeddings.
            label (torch.Tensor): Binary label for the input data.
        """
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image_embeddings = ImageEncoder(img_name).squeeze(0)  # Shape [768]

        text = self.data.iloc[idx, -1]
        text_embeddings = text_encoder(text).squeeze(0)  # Shape [768]

        combined_features = torch.cat((image_embeddings, text_embeddings), dim=0)  # Shape [1536]

        label = 1 if self.data.iloc[idx, 9] == "pneumonia" else 0  # Binary label

        return combined_features, torch.tensor(label, dtype=torch.float32)
