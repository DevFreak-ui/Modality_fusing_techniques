import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class CustomMedicalDataset(Dataset):
    def __init__(self, csv_files, image_dirs, transform=None):
        """
        Args:
            csv_files (list of str): List of paths to CSV files.
            image_dirs (dict): Dictionary mapping labels to a list of image directories.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        # Read and combine all CSV files into a single DataFrame
        data_frames = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            data_frames.append(df)
        self.data_frame = pd.concat(data_frames, ignore_index=True).reset_index(drop=True)
        
        self.image_dirs = image_dirs  # Dictionary mapping labels to image directories
        self.transform = transform
        self.label_dict = {'normal': 0, 'tuberculosis': 1, 'pneumonia': 2}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Retrieve image name and label
        img_name = self.data_frame.iloc[idx]['image']
        target_label = self.data_frame.iloc[idx]['target'].lower()

        # Search for the image in the provided directories
        img_path = None
        for dir_path in self.image_dirs[target_label]:
            potential_path = os.path.join(dir_path, img_name)
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        if img_path is None:
            raise FileNotFoundError(f"Image {img_name} not found in directories {self.image_dirs[target_label]}")

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        # Get text report
        text_report = self.data_frame.iloc[idx]['notes']

        # Convert label to numerical format
        label = self.label_dict[target_label]

        return image, text_report, label
