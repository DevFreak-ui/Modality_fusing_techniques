import os
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

# Load pre-trained image model and processor
img_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
img_model = AutoModel.from_pretrained('facebook/dinov2-base')

def ImageEncoder(img_path):
    """
    Encodes an image into a 768-dimensional vector using a pre-trained DINO model.
    
    Args:
        img_path (str): Path to the image file.
    
    Returns:
        torch.Tensor: Image embeddings of shape [1, 768].
    """
    img_path = os.path.abspath(img_path)  # Convert to absolute path
    image = Image.open(img_path)
    img_inputs = img_processor(images=image, return_tensors="pt")
    img_outputs = img_model(**img_inputs)
    image_embeddings = img_outputs.last_hidden_state.mean(dim=1)
    return image_embeddings
