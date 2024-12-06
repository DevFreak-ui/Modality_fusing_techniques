import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

class ImageEncoder(nn.Module):
    def __init__(self, device=None):
        super(ImageEncoder, self).__init__()
        self.device = device
        # Load pre-trained image model and processor
        self.img_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.img_model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)

    def forward(self, images):
        """
        Encodes images into embeddings using a pre-trained DINOv2-base model.
        
        Args:
            images (list of PIL.Image): List of images.
        
        Returns:
            torch.Tensor: Image embeddings of shape [batch_size, 768].
        """
        img_inputs = self.img_processor(images=images, return_tensors="pt")
        img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}
        img_outputs = self.img_model(**img_inputs)
        image_embeddings = img_outputs.last_hidden_state.mean(dim=1)
        return image_embeddings
