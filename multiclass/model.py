import torch
import torch.nn as nn
from image_encoder import ImageEncoder
from text_encoder import TextEncoder

class MultiModalClassifier(nn.Module):
    def __init__(self, num_classes, device=None):
        super(MultiModalClassifier, self).__init__()
        self.device = device
        self.image_encoder = ImageEncoder(device=self.device)
        self.text_encoder = TextEncoder(device=self.device)
        self.classifier = nn.Linear(768 + 768, num_classes)  # Both encoders output 768-dimensional vectors

    def forward(self, images, texts):
        img_features = self.image_encoder(images)
        txt_features = self.text_encoder(texts)
        combined_features = torch.cat((img_features, txt_features), dim=1)
        output = self.classifier(combined_features)
        return output
