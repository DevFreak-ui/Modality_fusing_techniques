import torch
from torch import nn

class MultimodalClassifier(nn.Module):
    """
    Multimodal classification model that takes combined image and text features as input.
    
    Architecture:
    - Fully connected layers: 1536 -> 512 -> 128 -> 1
    - Activation functions: ReLU for hidden layers, Sigmoid for output layer
    """
    def __init__(self):
        super(MultimodalClassifier, self).__init__()
        self.fc1 = nn.Linear(1536, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1536].
        
        Returns:
            torch.Tensor: Output probabilities for binary classification.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
