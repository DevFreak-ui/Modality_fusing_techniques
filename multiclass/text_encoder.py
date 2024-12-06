import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class TextEncoder(nn.Module):
    def __init__(self, device=None):
        super(TextEncoder, self).__init__()
        self.device = device
        # Load pre-trained BERT tokenizer and model
        self.text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)

    def forward(self, texts):
        """
        Encodes texts into embeddings using a pre-trained BERT model.
        
        Args:
            texts (list of str): List of input texts.
        
        Returns:
            torch.Tensor: Text embeddings of shape [batch_size, 768].
        """
        text_inputs = self.text_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        text_outputs = self.text_model(**text_inputs)
        text_embeddings = text_outputs.last_hidden_state.mean(dim=1)
        return text_embeddings
