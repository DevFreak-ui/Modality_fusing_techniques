import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT tokenizer and model
text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_model = BertModel.from_pretrained('bert-base-uncased')

def text_encoder(text):
    """
    Encodes a given text into a 768-dimensional vector using a pre-trained BERT model.
    
    Args:
        text (str): Input text to encode.
    
    Returns:
        torch.Tensor: Text embeddings of shape [1, 768].
    """
    text_inputs = text_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    text_outputs = text_model(**text_inputs)
    text_embeddings = text_outputs.last_hidden_state.mean(dim=1)
    return text_embeddings
