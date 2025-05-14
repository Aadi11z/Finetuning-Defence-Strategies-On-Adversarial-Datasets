import torch
import torch.nn as nn
import pandas as pd 
from functions import *
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, DistilBertForSequenceClassification

class RanMASKModel(nn.Module):
    def __init__(self, num_labels, t=6.0):
        super().__init__()
        self.bert = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=num_labels)
        self.activate = BELU(t)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Apply tSigmoid to logits
        logits = self.activate(outputs.logits)
        outputs.logits = logits
        
        return outputs

def random_mask(input_ids, attention_mask, mask_token_id, mask_prob=0.15):
    """
    Randomly mask tokens in the input sequence.
    
    Args:
        input_ids: Input token IDs
        attention_mask: Attention mask
        mask_token_id: ID of the [MASK] token
        mask_prob: Probability of masking each token
    
    Returns:
        Masked input_ids
    """
    # Only mask actual tokens (where attention_mask is 1)
    masked_input_ids = input_ids.clone()
    
    # Create a mask for tokens we can potentially mask
    can_mask = attention_mask.bool()
    
    # Create random mask based on probability
    rand = torch.rand(input_ids.shape, device=input_ids.device)
    mask_indices = (rand < mask_prob) & can_mask
    
    # Apply masking
    masked_input_ids[mask_indices] = mask_token_id
    
    return masked_input_ids

# RanMASK prediction with multiple random maskings
def ranmask_predict(model, input_ids, attention_mask, device, n_samples=100, mask_prob=0.15):
    model.eval()
    mask_token_id = 103  # [MASK] token ID for BERT/DistilBERT
    
    all_logits = []
    
    for _ in range(n_samples):
        # Apply random masking
        masked_input_ids = random_mask(input_ids, attention_mask, mask_token_id, mask_prob)
        
        with torch.no_grad():
            outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits)
    
    # Average predictions across all masked samples
    avg_logits = torch.stack(all_logits).mean(dim=0)
    return avg_logits

# Training function for RanMASK + tSigmoid
def train_ranmask(model, train_loader, val_loader, device, epochs=3, mask_prob=0.15):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    mask_token_id = 103  # [MASK] token ID for BERT/DistilBERT
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Apply random masking during training
            masked_input_ids = random_mask(input_ids, attention_mask, mask_token_id, mask_prob)
            
            optimizer.zero_grad()
            outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
        
        # Evaluation using randomized masking
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Use multiple random maskings for prediction
                avg_logits = ranmask_predict(
                    model, input_ids, attention_mask, device, n_samples=10, mask_prob=mask_prob
                )
                
                preds = torch.argmax(avg_logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        print(f"Validation Accuracy: {correct / total:.4f}")