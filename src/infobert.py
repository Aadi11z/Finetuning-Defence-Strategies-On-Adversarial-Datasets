import torch
import torch.nn as nn
import pandas as pd 
from functions import *
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer,  AutoModelForSequenceClassification
import re

class InfoBERTModel(nn.Module):
    def __init__(self, num_labels, t=0.05, ib_lambda=0.1, rf_lambda=0.1):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained("kweinmeister/distilbert-anli")
        self.activate = BELU(t)
        self.ib_lambda = ib_lambda  # Information Bottleneck lambda
        self.rf_lambda = rf_lambda  # Robust Feature lambda
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        
        logits = outputs.logits
        hidden_states = outputs.hidden_states
        
        hidden_states_processed = [self.activate(h) for h in hidden_states]
        
        loss = outputs.loss if outputs.loss is not None else 0
        
        if self.training and labels is not None:
            last_hidden = hidden_states_processed[-1]
            ib_loss = self._compute_ib_loss(last_hidden, input_ids)
            
            rf_loss = self._compute_rf_loss(hidden_states_processed)
            
            loss = loss + self.ib_lambda * ib_loss + self.rf_lambda * rf_loss
        
        # Apply tSigmoid to logits for bounded outputs
        logits = self.activate(logits)
        
        # Return custom output object
        outputs.loss = loss
        outputs.logits = logits
        return outputs
    
    def _compute_ib_loss(self, hidden_states, input_ids):
        # Simplified implementation of Information Bottleneck loss
        # In the actual implementation, this would estimate mutual information
        batch_size = hidden_states.size(0)
        hidden_norm = F.normalize(hidden_states, dim=-1)
        cosine_sim = torch.matmul(hidden_norm, hidden_norm.transpose(0, 1))
        mask = torch.eye(batch_size, device=hidden_states.device)
        ib_loss = (cosine_sim * (1 - mask)).mean()
        return ib_loss
    
    def _compute_rf_loss(self, hidden_states_list):
        # Simplified implementation of Robust Feature loss
        # In actual implementation, this would increase MI between local and global features
        local_features = hidden_states_list[-2]  # Second-to-last layer
        global_features = hidden_states_list[-1]  # Last layer
        
        local_norm = F.normalize(local_features.mean(dim=1), dim=-1)
        global_norm = F.normalize(global_features.mean(dim=1), dim=-1)
        
        rf_loss = -torch.mean(torch.sum(local_norm * global_norm, dim=-1))
        return rf_loss

def train_infobert(model, train_loader, val_loader, device, epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
        evaluate(model, val_loader, device)

