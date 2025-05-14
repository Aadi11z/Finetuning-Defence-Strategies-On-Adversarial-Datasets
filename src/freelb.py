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

class FreeLBModel(nn.Module):
    def __init__(self, num_labels, t=6):
        super().__init__()
        self.bert = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=num_labels)
        self.activate = BELU(t)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Apply tSigmoid to logits
        logits = self.activate(outputs.logits)
        outputs.logits = logits
        
        return outputs


def train_freelb(model, train_loader, val_loader, device, epochs=3, adv_steps=3, adv_lr=1e-2, adv_max_norm=0.5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Initialize delta (perturbation) for embeddings
            if isinstance(model, torch.nn.DataParallel):
                embeds_init = model.module.bert.get_input_embeddings()(input_ids)
            else:
                embeds_init = model.bert.get_input_embeddings()(input_ids)
                
            # Create attention mask tensor for embeddings
            input_mask = attention_mask.to(embeds_init)
            input_lengths = torch.sum(input_mask, dim=1)
            
            # Create perturbation delta initialized with zeros
            delta = torch.zeros_like(embeds_init).to(device)
            delta.requires_grad_()
            total_loss_batch = 0
            for astep in range(adv_steps):
                # Apply perturbation to embeddings
                delta_norm = torch.norm(delta, dim=-1).detach()
                exceed_mask = (delta_norm > adv_max_norm).to(embeds_init)
                delta = delta * (adv_max_norm / delta_norm.unsqueeze(-1)) * exceed_mask.unsqueeze(-1) + delta * (~exceed_mask.unsqueeze(-1).bool())
                
                # Forward pass with perturbed embeddings
                embeds = embeds_init + delta
                
                # Replace the embedding layer's forward pass
                outputs = model.bert(
                    inputs_embeds=embeds,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Apply tSigmoid for robustness
                logits = model.t_sigmoid(outputs.logits)
                loss = F.cross_entropy(logits, labels)
                
                total_loss_batch += loss
                
                # Calculate gradient for delta
                loss.backward(retain_graph=True)
                delta_grad = delta.grad.clone().detach()
                
                # Update delta (gradient ascent)
                delta.data = delta.data + adv_lr * delta_grad / torch.norm(delta_grad, dim=-1, keepdim=True)
                delta.grad.zero_()
            
            # Final forward pass with perturbed embeddings
            embeds = embeds_init + delta
            outputs = model.bert(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                labels=labels
            )
            
            logits = model.t_sigmoid(outputs.logits)
            loss = F.cross_entropy(logits, labels)
            
            # Standard backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
        evaluate(model, val_loader, device)