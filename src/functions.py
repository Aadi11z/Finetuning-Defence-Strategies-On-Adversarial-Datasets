import torch
import torch.nn as nn
import torch.nn.functional as F

class BReLU(nn.Module):
    def __init__(self, t=6.0):  # t is the upper bound, can be tuned
        super().__init__()
        self.t = t

    def forward(self, x):
        return torch.clamp(x, min=0.0, max=self.t)

class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x
    
class BELU(nn.Module):
    def __init__(self, t=6.0):  # t is the upper bound
        super().__init__()
        self.t = t

    def forward(self, x):
        x = F.gelu(x)
        return torch.clamp(x, min=0.0, max=self.t)

def evaluate(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")