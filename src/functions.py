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

class tSigmoid(nn.Module):
    def __init__(self, t=0.15):
        super().__init__()
        self.t = t
        
    def forward(self, x):
        return torch.clamp(x, min=self.t, max=(1-self.t))