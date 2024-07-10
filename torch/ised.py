import torch.nn as nn
import torch.nn.functional as F

class ISED(nn.Module):
    def __init__(self, bbox):
        super().__init__()
        self.f = bbox
        
    def forward(self, x):
        pass

    def backward():
        pass
