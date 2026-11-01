"""
FedCardio Clinical Neural Network Architecture
High-performance 3-layer Tabular Clinical Multi-Layer Perceptron (MLP)
with Group Normalization and Dropout for robust cardiac event prediction.
GroupNorm enables per-sample gradient vectorization for DP-SGD.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple

from fedcardio.config import TOTAL_FEATURES

class ClinicalNet(nn.Module):
    def __init__(
        self,
        in_features: int = TOTAL_FEATURES,
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features

        # Layer 1
        self.fc1 = nn.Linear(in_features, hidden_dim1)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=hidden_dim1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        # Layer 2
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=hidden_dim2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout / 2)

        # Output Head (Logit)
        self.out = nn.Linear(hidden_dim2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.relu1(self.gn1(self.fc1(x))))
        x = self.drop2(self.relu2(self.gn2(self.fc2(x))))
        logits = self.out(x)
        return logits
        
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns risk probabilities in [0, 1]."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
            
    def get_weights(self) -> List[np.ndarray]:
        """Extracts model weights as numpy arrays for Flower serialization."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]
        
    def set_weights(self, weights: List[np.ndarray]) -> None:
        """Loads weights from numpy arrays into PyTorch state_dict."""
        state_dict = dict(zip(self.state_dict().keys(), [torch.tensor(w) for w in weights]))
        self.load_state_dict(state_dict, strict=True)
        
    def calculate_payload_size_kb(self) -> float:
        """Calculates exact serialized weight payload size transferred across network."""
        total_bytes = sum(w.nbytes for w in self.get_weights())
        return round(total_bytes / 1024.0, 2)
