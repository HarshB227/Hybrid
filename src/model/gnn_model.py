import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class FraudGNN(nn.Module):

    def __init__(self, in_channels, hidden_channels=64, dropout=0.3):
        super().__init__()

        # Graph layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # regularization
        self.dropout = nn.Dropout(dropout)

        # classifier
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):

        # ----- Layer 1 -----
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # ----- Layer 2 -----
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # ----- Output -----
        x = self.classifier(x)

        # probability 0-1
        return torch.sigmoid(x)