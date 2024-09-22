import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.data import Data

class TreeTransformer(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_heads=4, num_layers=2, output_dim=1):
        super(TreeTransformer, self).__init__()

        self.embedding = nn.Linear(in_channels, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        # Extract node features, edge indices, and batch information
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Initial embedding
        x = self.embedding(x)
        
        # Apply Transformer layers
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
        
        # Global mean pooling across all the nodes in each graph
        x = global_mean_pool(x, batch)

        # Final fully connected layer to predict peak memory
        x = self.fc(x)
        
        return x.squeeze(-1)  # Ensure output size matches (batch_size,)