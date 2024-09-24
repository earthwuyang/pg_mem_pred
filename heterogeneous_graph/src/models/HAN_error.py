# Heterogeneous Graph Transformer (HGT) model implementation.


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import HANConv, global_mean_pool
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch.nn import Linear

# ---------------------- GNN Model Definition ---------------------- #

metadata = (
    ['operator', 'table', 'column', 'predicate'],  # Node types
    [
        ('table', 'scannedby', 'operator'),
        ('predicate', 'filters', 'operator'),
        ('column', 'outputby', 'operator'),
        ('column', 'connects', 'predicate'),
        ('operator', 'calledby', 'operator'),
        ('table', 'selfloop', 'table'),
        ('column', 'selfloop', 'column')
    ]  # Edge types
)



class HeteroGraph(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_column_features, num_heads=1, dropout=0.6):
        """
        Args:
            hidden_channels (int): Number of hidden units.
            out_channels (int): Number of output units.
            num_column_features (int): Number of features for 'column' nodes.
            metadata (tuple): Tuple containing node types and edge types.
            num_heads (int): Number of attention heads in GATConv.
            dropout (float): Dropout rate for attention.
        """
        super(HeteroGraph, self).__init__()
        self.metadata = (
        ['operator', 'table', 'column', 'predicate'],  # Node types
        [
            ('table', 'scannedby', 'operator'),
            ('predicate', 'filters', 'operator'),
            ('column', 'outputby', 'operator'),
            ('column', 'connects', 'predicate'),
            ('operator', 'calledby', 'operator'),
            ('table', 'selfloop', 'table'),
            ('column', 'selfloop', 'column')
        ]  # Edge types
    )
        
        # Project node features to hidden_channels
        self.lin_operator = nn.Linear(4, hidden_channels)    # Operators have 4 features
        self.lin_table = nn.Linear(2, hidden_channels)       # Tables have 2 features
        self.lin_column = nn.Linear(num_column_features, hidden_channels)  # Column size + one-hot data types
        self.lin_predicate = nn.Linear(1, hidden_channels)   # Predicates have 1 feature
        
        # Define HANConv layers
        self.conv1 = HANConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            metadata=metadata,
            heads=num_heads,
            dropout=dropout
        )
        
        self.conv2 = HANConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            metadata=metadata,
            heads=num_heads,
            dropout=dropout
        )
        
        # Final linear layer to produce the output
        self.lin = nn.Linear(hidden_channels * num_heads, out_channels)
        
        # Optional: Layer normalization
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        
    def forward(self, x_dict, edge_index_dict, batch_operator):
        """
        Args:
            data (HeteroData): The input heterogeneous graph.
            batch_operator (Tensor): Batch indices for 'operator' nodes.
        
        Returns:
            Tensor: Output predictions of shape [batch_size].
        """
        
        # Project node features with checks
        projected_x = {}
        for node_type, lin_layer in [('operator', self.lin_operator),
                                     ('table', self.lin_table),
                                     ('column', self.lin_column),
                                     ('predicate', self.lin_predicate)]:
            if node_type in x_dict and x_dict[node_type].shape[0] > 0:
                projected_x[node_type] = lin_layer(x_dict[node_type])
            else:
                # Assign an empty tensor with appropriate feature size
                projected_x[node_type] = torch.empty(
                    (0, lin_layer.out_features), 
                    device=x_dict[next(iter(x_dict))].device
                )
        
        x_dict = projected_x
        
        # First HANConv layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        x_dict = {key: self.norm1(x) for key, x in x_dict.items()}
        
        # Second HANConv layer
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        x_dict = {key: self.norm2(x) for key, x in x_dict.items()}
        
        # Aggregate operator node features per graph
        operator_features = x_dict['operator']
        # Global mean pooling over operator nodes per graph
        operator_embedding = global_mean_pool(operator_features, batch_operator)
        
        # Final output
        out = self.lin(operator_embedding)
        return out.squeeze()  # Shape: [batch_size]
