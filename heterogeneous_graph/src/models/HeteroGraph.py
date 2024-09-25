import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool
from torch_geometric.data import HeteroData
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
    def __init__(self, hidden_channels, out_channels, num_column_features, dropout=0.2):
        """
        Args:
            hidden_channels (int): Number of hidden units.
            out_channels (int): Number of output units.
            num_column_features (int): Number of features for 'column' nodes.
            metadata (tuple): A tuple containing node types and edge types.
            dropout (float): Dropout rate.
        """
        super().__init__()
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
        self.lin_operator = Linear(4, hidden_channels)    # Operators have 4 features
        self.lin_table = Linear(2, hidden_channels)       # Tables have 2 features
        self.lin_column = Linear(num_column_features, hidden_channels)  # Column size + one-hot data types
        self.lin_predicate = Linear(4, hidden_channels)   # Predicates have 4 features

        # Define HeteroConv layers with SAGEConv for each relation
        self.conv1 = HeteroConv({
            ('table', 'scannedby', 'operator'): SAGEConv(hidden_channels, hidden_channels),
            ('predicate', 'filters', 'operator'): SAGEConv(hidden_channels, hidden_channels),
            ('column', 'outputby', 'operator'): SAGEConv(hidden_channels, hidden_channels),
            ('column', 'connects', 'predicate'): SAGEConv(hidden_channels, hidden_channels),
            ('operator', 'calledby', 'operator'): SAGEConv(hidden_channels, hidden_channels),
            ('table', 'selfloop', 'table'): SAGEConv(hidden_channels, hidden_channels),
            ('column', 'selfloop', 'column'): SAGEConv(hidden_channels, hidden_channels),
        }, aggr='mean')  # Aggregation method can be 'mean', 'sum', or 'max'

        self.conv2 = HeteroConv({
            ('table', 'scannedby', 'operator'): SAGEConv(hidden_channels, hidden_channels),
            ('predicate', 'filters', 'operator'): SAGEConv(hidden_channels, hidden_channels),
            ('column', 'outputby', 'operator'): SAGEConv(hidden_channels, hidden_channels),
            ('column', 'connects', 'predicate'): SAGEConv(hidden_channels, hidden_channels),
            ('operator', 'calledby', 'operator'): SAGEConv(hidden_channels, hidden_channels),
            ('table', 'selfloop', 'table'): SAGEConv(hidden_channels, hidden_channels),
            ('column', 'selfloop', 'column'): SAGEConv(hidden_channels, hidden_channels),
        }, aggr='mean')

        # Final linear layer to produce the output
        self.lin = Linear(hidden_channels, out_channels)

        # Optional: Layer normalization
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_dict, edge_index_dict, batch_operator):
        """
        Args:
            data (HeteroData): The input heterogeneous graph.
            batch_operator (Tensor): Batch indices for 'operator' nodes.

        Returns:
            Tensor: Output predictions of shape [batch_size].
        """
        # # Extract x_dict and edge_index_dict from HeteroData
        # x_dict = data.x_dict
        # edge_index_dict = data.edge_index_dict

        # Project node features
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

        # First HeteroConv layer with SAGEConv
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}  # Apply activation
        x_dict = {key: self.norm1(x) for key, x in x_dict.items()}  # Apply layer normalization
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}  # Apply dropout

        # Second HeteroConv layer with SAGEConv
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}  # Apply activation
        x_dict = {key: self.norm2(x) for key, x in x_dict.items()}  # Apply layer normalization
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}  # Apply dropout

        # Aggregate operator node features per graph
        operator_features = x_dict['operator']
        # Global mean pooling over operator nodes per graph
        operator_embedding = global_mean_pool(operator_features, batch_operator)

        # Final output
        out = self.lin(operator_embedding)
        return out.squeeze()  # Shape: [batch_size]
