# Heterogeneous Graph Transformer (HGT) model implementation.


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import HGTConv, global_mean_pool
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch.nn import Linear

# ---------------------- GNN Model Definition ---------------------- #

metadata = (
    ['operator', 'table', 'column', 'predicate', 'operation', 'literal', 'numeral'],  # Node types
    [
        ('table', 'scannedby', 'operator'),
        ('predicate', 'filters', 'operator'),
        ('column', 'outputby', 'operator'),
        ('column', 'connects', 'predicate'),
        ('operator', 'calledby', 'operator'),
        {'operation', 'filters', 'operator'},
        ('operation', 'connects', 'predicate'),
        ('literal', 'connects', 'operation'),
        ('numeral', 'connects', 'operation'),
        ('literal', 'selfloop', 'literal'),
        ('numeral', 'selfloop', 'numeral'),
        ('table', 'selfloop', 'table'),
        ('column', 'selfloop', 'column'),
        ('predicate', 'connects', 'predicate')
    ]  # Edge types
)


class HeteroGraph(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_column_features, num_heads=4, dropout=0.2):
        """
        Args:
            hidden_channels (int): Number of hidden units.
            out_channels (int): Number of output units.
            num_column_features (int): Number of features for 'column' nodes.
            metadata (tuple): A tuple containing node types and edge types.
            num_heads (int): Number of attention heads in HGTConv.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.metadata = metadata  # (node_types, edge_types)

        # Project node features to hidden_channels
        self.lin_operator = Linear(4, hidden_channels)    # Operators have 4 features
        self.lin_table = Linear(2, hidden_channels)       # Tables have 2 features
        self.lin_column = Linear(num_column_features, hidden_channels)  # Column size + one-hot data types
        self.lin_predicate = Linear(4, hidden_channels)   # Predicates have 1 feature
        self.lin_operation = Linear(8, hidden_channels)  
        self.lin_literal = Linear(1, hidden_channels)     
        self.lin_numeral = Linear(1, hidden_channels)     

        # Define HGTConv layers
        self.conv1 = HGTConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            metadata=metadata,
            heads=num_heads,
        )
        
        self.conv2 = HGTConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            metadata=metadata,
            heads=num_heads,
        )
        
        # Final linear layer to produce the output
        self.lin = Linear(hidden_channels, out_channels)
        
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
        # Extract x_dict and edge_index_dict from HeteroData
        # x_dict = data.x_dict
        # edge_index_dict = data.edge_index_dict

        # Project node features with checks
        projected_x = {}
        for node_type, lin_layer in [('operator', self.lin_operator),
                                     ('table', self.lin_table),
                                     ('column', self.lin_column),
                                     ('predicate', self.lin_predicate)]:
            if node_type in x_dict and x_dict[node_type].shape[0] > 0:
                # print(f"{node_type} x_dict[node_type].shape {x_dict[node_type].shape}")
                projected_x[node_type] = lin_layer(x_dict[node_type])
            else:
                # Assign an empty tensor with appropriate feature size
                projected_x[node_type] = torch.empty(
                    (0, lin_layer.out_features), 
                    device=x_dict[next(iter(x_dict))].device
                )
        
        x_dict = projected_x

        # Combine all edge indices and assign edge type indices
        edge_index = []
        edge_type = []
        edge_type_mapping = {etype: idx for idx, etype in enumerate(self.metadata[1])}
        for etype, eindex in edge_index_dict.items():
            edge_index.append(eindex)
            edge_type += [edge_type_mapping[etype]] * eindex.size(1)
        
        if edge_index:
            edge_index = torch.cat(edge_index, dim=1)
            edge_type = torch.tensor(edge_type, dtype=torch.long, device=edge_index.device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x_dict[next(iter(x_dict))].device)
            edge_type = torch.empty((0,), dtype=torch.long, device=x_dict[next(iter(x_dict))].device)

        # First HGTConv layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        # Apply activation and normalization
        x_dict = {key: self.norm1(F.elu(x)) for key, x in x_dict.items()}

        # Second HGTConv layer
        x_dict = self.conv2(x_dict, edge_index_dict)
        # Apply activation and normalization
        x_dict = {key: self.norm2(F.elu(x)) for key, x in x_dict.items()}

        # Aggregate operator node features per graph
        operator_features = x_dict['operator']
        # Global mean pooling over operator nodes per graph
        operator_embedding = global_mean_pool(operator_features, batch_operator)
        
        # print(f"self.lin {self.lin}")
        # print(f"operator_embedding {operator_embedding.shape}")
        # Final output
        out = self.lin(operator_embedding)
        return out.squeeze()  # Shape: [batch_size]
