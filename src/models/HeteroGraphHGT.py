# Heterogeneous Graph Transformer (HGT) model implementation.


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import HGTConv, global_mean_pool
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch.nn import Linear

# ---------------------- GNN Model Definition ---------------------- #




class HeteroGraphHGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, num_operator_features, num_heads=4, dropout=0.2):
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
        self.metadata = (
                ['operator'],  # Node types
                [
                    ('operator', 'calledby', 'operator'),
                ]  # Edge types
            )
        metadata = self.metadata

        self.num_layers = num_layers
        # Project node features to hidden_channels
        self.lin_operator = Linear(num_operator_features, hidden_channels)    # Operators have 4 features
    
        self.conv = HGTConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            metadata=metadata,
            heads=num_heads,
        )
        
        # Final linear layer to produce the output
        self.lin_mem = Linear(hidden_channels, out_channels)
        self.lin_time = Linear(hidden_channels, out_channels)
        
        # Optional: Layer normalization
        self.norm = nn.LayerNorm(hidden_channels)

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
        for node_type, lin_layer in [('operator', self.lin_operator)]:
            if node_type in x_dict and x_dict[node_type].shape[0] > 0:
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


        for i in range(self.num_layers):
            # Apply HGTConv layer
            x_dict = self.conv(x_dict, edge_index_dict)
            # Apply activation and normalization
            x_dict = {key: self.norm(F.elu(x)) for key, x in x_dict.items()}

        # Aggregate operator node features per graph
        operator_features = x_dict['operator']
        # Global mean pooling over operator nodes per graph
        operator_embedding = global_mean_pool(operator_features, batch_operator)
        
        # Final output
        mem_pred = self.lin_mem(operator_embedding)
        time_pred = self.lin_time(operator_embedding)
        return mem_pred.squeeze(), time_pred.squeeze()  # Shape: [batch_size]