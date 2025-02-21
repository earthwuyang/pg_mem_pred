import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool
from torch_geometric.data import HeteroData
from torch.nn import Linear

# ---------------------- GNN Model Definition ---------------------- #



class HeteroGraphSAGE(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, num_operator_features, dropout=0.2):
        """
        Args:
            hidden_channels (int): Number of hidden units.
            out_channels (int): Number of output units.
            num_column_features (int): Number of features for 'column' nodes.
            metadata (tuple): A tuple containing node types and edge types.
            dropout (float): Dropout rate.
        """
        super().__init__()

        self.num_layers = num_layers
        # Project node features to hidden_channels
        self.lin_operator = Linear(num_operator_features, hidden_channels)    # Operators have 4 features


        # Define HeteroConv layers with SAGEConv for each relation
        self.conv = HeteroConv({
            ('operator', 'calledby', 'operator'): SAGEConv(hidden_channels, hidden_channels),
        }, aggr='mean')  # Aggregation method can be 'mean', 'sum', or 'max'

        # Final linear layer to produce the output
        self.lin_mem = Linear(hidden_channels, out_channels)
        self.lin_time = Linear(hidden_channels, out_channels)

        # Optional: Layer normalization
        self.norm = nn.LayerNorm(hidden_channels)

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

        # Project node features
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

        for i in range(self.num_layers):
            x_dict = self.conv(x_dict, edge_index_dict)
            x_dict = {key: F.elu(x) for key, x in x_dict.items()}  # Apply activation
            x_dict = {key: self.norm(x) for key, x in x_dict.items()}  # Apply layer normalization
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}  # Apply dropout

        # Aggregate operator node features per graph
        operator_features = x_dict['operator']
        # Global mean pooling over operator nodes per graph
        operator_embedding = global_mean_pool(operator_features, batch_operator)

        # Final output
        mem_pred = self.lin_mem(operator_embedding)
        time_pred = self.lin_time(operator_embedding)
        return mem_pred.squeeze(), time_pred.squeeze()  # Shape: [batch_size]
