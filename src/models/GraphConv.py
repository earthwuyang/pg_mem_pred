import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GraphConv, global_mean_pool
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch.nn import Linear
# ---------------------- GNN Model Definition ---------------------- #

class HeteroGraph(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_column_features):
        super().__init__()
        # Project node features to hidden_channels
        self.lin_operator = Linear(4, hidden_channels)    # Operators have 4 features
        self.lin_table = Linear(2, hidden_channels)       # Tables have 2 features
        self.lin_column = Linear(num_column_features, hidden_channels)  # Column size + one-hot data types
        self.lin_predicate = Linear(1, hidden_channels)   # Predicates have 1 feature
        
        # Define HeteroConv layers without 'add_self_loops=False'
        self.conv1 = HeteroConv({
            ('table', 'scannedby', 'operator'): GraphConv(hidden_channels, hidden_channels),
            ('predicate', 'filters', 'operator'): GraphConv(hidden_channels, hidden_channels),
            ('column', 'outputby', 'operator'): GraphConv(hidden_channels, hidden_channels),
            ('column', 'connects', 'predicate'): GraphConv(hidden_channels, hidden_channels),
            ('operator', 'calledby', 'operator'): GraphConv(hidden_channels, hidden_channels),
            ('table', 'selfloop', 'table'): GraphConv(hidden_channels, hidden_channels),
            ('column', 'selfloop', 'column'): GraphConv(hidden_channels, hidden_channels),
        }, aggr='sum')
        
        self.conv2 = HeteroConv({
            ('table', 'scannedby', 'operator'): GraphConv(hidden_channels, hidden_channels),
            ('predicate', 'filters', 'operator'): GraphConv(hidden_channels, hidden_channels),
            ('column', 'outputby', 'operator'): GraphConv(hidden_channels, hidden_channels),
            ('column', 'connects', 'predicate'): GraphConv(hidden_channels, hidden_channels),
            ('operator', 'calledby', 'operator'): GraphConv(hidden_channels, hidden_channels),
            ('table', 'selfloop', 'table'): GraphConv(hidden_channels, hidden_channels),
            ('column', 'selfloop', 'column'): GraphConv(hidden_channels, hidden_channels),
        }, aggr='sum')
        
        # Final linear layer to produce the output
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, batch_operator):
        # # Debugging: Print shapes before projection
        # for key, x in x_dict.items():
        #     print(f"Node type '{key}' has features shape: {x.shape}")
        
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
                projected_x[node_type] = torch.empty((0, lin_layer.out_features), device=x_dict[next(iter(x_dict))].device)
        
        # Debugging: Print shapes after projection
        # for key, x in projected_x.items():
        #     print(f"After projection, node type '{key}' has features shape: {x.shape}")
        
        x_dict = projected_x
        
        # First HeteroConv layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Second HeteroConv layer
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Aggregate operator node features per graph
        operator_features = x_dict['operator']
        # Global mean pooling over operator nodes per graph
        operator_embedding = global_mean_pool(operator_features, batch_operator)
        out = self.lin(operator_embedding)
        return out.squeeze()  # Shape: [batch_size]