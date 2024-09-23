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
        # Initialize projected_x
        projected_x = {}

        # Conditionally project node features
        if 'operator' in x_dict and x_dict['operator'].shape[0] > 0:
            projected_x['operator'] = self.lin_operator(x_dict['operator'])
        if 'table' in x_dict and x_dict['table'].shape[0] > 0:
            projected_x['table'] = self.lin_table(x_dict['table'])
        if 'column' in x_dict and x_dict['column'].shape[0] > 0:
            projected_x['column'] = self.lin_column(x_dict['column'])
        if 'predicate' in x_dict and x_dict['predicate'].shape[0] > 0:
            projected_x['predicate'] = self.lin_predicate(x_dict['predicate'])

        # Apply HeteroConv layers
        x_dict = self.conv1(projected_x, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Aggregate operator node features per graph
        if 'operator' in x_dict and x_dict['operator'].shape[0] > 0:
            operator_features = x_dict['operator']
            operator_embedding = global_mean_pool(operator_features, batch_operator)
            out = self.lin(operator_embedding)
        else:
            # Handle cases with no operator nodes, e.g., assign a default value or skip
            out = torch.zeros(batch_operator.max().item() + 1, self.lin.out_features, device=x_dict[next(iter(x_dict))].device)

        return out.squeeze()