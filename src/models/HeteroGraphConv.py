import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GraphConv, global_mean_pool
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch.nn import Linear
from types import SimpleNamespace
# ---------------------- GNN Model Definition ---------------------- #

class HeteroGraphConv(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, encode_schema, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.encode_schema = encode_schema
        # Project node features to hidden_channels
        kwargs = SimpleNamespace(**kwargs)
        self.lin_operator = Linear(kwargs.num_operator_features, hidden_channels)    # Operators have 4 features
        if encode_schema:
            self.lin_table = Linear(kwargs.num_table_features, hidden_channels)
            self.lin_column = Linear(kwargs.num_column_features, hidden_channels)
        
        # Define HeteroConv layers without 'add_self_loops=False'
        layers = {
            ('operator', 'calledby', 'operator'): GraphConv(hidden_channels, hidden_channels),
        }
        if encode_schema:
            layers.update({
                ('table', 'scannedby', 'operator'): GraphConv(hidden_channels, hidden_channels),
                ('column', 'outputby', 'operator'): GraphConv(hidden_channels, hidden_channels),
                ('column', 'referencedby', 'column'): GraphConv(hidden_channels, hidden_channels),
                ('column', 'containedby', 'table'): GraphConv(hidden_channels, hidden_channels),
                ('column', 'selfloop', 'column'): GraphConv(hidden_channels, hidden_channels)
            })
        self.conv = HeteroConv(layers, aggr='sum')
        
        # Final linear layer to produce the output
        self.lin_mem = Linear(hidden_channels, out_channels)
        self.lin_time = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, batch_operator):
        # # Debugging: Print shapes before projection
        # for key, x in x_dict.items():
        #     print(f"Node type '{key}' has features shape: {x.shape}")
        
        # Project node features with checks
        projected_x = {}
        node_layers =  [('operator', self.lin_operator)]
        if self.encode_schema:
            node_layers += [('table', self.lin_table), ('column', self.lin_column)]
        for node_type, lin_layer in node_layers:
            if node_type in x_dict and x_dict[node_type].shape[0] > 0:
                projected_x[node_type] = lin_layer(x_dict[node_type])
            else:
                # Assign an empty tensor with appropriate feature size
                projected_x[node_type] = torch.empty((0, lin_layer.out_features), device=x_dict[next(iter(x_dict))].device)

        
        x_dict = projected_x
        
        # new_edge_index_dict = {}
        # for edge_type, edge_index in edge_index_dict.items():
        #     if edge_type[0] == 'operator' and edge_type[2] == 'operator':
        #         new_edge_index_dict[edge_type] = edge_index

        for i in range(self.num_layers):
            x_dict = self.conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Aggregate operator node features per graph
        operator_features = x_dict['operator']
        # Global mean pooling over operator nodes per graph
        operator_embedding = global_mean_pool(operator_features, batch_operator)

        # Final output
        mem_pred = self.lin_mem(operator_embedding)
        time_pred = self.lin_time(operator_embedding)
        return mem_pred.squeeze(), time_pred.squeeze()  # Shape: [batch_size]