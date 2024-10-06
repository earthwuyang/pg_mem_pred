import torch
from torch.nn import Linear, ModuleList
from torch_geometric.nn import global_mean_pool, TransformerConv
import torch.nn.functional as F

class GraphTransformer(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, num_node_features, dropout, heads=4):
        super(GraphTransformer, self).__init__()
        
        # TransformerConv with multi-head attention
        self.conv1 = TransformerConv(num_node_features, hidden_channels // heads, heads=heads) 
        self.conv2 = TransformerConv(hidden_channels, hidden_channels // heads, heads=heads) 
        
        # Final linear layer to produce the output
        self.lin_mem = Linear(hidden_channels, out_channels)
        self.lin_time = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First TransformerConv layer with multi-head attention
        x = self.conv1(x, edge_index)
        x = x.relu()

        # Second TransformerConv layer with multi-head attention
        x = self.conv2(x, edge_index)
        x = x.relu()

        # Global mean pooling
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # Final output
        mem_pred = self.lin_mem(x)
        time_pred = self.lin_time(x)
        return mem_pred.squeeze(), time_pred.squeeze()  # Shape: [batch_size]