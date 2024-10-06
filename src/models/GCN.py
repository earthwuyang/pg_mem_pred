import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, num_node_features, dropout):
        super(GCN, self).__init__()
        self.num_layers = num_layers

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Final linear layer to produce the output
        self.lin_mem = Linear(hidden_channels, out_channels)
        self.lin_time = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First Graph Convolutional layer
        x = self.conv1(x, edge_index)
        x = x.relu()

        for i in range(self.num_layers-1):  
            # Second Graph Convolutional layer
            x = self.conv2(x, edge_index)
            x = x.relu()

        # Global mean pooling
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # Final output
        mem_pred = self.lin_mem(x)
        time_pred = self.lin_time(x)
        return mem_pred.squeeze(), time_pred.squeeze()  # Shape: [batch_size]