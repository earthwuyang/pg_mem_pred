import torch
from torch.nn import Linear
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, num_node_features, dropout, heads=4):
        super(GAT, self).__init__()
        self.dropout = dropout
        
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads, dropout=self.dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout =self.dropout)
        
        # Final linear layer to produce the output
        self.lin_mem = Linear(hidden_channels, out_channels)
        self.lin_time = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = global_mean_pool(x, batch)
        
        # Final output
        mem_pred = self.lin_mem(x)
        time_pred = self.lin_time(x)
        return mem_pred.squeeze(), time_pred.squeeze()  # Shape: [batch_size]