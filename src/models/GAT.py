import torch
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F

class GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, heads=4):
        super(GAT, self).__init__()
        torch.manual_seed(42)
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.6)
        self.fc = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x.squeeze()