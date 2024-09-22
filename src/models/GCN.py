import torch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, 1)  # Regression output

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First Graph Convolutional layer
        x = self.conv1(x, edge_index)
        x = x.relu()

        # Second Graph Convolutional layer
        x = self.conv2(x, edge_index)
        x = x.relu()

        # Global mean pooling
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # Final linear layer
        x = self.fc(x)

        return x.squeeze()  # [batch_size]