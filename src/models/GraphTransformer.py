import torch
from torch_geometric.nn import global_mean_pool, TransformerConv
import torch.nn.functional as F

class GraphTransformer(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, heads=4):
        super(GraphTransformer, self).__init__()
        torch.manual_seed(42)
        
        # TransformerConv with multi-head attention
        self.conv1 = TransformerConv(num_node_features, hidden_channels // heads, heads=heads)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels // heads, heads=heads)
        self.fc = torch.nn.Linear(hidden_channels, 1)  # Regression output

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

        # Final linear layer
        x = self.fc(x)

        return x.squeeze()  # [batch_size]