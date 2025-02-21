import torch
from torch.nn import Linear
from torch_geometric.nn import GINConv, global_mean_pool
import torch.nn.functional as F

class GIN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, num_node_features, dropout):
        super(GIN, self).__init__()
        self.num_layers = num_layers

        nn1 = torch.nn.Sequential(
            torch.nn.Linear(num_node_features, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv1 = GINConv(nn1)
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv2 = GINConv(nn2)
     
        # Final linear layer to produce the output
        self.lin_mem = Linear(hidden_channels, out_channels)
        self.lin_time = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        for i in range(self.num_layers-1):
            x = self.conv2(x, edge_index)
            x = F.relu(x)
       
        x = global_mean_pool(x, batch)

        # Final output
        mem_pred = self.lin_mem(x)
        time_pred = self.lin_time(x)
        return mem_pred.squeeze(), time_pred.squeeze()  # Shape: [batch_size]