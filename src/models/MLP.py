import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_mean_pool

class MLPOnly(torch.nn.Module):
    """
    A purely MLP-based model replacing the GIN class. It ignores edges entirely
    and applies an MLP to each node’s features, then aggregates (e.g., mean)
    across nodes in a batch. Finally, produces two outputs: memory prediction
    and time prediction.

    Args:
        hidden_channels (int): Hidden dimension in the MLP layers.
        out_channels (int): Dimension of final output predictions (mem/time).
        num_layers (int): Unused for message passing, but we can still treat it
                          as how many MLP layers we want. Or keep it for parity.
        num_node_features (int): Number of input features per node (data.x).
        dropout (float): Dropout probability.
    """
    def __init__(self, hidden_channels, out_channels, num_layers, num_node_features, dropout):
        super().__init__()

        self.num_layers = num_layers
        
        # Example MLP: input -> hidden -> hidden -> ... -> hidden
        # We'll implement two layers for demonstration; you could
        # use num_layers to add more if desired.
        self.mlp1 = nn.Linear(num_node_features, hidden_channels)
        self.mlp2 = nn.Linear(hidden_channels, hidden_channels)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Final linear layers to produce the output
        self.lin_mem  = nn.Linear(hidden_channels, out_channels)
        self.lin_time = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        """
        data.x: [num_nodes, num_node_features]
        data.edge_index: ignored in MLP approach
        data.batch: batch assignments if multiple graphs in a batch
        """
        x, batch = data.x, data.batch

        # MLP for each node’s features (no message passing)
        x = F.relu(self.mlp1(x))
        x = self.dropout(x)

        # Possibly repeat if you want more layers
        x = F.relu(self.mlp2(x))
        x = self.dropout(x)

        # Aggregate over nodes in the batch
        x = global_mean_pool(x, batch)  # shape: [batch_size, hidden_channels]

        # Final output
        mem_pred  = self.lin_mem(x)   # shape: [batch_size, out_channels]
        time_pred = self.lin_time(x)  # shape: [batch_size, out_channels]

        # Squeeze to produce shape [batch_size] if out_channels=1, etc.
        return mem_pred.squeeze(), time_pred.squeeze()
