# model.py

from torch_geometric.nn import global_mean_pool, GINConv
import torch.nn as nn
import torch

# Define the Graph GIN Model
class GraphGINModel(nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim, num_layers=3, dropout=0.1):
        super(GraphGINModel, self).__init__()
        self.convs = nn.ModuleList()
        self.pos_embedding = nn.Embedding(100, hidden_dim)  # Adjust size as needed
        self.linear = nn.Linear(num_node_features, hidden_dim)  # Projection layer

        # Define initial GINConv layer
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(mlp))

        # Define hidden GINConv layers
        for _ in range(num_layers - 2):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))

        # Define output GINConv layer
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(mlp))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Predict a single value

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        pos = data.node_level  # Node levels for positional embedding

        # Project node features to hidden_dim
        x = self.linear(x)  # Shape: (num_nodes, hidden_dim)

        # Add positional embeddings
        if pos is not None:
            x = x + self.pos_embedding(pos)  # Shape: (num_nodes, hidden_dim)

        # Apply GINConv layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)

        # Global Pooling
        x = global_mean_pool(x, batch)  # Shape: (batch_size, hidden_dim)
        out = self.fc(x)  # Shape: (batch_size, 1)
        return out
