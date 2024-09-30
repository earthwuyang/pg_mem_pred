import torch
import torch.nn as nn
from torch_geometric.data import Data

class TreeLSTM(nn.Module):
    def __init__(self, in_channels, hidden_dim, output_dim=1):
        super(TreeLSTM, self).__init__()

        # Define LSTM cell for node updates
        self.lstm_cell = nn.LSTMCell(in_channels, hidden_dim)
        
        # Fully connected layer for the final regression task
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Initialize hidden and cell states for each node
        h = torch.zeros(x.size(0), self.lstm_cell.hidden_size, device=x.device)
        c = torch.zeros(x.size(0), self.lstm_cell.hidden_size, device=x.device)
        
        # Identify node dependencies: count how many incoming edges each node has (number of children)
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        children_count = torch.zeros(num_nodes, device=x.device, dtype=torch.long)
        
        for i in range(num_edges):
            parent_idx = edge_index[1, i]
            children_count[parent_idx] += 1
        
        # Process nodes in a bottom-up manner (from leaf to root)
        processed = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
        
        # Keep processing until all nodes have been visited
        while not processed.all():
            for node_idx in range(num_nodes):
                if processed[node_idx]:
                    continue
                
                # Get the children of the current node
                children = (edge_index[1] == node_idx).nonzero(as_tuple=False).squeeze(1)
                
                # If all children have been processed, update this node
                if all(processed[edge_index[0, child]] for child in children):
                    # Aggregate children's hidden states
                    if len(children) > 0:
                        h_children = h[edge_index[0, children]].mean(dim=0)
                        c_children = c[edge_index[0, children]].mean(dim=0)
                    else:
                        h_children = torch.zeros_like(h[node_idx])
                        c_children = torch.zeros_like(c[node_idx])
                    
                    # Update the current node using its features and aggregated children's states
                    h[node_idx], c[node_idx] = self.lstm_cell(x[node_idx], (h_children, c_children))
                    processed[node_idx] = True

        # Global pooling (mean of hidden states across all nodes)
        pooled = torch.mean(h, dim=0, keepdim=True)

        # Final prediction through fully connected layer
        output = self.fc(pooled)
        
        return output.squeeze(-1)