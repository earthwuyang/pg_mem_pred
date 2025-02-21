# Relational Graph Convolutional Network (RGCN) model implementation.

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import RGCNConv, global_mean_pool
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch.nn import Linear
from types import SimpleNamespace

# ---------------------- GNN Model Definition ---------------------- #


class HeteroGraphRGCN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, encode_table_column, **kwargs):
        """
        Args:
            hidden_channels (int): Number of hidden units.
            out_channels (int): Number of output units.
            num_column_features (int): Number of features for 'column' nodes.
            num_relations (int): Number of relation types.
            dropout (float): Dropout rate.
        """
        super().__init__()

        metadata = (
                ['operator'],  # Node types
                [
                    ('operator', 'calledby', 'operator'),
                ]  # Edge types
            )
        
        if encode_table_column:
            metadata = (
                ['operator', 'table', 'column'],  # Node types
                [
                    ('operator', 'calledby', 'operator'),
                    ('table', 'scannedby', 'operator'),
                    ('column', 'outputby', 'operator'),
                    ('table', 'selfloop', 'table'),
                    ('column', 'selfloop', 'column')
                ]  # Edge types
            )
        self.metadata = metadata
        self.num_layers = num_layers
        self.encode_table_column = encode_table_column
        self.edge_type_mapping = {etype: idx for idx, etype in enumerate(metadata[1])}
        num_relations = len(self.edge_type_mapping)

        # Project node features to hidden_channels with separate linear layers
        kwargs = SimpleNamespace(**kwargs)
        self.lin_operator = Linear(kwargs.num_operator_features, hidden_channels)    
        if encode_table_column:
            self.lin_table = Linear(kwargs.num_table_features, hidden_channels)
            self.lin_column = Linear(kwargs.num_column_features, hidden_channels)

        # RGCNConv layers
        self.conv = RGCNConv(hidden_channels, hidden_channels, num_relations)

        # Final linear layer to produce the output
        self.lin_mem = Linear(hidden_channels, out_channels)
        self.lin_time = Linear(hidden_channels, out_channels)

        # Optional: Layer normalization
        self.norm = nn.LayerNorm(hidden_channels)

        # Dropout layer
        self.dropout = nn.Dropout(kwargs.dropout)

    def forward(self, x_dict, edge_index_dict, batch_operator):
        """
        Args:
            x_dict (dict): Node feature dictionary.
            edge_index_dict (dict): Edge index dictionary.
            edge_type_dict (dict): Edge type dictionary mapping edge types to relation indices.
            batch_operator (Tensor): Batch indices for 'operator' nodes.

        Returns:
            Tensor: Output predictions of shape [batch_size].
        """
        device = next(self.parameters()).device

        # Project node features
        operator = self.lin_operator(x_dict['operator'])
        if self.encode_table_column:
            table = self.lin_table(x_dict['table'])
            column = self.lin_column(x_dict['column'])
        
        x = torch.cat([
            operator
        ], dim=0)

        if self.encode_table_column:
            x = torch.cat([
                operator, 
                table,
                column
            ], dim=0)

        # Create a list to keep track of node type offsets
        node_type_offsets = {}
        offset = 0
        for node_type in self.metadata[0]:
            node_type_offsets[node_type] = offset
            offset += x_dict[node_type].size(0)  # how many nodes of this type

        # Prepare edge_index and edge_type tensors
        all_edge_index = []
        all_edge_type = []
        edge_type_dict = self.edge_type_mapping
        for etype, eindex in edge_index_dict.items():
            # eindex: (2, num_edges)
            # Get the relation index for this edge type
            relation = edge_type_dict[etype]
            # Adjust node indices based on node type offsets
            src_node_type, _, dst_node_type = etype
            src_offset = node_type_offsets[src_node_type]
            dst_offset = node_type_offsets[dst_node_type]
            adjusted_eindex = eindex + torch.tensor([src_offset, dst_offset], device=device).unsqueeze(1)

            all_edge_index.append(adjusted_eindex)
            all_edge_type.append(torch.full((eindex.size(1),), relation, dtype=torch.long, device=device)) 

        if all_edge_index:
            edge_index = torch.cat(all_edge_index, dim=1) # (2, num_edges)
            edge_type = torch.cat(all_edge_type, dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_type = torch.empty((0,), dtype=torch.long, device=device)

        for i in range(self.num_layers):
            x = self.conv(x, edge_index, edge_type)
            x = F.elu(x)
            x = self.norm(x)
            x = self.dropout(x)

        # Extract operator node features
        operator_offset = node_type_offsets['operator']
        num_operators = x_dict['operator'].size(0)
        operator_features = x[operator_offset:operator_offset + num_operators]

        # Global mean pooling over operator nodes per graph
        operator_embedding = global_mean_pool(operator_features, batch_operator)

        # Final output
        mem_pred = self.lin_mem(operator_embedding)
        time_pred = self.lin_time(operator_embedding)
        return mem_pred.squeeze(), time_pred.squeeze()  # Shape: [batch_size]
