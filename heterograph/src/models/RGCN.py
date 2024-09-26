# Relational Graph Convolutional Network (RGCN) model implementation.

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import RGCNConv, global_mean_pool
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch.nn import Linear

# ---------------------- GNN Model Definition ---------------------- #


class HeteroGraph(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        out_channels,
        num_column_features=10,
        dropout=0.2,
    ):
        """
        Args:
            hidden_channels (int): Number of hidden units.
            out_channels (int): Number of output units.
            num_column_features (int): Number of features for 'column' nodes.
            num_relations (int): Number of relation types.
            num_node_types (int): Number of node types.
            dropout (float): Dropout rate.
        """
        super().__init__()

        metadata = (
                ['operator', 'table', 'column', 'predicate', 'operation', 'literal', 'numeral'],  # Node types
                [
                    ('table', 'scannedby', 'operator'),
                    ('predicate', 'filters', 'operator'),
                    ('column', 'outputby', 'operator'),
                    ('column', 'connects', 'operation'),
                    ('operator', 'calledby', 'operator'),
                    ('operation', 'filters', 'operator'),
                    ('operation', 'connects', 'predicate'),
                    ('literal', 'connects', 'operation'),
                    ('numeral', 'connects', 'operation'),
                    ('literal', 'selfloop', 'literal'),
                    ('numeral', 'selfloop', 'numeral'),
                    ('table', 'selfloop', 'table'),
                    ('column', 'selfloop', 'column'),
                    ('predicate', 'selfloop', 'predicate')
                ]  # Edge types
            )
        self.edge_type_mapping = {etype: idx for idx, etype in enumerate(metadata[1])}
        num_relations = len(self.edge_type_mapping)
        num_node_types = len(metadata[0])

        # Project node features to hidden_channels with separate linear layers
        self.lin_operator = Linear(4, hidden_channels)    # Operators have 4 features
        self.lin_table = Linear(2, hidden_channels)       # Tables have 2 features
        self.lin_column = Linear(num_column_features, hidden_channels)  # Column size + one-hot data types
        self.lin_predicate = Linear(4, hidden_channels)   # Predicates have 4 features
        self.lin_operation = Linear(8, hidden_channels)   # Operations have 8 features
        self.lin_literal = Linear(1, hidden_channels)     # Literals have 1 feature
        self.lin_numeral = Linear(1, hidden_channels)     # Numerals have 1 feature

        # Total number of node types
        self.num_node_types = num_node_types

        # RGCNConv layers
        self.conv1 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)

        # Final linear layer to produce the output
        self.lin = Linear(hidden_channels, out_channels)

        # Optional: Layer normalization
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

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
        table = self.lin_table(x_dict['table'])
        column = self.lin_column(x_dict['column'])
        predicate = self.lin_predicate(x_dict['predicate'])
        operation = self.lin_operation(x_dict['operation'])
        literal = self.lin_literal(x_dict['literal'])
        numeral = self.lin_numeral(x_dict['numeral'])

        # Concatenate all node features into a single tensor
        x = torch.cat([
            operator,
            table,
            column,
            predicate,
            operation,
            literal,
            numeral
        ], dim=0)

        # Create a list to keep track of node type offsets
        node_type_offsets = {}
        offset = 0
        for node_type in ['operator', 'table', 'column', 'predicate', 'operation', 'literal', 'numeral']:
            node_type_offsets[node_type] = offset
            offset += x_dict[node_type].size(0)

        # Prepare edge_index and edge_type tensors
        all_edge_index = []
        all_edge_type = []
        edge_type_dict = self.edge_type_mapping
        for etype, eindex in edge_index_dict.items():
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
            edge_index = torch.cat(all_edge_index, dim=1)
            edge_type = torch.cat(all_edge_type, dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_type = torch.empty((0,), dtype=torch.long, device=device)

        # First RGCNConv layer
        x = self.conv1(x, edge_index, edge_type)
        x = F.elu(x)
        x = self.norm1(x)
        x = self.dropout(x)

        # Second RGCNConv layer
        x = self.conv2(x, edge_index, edge_type)
        x = F.elu(x)
        x = self.norm2(x)
        x = self.dropout(x)

        # Extract operator node features
        operator_offset = node_type_offsets['operator']
        num_operators = x_dict['operator'].size(0)
        operator_features = x[operator_offset:operator_offset + num_operators]

        # Global mean pooling over operator nodes per graph
        operator_embedding = global_mean_pool(operator_features, batch_operator)

        # Final output
        out = self.lin(operator_embedding)
        return out.squeeze()  # Shape: [batch_size]