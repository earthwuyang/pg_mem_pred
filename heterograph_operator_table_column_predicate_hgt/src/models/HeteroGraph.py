# Relational Graph Convolutional Network (RGCN) model implementation.

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import RGCNConv, global_mean_pool, HGTConv
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch.nn import Linear

# ---------------------- GNN Model Definition ---------------------- #


class HeteroGraph(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        out_channels,
        num_layers,
        num_operator_features, 
        num_table_features, 
        num_column_features, 
        num_predicate_features, 
        num_operation_features, 
        num_literal_features, 
        num_numeral_features,
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
                    ('operator', 'calledby', 'operator'),
                    ('table', 'scannedby', 'operator'),
                    ('column', 'containedby', 'table'),
                    ('column', 'outputby', 'operator'),
                    ('column', 'selfloop', 'column'),
                    ('predicate', 'filters', 'operator'),
                    ('operation', 'filters', 'operator'),
                    ('operation', 'connects', 'predicate'),
                    ('predicate', 'connects', 'predicate'),
                    ('column', 'connects', 'operation'),
                    ('literal', 'connects', 'operation'),
                    ('numeral', 'connects', 'operation'),
                    ('literal', 'selfloop', 'literal'),
                    ('numeral', 'selfloop', 'numeral'),
                ]  # Edge types
            )
        self.metadata = metadata
        
        self.edge_type_mapping = {etype: idx for idx, etype in enumerate(metadata[1])}
        num_relations = len(self.edge_type_mapping)
        num_node_types = len(metadata[0])

        self.num_layers = num_layers
        # Project node features to hidden_channels with separate linear layers
        self.lin_operator = Linear(num_operator_features, hidden_channels)    
        self.lin_table = Linear(num_table_features, hidden_channels)       
        self.lin_column = Linear(num_column_features, hidden_channels)  
        self.lin_predicate = Linear(num_predicate_features, hidden_channels)  
        self.lin_operation = Linear(num_operation_features, hidden_channels)  
        self.lin_literal = Linear(num_literal_features, hidden_channels)     
        self.lin_numeral = Linear(num_numeral_features, hidden_channels)     

        # Total number of node types
        self.num_node_types = num_node_types

        # RGCNConv layers
        num_heads = 4
        self.conv = HGTConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            metadata=metadata,
            heads=num_heads,
        )

        # Final linear layer to produce the output
        self.lin_mem = Linear(hidden_channels, out_channels)
        self.lin_time = Linear(hidden_channels, out_channels)

        # Optional: Layer normalization
        self.norm = nn.LayerNorm(hidden_channels)

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

        # # Project node features
        # operator = self.lin_operator(x_dict['operator'])
        # table = self.lin_table(x_dict['table'])
        # column = self.lin_column(x_dict['column'])
        # predicate = self.lin_predicate(x_dict['predicate'])
        # operation = self.lin_operation(x_dict['operation'])
        # literal = self.lin_literal(x_dict['literal'])
        # numeral = self.lin_numeral(x_dict['numeral'])

        # # Concatenate all node features into a single tensor
        # x = torch.cat([
        #     operator,
        #     table,
        #     column,
        #     predicate,
        #     operation,
        #     literal,
        #     numeral
        # ], dim=0)

         # Project node features with checks
        projected_x = {}
        for node_type, lin_layer in [('operator', self.lin_operator),
                                     ('table', self.lin_table),
                                     ('column', self.lin_column),
                                     ('predicate', self.lin_predicate),
                                     ('operation', self.lin_operation),
                                     ('literal', self.lin_literal),
                                     ('numeral', self.lin_numeral)]:
            if node_type in x_dict and x_dict[node_type].shape[0] > 0:
                projected_x[node_type] = lin_layer(x_dict[node_type])
            else:
                # Assign an empty tensor with appropriate feature size
                projected_x[node_type] = torch.empty(
                    (0, lin_layer.out_features), 
                    device=x_dict[next(iter(x_dict))].device
                )
        
        x_dict = projected_x
  
        # # Create a list to keep track of node type offsets
        # node_type_offsets = {}
        # offset = 0
        # for node_type in self.metadata[0]:
        #     node_type_offsets[node_type] = offset
        #     offset += x_dict[node_type].size(0)  # how many nodes of this type

        # # Prepare edge_index and edge_type tensors
        # all_edge_index = []
        # all_edge_type = []
        # edge_type_dict = self.edge_type_mapping
        # for etype, eindex in edge_index_dict.items():
        #     # eindex: (2, num_edges)
        #     # Get the relation index for this edge type
        #     relation = edge_type_dict[etype]
        #     # Adjust node indices based on node type offsets
        #     src_node_type, _, dst_node_type = etype
        #     src_offset = node_type_offsets[src_node_type]
        #     dst_offset = node_type_offsets[dst_node_type]
        #     adjusted_eindex = eindex + torch.tensor([src_offset, dst_offset], device=device).unsqueeze(1)

        #     all_edge_index.append(adjusted_eindex)
        #     all_edge_type.append(torch.full((eindex.size(1),), relation, dtype=torch.long, device=device)) 

        # if all_edge_index:
        #     edge_index = torch.cat(all_edge_index, dim=1) # (2, num_edges)
        #     edge_type = torch.cat(all_edge_type, dim=0)
        # else:
        #     edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        #     edge_type = torch.empty((0,), dtype=torch.long, device=device)

        for i in range(self.num_layers):
            x = self.conv(x_dict, edge_index_dict)
            x_dict = {key: self.norm(F.elu(x)) for key, x in x_dict.items()}

        # Aggregate operator node features per graph
        operator_features = x_dict['operator']

        # Global mean pooling over operator nodes per graph
        operator_embedding = global_mean_pool(operator_features, batch_operator)

        # Final output
        mem_pred = self.lin_mem(operator_embedding)
        time_pred = self.lin_time(operator_embedding)
        return mem_pred.squeeze(), time_pred.squeeze()
