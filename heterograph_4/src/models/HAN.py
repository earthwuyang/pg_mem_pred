import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv, HANConv, global_mean_pool, LayerNorm
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch.nn import Linear

# ---------------------- HAN Model Definition ---------------------- #

class HeteroGraphHAN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_column_features, metadata, dropout=0.2):
        """
        Args:
            hidden_channels (int): Number of hidden units.
            out_channels (int): Number of output units.
            num_column_features (int): Number of features for 'column' nodes.
            metadata (tuple): A tuple containing node types and edge types.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.metadata = metadata

        # Project node features to hidden_channels
        self.lin_operator = Linear(4, hidden_channels)    # Operators have 4 features
        self.lin_table = Linear(2, hidden_channels)       # Tables have 2 features
        self.lin_column = Linear(num_column_features, hidden_channels)  # Column size + one-hot data types
        self.lin_predicate = Linear(4, hidden_channels)   # Predicates have 4 features
        self.lin_operation = Linear(8, hidden_channels)   # Operations have 8 features
        self.lin_literal = Linear(1, hidden_channels)    
        self.lin_numeral = Linear(1, hidden_channels)    

        # Define meta-paths based on the actual edge types
        self.meta_paths = [
            # 1. Direct operator invocation
            [('operator', 'calledby', 'operator')],

            # 2. Operators connected via tables
            [('operator', 'scannedby', 'table'), ('table', 'scannedby', 'operator')],

            # 3. Operators connected via predicates
            [('operator', 'filters', 'predicate'), ('predicate', 'filters', 'operator')],

            # 4. Operators connected via columns
            [('operator', 'outputby', 'column'), ('column', 'outputby', 'operator')],

            # 5. Operators connected through operations that filter operators
            [('operator', 'connects', 'operation'), ('operation', 'filters', 'operator')],

            # 6. Operators connected through operations and predicates
            [('operator', 'connects', 'operation'), ('operation', 'connects', 'predicate'), ('predicate', 'filters', 'operator')],

            # 7. Operators connected through operations and columns
            [('operator', 'connects', 'operation'), ('operation', 'connects', 'column'), ('column', 'outputby', 'operator')],

            # 8. Operators connected through operations and literals
            [('operator', 'connects', 'operation'), ('operation', 'connects', 'literal'), ('literal', 'connects', 'operation'), ('operation', 'filters', 'operator')],

            # 9. Operators connected through operations and numerals
            [('operator', 'connects', 'operation'), ('operation', 'connects', 'numeral'), ('numeral', 'connects', 'operation'), ('operation', 'filters', 'operator')],

            # 10. Operators connected via multiple 'calledby' relationships
            [('operator', 'calledby', 'operator'), ('operator', 'calledby', 'operator')]
        ]

        # Initialize HANConv layers
        self.conv1 = HANConv(
            metadata=metadata,
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            heads=4,
            meta_paths=self.meta_paths,
            dropout=dropout
        )

        self.conv2 = HANConv(
            metadata=metadata,
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            heads=4,
            meta_paths=self.meta_paths,
            dropout=dropout
        )

        # Final linear layer to produce the output
        self.lin = Linear(hidden_channels, out_channels)

        # Optional: Layer normalization and dropout
        self.norm1 = LayerNorm(hidden_channels)
        self.norm2 = LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict, batch_operator):
        """
        Args:
            x_dict (dict): Node feature dictionary.
            edge_index_dict (dict): Edge index dictionary.
            batch_operator (Tensor): Batch indices for 'operator' nodes.

        Returns:
            Tensor: Output predictions of shape [batch_size].
        """
        # Project node features
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

        # First HANConv layer
        x_dict = self.conv1(x_dict, edge_index_dict, self.meta_paths)
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        x_dict = {key: self.norm1(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        # Second HANConv layer
        x_dict = self.conv2(x_dict, edge_index_dict, self.meta_paths)
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        x_dict = {key: self.norm2(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        # Aggregate operator node features per graph
        operator_features = x_dict['operator']
        # Global mean pooling over operator nodes per graph
        operator_embedding = global_mean_pool(operator_features, batch_operator)
        
        # Final output
        out = self.lin(operator_embedding)
        return out.squeeze()  # Shape: [batch_size]