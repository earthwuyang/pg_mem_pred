import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from lfm_torch.model import LFModel
from loguru import logger

# Data Transformation Function
def graph_to_sequence(batch_graphs, sequence_length, embedding_dim):
    batch_size = batch_graphs.num_graphs
    sequences = torch.zeros(batch_size, sequence_length, embedding_dim)
    
    for i in range(batch_size):
        node_features = batch_graphs.x[batch_graphs.batch == i]
        num_nodes = node_features.size(0)
        if num_nodes >= sequence_length:
            sequences[i] = node_features[:sequence_length]
        else:
            sequences[i, :num_nodes] = node_features
    return sequences

# LFModel-Based Predictor
class LFMPredictor(nn.Module):
    def __init__(self, 
                 batch_size=32, 
                 seq_length=128, 
                 embedding_dim=512, 
                 token_dim=512, 
                 channel_dim=512, 
                 expert_dim=512, 
                 adapt_dim=128, 
                 num_experts=4):
        super(LFMPredictor, self).__init__()
        self.lf_model = LFModel(
            token_dim=token_dim,
            channel_dim=channel_dim,
            expert_dim=expert_dim,
            adapt_dim=adapt_dim,
            num_experts=num_experts
        )
        
        # Prediction layers
        self.mem_pred_layer = nn.Linear(512, 1)  # Adjust based on LFModel output
        self.time_pred_layer = nn.Linear(512, 1)  # Adjust based on LFModel output

    def forward(self, x):
        logger.info("LFMPredictor forward pass started.")
        lf_output = self.lf_model(x)  # [batch_size, embedding_dim]
        mem_pred = self.mem_pred_layer(lf_output).squeeze(-1)  # [batch_size]
        time_pred = self.time_pred_layer(lf_output).squeeze(-1)  # [batch_size]
        logger.info("LFMPredictor forward pass complete.")
        return mem_pred, time_pred