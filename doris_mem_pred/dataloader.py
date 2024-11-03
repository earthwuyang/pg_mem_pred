from parse import parse_tree
import torch
from torch_geometric.data import Data, Dataset
import pandas as pd
from tqdm import tqdm
import math
import os
import numpy as np
import json
from sklearn.preprocessing import FunctionTransformer, RobustScaler
# Custom Dataset Class
class PlanDataset(Dataset):
    def __init__(self, dataset):
        super(PlanDataset, self).__init__()
        data_file = 'data.json'
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        

        self.trees = []
        self.labels = []
        for plan in self.data:
            peakmem = int(plan['peak_memory_bytes']) / 1e6 # bytes -> MB
            self.trees.append({"plan": plan['plan'], "peak_memory": peakmem})
            self.labels.append(peakmem)
        # self.scaler = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
        self.scaler = RobustScaler()
        self.labels = self.scaler.fit_transform(np.array(self.labels).reshape(-1, 1)).reshape(-1)

    def len(self):
        return len(self.trees)

    def get(self, idx):

        tree = self.trees[idx]
        plan_str = tree["plan"]
        root, edge_src, edge_tgt, features, node_levels = parse_tree(plan_str)

        # Create edge index tensor
        edge_index = torch.tensor([edge_src, edge_tgt], dtype=torch.long)

        # Node features
        x = torch.tensor(features, dtype=torch.float)

        # Node levels
        node_level_tensor = torch.tensor(node_levels, dtype=torch.long)

        # Label
        y = torch.tensor(self.labels[idx], dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, y=y, node_level=node_level_tensor)
        return data

# Helper Function to Traverse the Tree
def traverse_tree(node):
    yield node
    for child in node.children:
        yield from traverse_tree(child)


if __name__ == '__main__':
    dataset = PlanDataset('tpch')
    print(dataset.data)