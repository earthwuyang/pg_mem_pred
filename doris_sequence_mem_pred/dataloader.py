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
    def __init__(self, statistics, dataset, K = 10): # k-segments
        super(PlanDataset, self).__init__()
        data_dir = f'/home/wuy/DB/doris_mem_pred/{dataset}_data/'
        data_file = os.path.join(data_dir, f'query_mem_data_{dataset}_sf100.csv')
        plan_dir = os.path.join(data_dir, 'plans')
        df=pd.read_csv(data_file, sep=';')
        

        self.trees = []
        self.time_labels = []
        self.mem_labels = []
        self.labels = []
        for row in tqdm(df.iterrows(), total=len(df)):
            row = row[1]
            query_id = row['queryid']
            runtime = row['time'] # int
            runtime = (runtime - statistics['time']['center']) / statistics['time']['scale']
            mem_list = eval(row['mem_list'].strip()) # list of int
            mem_list = [(mem - statistics['mem']['center']) / statistics['mem']['scale'] for mem in mem_list]
            plan_file = os.path.join(plan_dir, f'{query_id}.txt')
            with open(plan_file, 'r') as f:
                plan = f.readlines()
            self.trees.append({"query_id": query_id, "plan": plan})
            self.time_labels.append(runtime)
            self.mem_labels.append(mem_list)
            self.labels.append([runtime] + mem_list)
        # print(f"Total {len(self.trees)} plans loaded.")

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

        data = Data(x=x, edge_index=edge_index, node_level=node_level_tensor, mem_label=self.mem_labels[idx], time_label=self.time_labels[idx], y = self.labels[idx])
        return data

# Helper Function to Traverse the Tree
def traverse_tree(node):
    yield node
    for child in node.children:
        yield from traverse_tree(child)


if __name__ == '__main__':
    with open('statistics.json', 'r') as f:
        statistics = json.load(f)
    dataset = PlanDataset(statistics, 'tpch')
    for data in dataset:
        assert len(data.y) == 11, "Label should have 11 dimensions"
        