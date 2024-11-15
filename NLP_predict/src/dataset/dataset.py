import json
import os
import psycopg2
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import RobustScaler
import pickle
import torch
from torch_geometric.data import Data
from ..utils.database import get_unique_data_types, get_tables, get_relpages_reltuples, get_table_size, get_columns_info, get_column_features, get_db_stats

from .plan_to_graph import parse_query_plan, create_hetero_graph, connect_to_db
from .parse_plan import parse_plan
# import torch
# from transformers import RobertaTokenizer, RobertaModel

# tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
# model = RobertaModel.from_pretrained('microsoft/codebert-base')

def load_json(json_file):
    # json_file = '/home/wuy/DB/pg_mem_data/tpch_sf1/tiny_plans.json' # for debug
    with open(json_file, 'r') as f:
        query_plans = json.load(f)
    return query_plans

    
class QueryPlanDataset(Dataset):
    def __init__(self, dataset, scaler = None):
        super(QueryPlanDataset, self).__init__()

        print(f"loading data...")
        with open(f'{dataset}_data.json', 'r') as f:
            data = json.load(f)


        self.data = []
        self.labels = []
        from tqdm import tqdm
        for d in tqdm(data):
            query_feature = d['query_feature'][0]
            # label = d['query_mem']
            label = d['query_time']
          
            self.data.append(query_feature)
            self.labels.append(label)
        from sklearn.preprocessing import RobustScaler
        self.data = torch.tensor(self.data)
        self.labels = torch.tensor(self.labels)
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = RobustScaler()
            self.scaler.fit(self.data.reshape(-1, 1))
        self.labels = self.scaler.transform(self.labels.reshape(-1, 1)).flatten()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
