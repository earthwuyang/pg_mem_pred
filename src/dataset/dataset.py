import json
import os
import psycopg2
from tqdm import tqdm
from torch_geometric.data import Dataset
import numpy as np
from sklearn.preprocessing import RobustScaler
import pickle
import torch
from torch_geometric.data import Data
from ..utils.database import get_unique_data_types, get_tables, get_relpages_reltuples, get_table_size, get_columns_info, get_column_features, get_db_stats

from .plan_to_graph import parse_query_plan, create_hetero_graph, connect_to_db
from .parse_plan import parse_plan


def load_json(json_file):
    # json_file = '/home/wuy/DB/pg_mem_data/tpch_sf1/tiny_plans.json' # for debug
    with open(json_file, 'r') as f:
        query_plans = json.load(f)
    return query_plans

    
class QueryPlanDataset(Dataset):
    def __init__(self, logger, model, encode_schema, dataset_dir, dataset, mode, statistics, debug):
        super(QueryPlanDataset, self).__init__()
        self.logger = logger
        self.model = model
        self.statistics = statistics
        self.encode_schema = encode_schema

        if self.encode_schema:
            schema_file_path = os.path.join(dataset_dir, dataset, 'schema.json')
            with open(schema_file_path, 'r') as f:
                self.schema = json.load(f)

        if model.startswith('Hetero'):
            self.db_stats = get_db_stats(dataset)

        if debug:
            json_file_path = '/home/wuy/DB/pg_mem_data/tpch_sf1/tiny_plans.json' # for debug
            self.dataset = self.get_dataset(logger, json_file_path, dataset)

        else:
            if self.model.startswith('Hetero'):
                dataset_pickle_path = os.path.join('data', f'{dataset}_{mode}_dataset.pkl')
                
                os.makedirs('data', exist_ok=True)

                if os.path.exists(dataset_pickle_path):
                    self.logger.info(f"Loading dataset from {dataset_pickle_path}")
                    with open(dataset_pickle_path, 'rb') as f:
                        self.dataset = pickle.load(f)
                else:
                    json_file_path = os.path.join(dataset_dir, dataset, f'{mode}_plans.json')
                    self.logger.info(f"Creating dataset from {json_file_path}")

                    self.dataset = self.get_dataset(logger, json_file_path, dataset)
                    with open(dataset_pickle_path, 'wb') as f:
                        pickle.dump(self.dataset, f)
                

            else:   # homogeneous graph
                json_file_path = os.path.join(dataset_dir, dataset, f'{mode}_plans.json')
                self.logger.info(f"Creating dataset from {json_file_path}")
                self.dataset = self.get_dataset(logger, json_file_path, dataset)


    def get_dataset(self, logger, json_file_path, dataset):
        # logger.info(f"Processing query plans from {json_file_path}")
        plans = load_json(json_file_path)
    
        # Database connection details
        DB_CONFIG = {
            'dbname': dataset,
            'user': 'wuy',
            'password': '',
            'host': 'localhost',  # e.g., 'localhost'
        }
        # Connect to PostgreSQL
        conn = connect_to_db(DB_CONFIG)
        if not conn:
            raise Exception("Failed to connect to the database")
            exit(1)  # Exit if connection failed
        # Parse all query plans and create graphs
        self.dataset = []
        for idx, plan in tqdm(enumerate(plans), total=len(plans)):
            peakmem = (plan['peakmem'] - self.statistics['peakmem']['center']) / self.statistics['peakmem']['scale']
            time = (plan['time'] - self.statistics['time']['center']) / self.statistics['time']['scale']

            if self.model.startswith('Hetero'):
                graph = create_hetero_graph(logger, plan, conn, self.statistics, self.db_stats, self.schema if self.encode_schema else None, self.encode_schema, peakmem, time)
                # logger.info(graph)
                self.dataset.append(graph)
            else:   # homogeneous graph
                nodes = []
                edges = []
                parse_plan(plan, self.statistics, nodes=nodes, edges=edges)
        
                # Convert lists to tensors
                x = torch.tensor(nodes, dtype=torch.float)
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                y = torch.tensor(np.array([peakmem, time]), dtype=torch.float)
                data = Data(x=x, edge_index=edge_index, y=y)
                self.dataset.append(data)

        # Close the database connection
        conn.close()

        return self.dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]




if __name__ == '__main__':
    import logging
    logger = logging
    json_file_path = '/home/wuy/DB/pg_mem_data/tpch_sf1/tiny_plans.json'
    dataset = 'tpch_sf1'
    dataset = QueryPlanDataset(logger, '/home/wuy/DB/pg_mem_data', dataset, 'train')
    for graph in dataset:
        print(graph)
