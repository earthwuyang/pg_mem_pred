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
    def __init__(self, logger, model, encode_table_column, dataset_dir, dataset, mode, statistics, debug, conn_info):
        super(QueryPlanDataset, self).__init__()
        self.logger = logger
        self.model = model
        self.statistics = statistics
        self.encode_table_column = encode_table_column
        self.conn_info = conn_info

        if debug:
            json_file_path = '/home/wuy/DB/pg_mem_data/tpch_sf1/tiny_plans.json' # for debug
            self.dataset_list = self.get_dataset(logger, json_file_path, dataset)

        else:
            if self.model.startswith('Hetero'):
                dataset_pickle_dir = 'data' + ('_table_column' if self.encode_table_column else '')
                dataset_pickle_path = os.path.join(dataset_pickle_dir, f'{"_".join(dataset)}_{mode}_dataset.pkl')
                
                os.makedirs(dataset_pickle_dir, exist_ok=True)

                if os.path.exists(dataset_pickle_path):
                    self.logger.info(f"Loading dataset from {dataset_pickle_path}")
                    with open(dataset_pickle_path, 'rb') as f:
                        self.dataset_list = pickle.load(f)
                else:
                    self.dataset_list = []
                    if not isinstance(dataset, list):
                        dataset = [dataset]
                    for ds in dataset:
                        json_file_path = os.path.join(dataset_dir, ds, f'{mode}_plans.json')
                  
                        self.logger.info(f"Creating dataset from {json_file_path} for {ds}")
                        self.dataset_list.extend(self.get_dataset(logger, json_file_path, ds))

                    with open(dataset_pickle_path, 'wb') as f:
                        pickle.dump(self.dataset_list, f)
                

            else:   # homogeneous graph
                self.dataset_list = []
                for ds in dataset:
                    json_file_path = os.path.join(dataset_dir, ds, f'{mode}_plans.json')
                    self.logger.info(f"Creating dataset from {json_file_path} for {ds}")
                    self.dataset_list.extend(self.get_dataset(logger, json_file_path, ds))


    def get_dataset(self, logger, json_file_path, dataset):
        # logger.info(f"Processing query plans from {json_file_path}")
        plans = load_json(json_file_path)
    
        # Connect to PostgreSQL
        conn = connect_to_db(self.conn_info)
        if not conn:
            raise Exception("Failed to connect to the database")
            exit(1)  # Exit if connection failed
        # Parse all query plans and create graphs
        graph_list = []
        if self.model.startswith('Hetero'):
            db_stats = get_db_stats(dataset, self.conn_info)
        for idx, plan in tqdm(enumerate(plans), total=len(plans)):
            peakmem = (plan['peakmem'] - self.statistics['peakmem']['center']) / self.statistics['peakmem']['scale']
            time = (plan['time'] - self.statistics['time']['center']) / self.statistics['time']['scale']

            if self.model.startswith('Hetero'):
                graph = create_hetero_graph(logger, plan, conn, self.statistics, db_stats, self.encode_table_column, peakmem, time)
                # logger.info(graph)
                graph_list.append(graph)
            else:   # homogeneous graph
                nodes = []
                edges = []
                parse_plan(plan, self.statistics, nodes=nodes, edges=edges)
        
                # Convert lists to tensors
                x = torch.tensor(nodes, dtype=torch.float)
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                y = torch.tensor(np.array([peakmem, time]), dtype=torch.float)
                data = Data(x=x, edge_index=edge_index, y=y)
                graph_list.append(data)

        # Close the database connection
        conn.close()

        return graph_list

    def __len__(self):
        return len(self.dataset_list)
    
    def __getitem__(self, idx):
        return self.dataset_list[idx]




if __name__ == '__main__':
    import logging
    logger = logging
    json_file_path = '/home/wuy/DB/pg_mem_data/tpch_sf1/tiny_plans.json'
    dataset = 'tpch_sf1'
    dataset = QueryPlanDataset(logger, '/home/wuy/DB/pg_mem_data', dataset, 'train')
    for graph in dataset:
        print(graph)
