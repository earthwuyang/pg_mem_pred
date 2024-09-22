import json
import os
import psycopg2
from tqdm import tqdm
from torch_geometric.data import Dataset
import pickle

from .plan_to_graph import parse_query_plan, create_hetero_graph, connect_to_db

def load_json(json_file):
    with open(json_file, 'r') as f:
        query_plans = json.load(f)
    return query_plans


    
class QueryPlanDataset(Dataset):
    def __init__(self, logger, dataset_dir, dataset, mode, data_type_mapping):
        super(QueryPlanDataset, self).__init__()
        self.logger = logger
        json_file_path = os.path.join(dataset_dir, dataset, f'{mode}_plans.json')
        dataset_pickle_path = os.path.join('data', f'{dataset}_{mode}_dataset.pkl')
        os.makedirs('data', exist_ok=True)
        if os.path.exists(dataset_pickle_path):
            self.logger.info(f"Loading dataset from {dataset_pickle_path}")
            with open(dataset_pickle_path, 'rb') as f:
                self.dataset = pickle.load(f)
        else:
            self.logger.info(f"Creating dataset from {json_file_path}")
            self.dataset = self.get_dataset(logger, json_file_path, dataset, data_type_mapping)
            with open(dataset_pickle_path, 'wb') as f:
                pickle.dump(self.dataset, f)

    def get_dataset(self, logger, json_file_path, dataset, data_type_mapping):
        # logger.info(f"Processing query plans from {json_file_path}")
        plans = load_json(json_file_path)

        if dataset == 'tpch':
            database = 'tpc_h'
        elif dataset == 'tpcds':
            database = 'tpc_ds'
        else:
            raise ValueError(f'Invalid dataset name {dataset}')
    
        # Database connection details
        DB_CONFIG = {
            'dbname': database,
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
            table_nodes, column_nodes, predicate_nodes, operator_nodes, \
            table_scannedby_operator_edges, predicate_filters_operator_edges, \
            column_outputby_operator_edges, column_connects_predicate_edges, \
            operator_calledby_operator_edges = parse_query_plan(plan['Plan'], conn, data_type_mapping)

            
            graph = create_hetero_graph(
                table_nodes, column_nodes, predicate_nodes, operator_nodes,
                table_scannedby_operator_edges, predicate_filters_operator_edges, column_outputby_operator_edges,
                column_connects_predicate_edges, operator_calledby_operator_edges, 
                plan['peakmem']
            )
            self.dataset.append(graph)

        # print(f"Number of query plans: {len(self.dataset)}")

        # Close the database connection
        conn.close()
        # save dataset to file
        # with open(os.path.join(os.path.dirname(json_file_path), 'dataset.pkl'), 'wb') as f:
        #     pickle.dump(self.dataset, f)

        return self.dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]




if __name__ == '__main__':
    import logging
    logger = logging
    json_file_path = '/home/wuy/DB/pg_mem_data/tpch/val_plans.json'
    dataset = 'tpch'
    dataset = QueryPlanDataset(logger, json_file_path, dataset)
    print(dataset[0])
