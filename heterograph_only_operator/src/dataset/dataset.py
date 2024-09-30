import json
import os
import psycopg2
from tqdm import tqdm
from torch_geometric.data import Dataset
import numpy as np
from sklearn.preprocessing import RobustScaler
import pickle
from ..utils.database import get_unique_data_types, get_tables, get_relpages_reltuples, get_table_size, get_columns_info, get_column_features, get_db_stats

from .plan_to_graph import parse_query_plan, create_hetero_graph, connect_to_db

def load_json(json_file):
    # json_file = '/home/wuy/DB/pg_mem_data/tpch_sf1/tiny_plans.json' # for debug
    with open(json_file, 'r') as f:
        query_plans = json.load(f)
    return query_plans

    
class QueryPlanDataset(Dataset):
    def __init__(self, logger, dataset_dir, dataset, mode, scalers, debug):
        super(QueryPlanDataset, self).__init__()
        self.logger = logger
        self.scalers = scalers
        # db_stats_file_path = os.path.join(dataset_dir, dataset, 'database_stats.json')
        # with open(db_stats_file_path, 'r') as f:
        #     self.db_stats = json.load(f)

        dataset_pickle_path = os.path.join('data', f'{dataset}_{mode}_dataset.pkl')
        
        os.makedirs('data', exist_ok=True)

        if os.path.exists(dataset_pickle_path):
            self.logger.info(f"Loading dataset from {dataset_pickle_path}")
            with open(dataset_pickle_path, 'rb') as f:
                self.dataset = pickle.load(f)
            # if mode is 'val' or 'test', they will also use the mem_scaler of train set
        else:
            json_file_path = os.path.join(dataset_dir, dataset, f'{mode}_plans.json')
            if debug: 
                json_file_path = '/home/wuy/DB/pg_mem_data/tpch_sf1/tiny_plans.json' # for debug
            self.logger.info(f"Creating dataset from {json_file_path}")
            self.db_stats = get_db_stats(dataset)
            self.dataset = self.get_dataset(logger, json_file_path, dataset)
            with open(dataset_pickle_path, 'wb') as f:
                pickle.dump(self.dataset, f)


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
            # for idx, plan in enumerate(plans):
            table_nodes, column_nodes, predicate_nodes, operation_nodes, operator_nodes, literal_nodes, numeral_nodes, \
                      table_scannedby_operator_edges, predicate_filters_operator_edges, column_outputby_operator_edges, \
                      column_connects_operation_edges, operator_calledby_operator_edges, operation_filters_operator_edges, operation_connects_predicate_edges,  \
                      literal_connects_operation_edges, numeral_connects_operation_edges, column_containedby_table_edges, \
                      literal_selfloop_literal_edges, numeral_selfloop_numeral_edges,  \
                      table_selfloop_table_edges, column_selfloop_column_edges, predicate_connects_predicate_edges = parse_query_plan(logger, plan, conn, self.db_stats)

            peakmem = self.scalers['peakmem'].transform(np.array([plan['peakmem']]).reshape(-1, 1)).reshape(-1)
            time = self.scalers['time'].transform(np.array(plan['time']).reshape(-1, 1)).reshape(-1)
            graph = create_hetero_graph(logger, 
                table_nodes, column_nodes, predicate_nodes, operation_nodes, operator_nodes, literal_nodes, numeral_nodes, 
                      table_scannedby_operator_edges, predicate_filters_operator_edges, column_outputby_operator_edges, 
                      column_connects_operation_edges, operator_calledby_operator_edges, operation_filters_operator_edges, operation_connects_predicate_edges,  
                      literal_connects_operation_edges, numeral_connects_operation_edges, column_containedby_table_edges, 
                      literal_selfloop_literal_edges, numeral_selfloop_numeral_edges,  
                      table_selfloop_table_edges, column_selfloop_column_edges, predicate_connects_predicate_edges, peakmem, time
            )
            # logger.info(graph)
            self.dataset.append(graph)

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
    json_file_path = '/home/wuy/DB/pg_mem_data/tpch_sf1/tiny_plans.json'
    dataset = 'tpch_sf1'
    dataset = QueryPlanDataset(logger, '/home/wuy/DB/pg_mem_data', dataset, 'train')
    for graph in dataset:
        print(graph)
