import json
import os
import psycopg2
from tqdm import tqdm
from torch_geometric.data import Dataset
import numpy as np
from sklearn.preprocessing import RobustScaler
import pickle
from ..utils.database import get_unique_data_types, get_tables, get_relpages_reltuples, get_table_size, get_columns_info, get_column_features

from .plan_to_graph import parse_query_plan, create_hetero_graph, connect_to_db

def load_json(json_file):
    json_file = '/home/wuy/DB/pg_mem_data/tpch/tiny_plans.json' # for debug
    with open(json_file, 'r') as f:
        query_plans = json.load(f)
    return query_plans


def get_db_stats(dataset):
    database = dataset
    db_stats = {}
    conn = psycopg2.connect(database=database, user="wuy", password='', host='localhost')

    unique_data_types = get_unique_data_types(conn)
    db_stats['unique_data_types'] = unique_data_types
    db_stats['tables'] = {}

    relpages_list = []
    reltuples_list = []
    table_size_list = []

    # Initialize lists for scaling
    avg_widths = []
    correlations = []
    n_distincts = []
    null_fracs = []
    
    tables = get_tables(conn)
    for table in tables:
        relpages, reltuples = get_relpages_reltuples(conn, table)
        table_size = get_table_size(conn, table)

        # Collect relpages and reltuples for scaling
        relpages_list.append(relpages)
        reltuples_list.append(reltuples)
        table_size_list.append(table_size)

        
        db_stats['tables'][table] = {'relpages': relpages, 'reltuples': reltuples, 'table_size': table_size}
        db_stats['tables'][table]['column_features'] = {}


        columns = get_columns_info(conn, table)
        for column in columns:
            column_name = column[0]
            data_type = column[1]
            avg_width, correlation, n_distinct, null_frac, data_type = get_column_features(conn, table, column_name)

            # Collect values for scaling
            avg_widths.append(avg_width)
            correlations.append(correlation)
            n_distincts.append(n_distinct)
            null_fracs.append(null_frac)

            db_stats['tables'][table]['column_features'][f"{table}.{column_name}"] = {
                'avg_width': avg_width,
                'correlation': correlation,
                'n_distinct': n_distinct,
                'null_frac': null_frac,
                'data_type': data_type
            }
        

    # Scale the collected values using RobustScaler
    column_scaler = RobustScaler()
    table_scaler = RobustScaler()
    column_scaled_features = column_scaler.fit_transform(np.array([avg_widths, correlations, n_distincts, null_fracs]).T)
    table_scaled_features = table_scaler.fit_transform(np.array([relpages_list, reltuples_list, table_size_list]).T)


    # Assign scaled values back to the db_stats
    for i, column in enumerate(columns):
        column_name = column[0]
        db_stats['tables'][table]['column_features'][f"{table}.{column_name}"]['avg_width'] = column_scaled_features[i][0]
        db_stats['tables'][table]['column_features'][f"{table}.{column_name}"]['correlation'] = column_scaled_features[i][1]
        db_stats['tables'][table]['column_features'][f"{table}.{column_name}"]['n_distinct'] = column_scaled_features[i][2]
        db_stats['tables'][table]['column_features'][f"{table}.{column_name}"]['null_frac'] = column_scaled_features[i][3]

    # Update relpages and reltuples in db_stats
    for i, table_name in enumerate(tables):
        db_stats['tables'][table_name]['relpages'] = table_scaled_features[i][0]
        db_stats['tables'][table_name]['reltuples'] = table_scaled_features[i][1]
        db_stats['tables'][table_name]['table_size'] = table_scaled_features[i][2]


    return db_stats
    
class QueryPlanDataset(Dataset):
    def __init__(self, logger, dataset_dir, dataset, mode, mem_scaler):
        super(QueryPlanDataset, self).__init__()
        self.logger = logger
        self.mem_scaler = mem_scaler
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
                      literal_connects_operation_edges, numeral_connects_operation_edges, \
                      literal_selfloop_literal_edges, numeral_selfloop_numeral_edges,  \
                      table_selfloop_table_edges, column_selfloop_column_edges, predicate_connects_predicate_edges = parse_query_plan(logger, plan, conn, self.db_stats)

            graph = create_hetero_graph(logger, 
                table_nodes, column_nodes, predicate_nodes, operation_nodes, operator_nodes, literal_nodes, numeral_nodes, 
                      table_scannedby_operator_edges, predicate_filters_operator_edges, column_outputby_operator_edges, 
                      column_connects_operation_edges, operator_calledby_operator_edges, operation_filters_operator_edges, operation_connects_predicate_edges,  
                      literal_connects_operation_edges, numeral_connects_operation_edges, 
                      literal_selfloop_literal_edges, numeral_selfloop_numeral_edges,  
                      table_selfloop_table_edges, column_selfloop_column_edges, predicate_connects_predicate_edges, plan['peakmem'], self.mem_scaler
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
    json_file_path = '/home/wuy/DB/pg_mem_data/tpch/tiny_plans.json'
    dataset = 'tpch_sf1'
    dataset = QueryPlanDataset(logger, '/home/wuy/DB/pg_mem_data', dataset, 'train')
    for graph in dataset:
        print(graph)
