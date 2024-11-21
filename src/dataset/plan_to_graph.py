import torch
import re
import psycopg2
from torch_geometric.data import HeteroData
from moz_sql_parser import parse
from sklearn.preprocessing import RobustScaler
import numpy as np
import json
import os
from ..utils.database import get_relpages_reltuples, get_table_size, get_columns_info, get_column_features, get_unique_data_types, get_tables, get_db_stats


    
# ---------------------- Helper Functions ---------------------- #
# One-hot encode data types with an 'unknown' category
def one_hot_encode_data_type(data_type, data_type_mapping):
    one_hot = [0] * (len(data_type_mapping) + 1)  # +1 for 'unknown'
    if data_type in data_type_mapping:
        index = data_type_mapping[data_type]
        one_hot[index] = 1
    else:
        # Assign 'unknown' category
        one_hot[-1] = 1
    assert len(one_hot) == (len(data_type_mapping) + 1), f"One-hot encoding should have {(len(data_type_mapping) + 1)} dimensions"
    return one_hot


def extract_columns(string):
    # Regex to extract column names with at least one letter in the table/alias part
    # column_pattern = re.compile(r'\b([A-Za-z]+\.\w+)\b')
    column_pattern = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*\.\w+)\b')
    columns = column_pattern.findall(string)

    return columns

def one_hot_encode(n, num_classes):
    """
    One-hot encode a number into a binary vector of length num_classes.
    
    Args:
        n (int): The number to encode.
        num_classes (int): The number of classes.
        
    Returns:
        one_hot (list): A binary vector of length num_classes.
    """
    one_hot = [0] * num_classes
    one_hot[n] = 1
    return one_hot

def extract_features(plan_node, statistics):
    """
    Extract relevant features from a plan node.
    
    Args:
        plan_node (dict): A single plan node from the JSON.
        
    Returns:
        feature_vector (list): A list of numerical features.
    """
    # Define which features to extract
    feature_vector = []
    for key in ['Startup Cost', 'Total Cost', 'Plan Rows', 'Plan Width', 'Node Type']:
        if statistics[key]['type'] == 'numeric':
            value = ( plan_node[key] - statistics[key]['center']) / statistics[key]['scale']
            feature_vector.append(value)
        elif statistics[key]['type'] == 'categorical':
            value = plan_node.get(key, 'unknown')
            one_hot_features = one_hot_encode(statistics[key]['value_dict'].get(value, statistics[key]['no_vals']), statistics[key]['no_vals']+1)  # unknown will map to an extra number in the directory
            feature_vector.extend(one_hot_features)
   
    return feature_vector

# Helper function to traverse operators and extract tables, columns, and predicates
def traverse_operators(statistics, db_stats, data_type_mapping, plan, encode_table_column, operator_nodes, table_nodes, column_nodes,
                       operator_calledby_operator_edges, table_scannedby_operator_edges, column_outputby_operator_edges, 
                       table_selfloop_table_edges, column_selfloop_column_edges, 
                        operator_id_counter, parent_operator_id=None):
    current_operator_id = operator_id_counter[0]
    operator_id_counter[0] += 1  # Increment the operator ID counter

    if 'Plan' in plan:
        plan_parameters = plan.get('Plan', {})
    else:
        plan_parameters = plan


    # # Extract operator features
    operator_features = extract_features(plan_parameters, statistics) 
    operator_nodes.append({
        'id': current_operator_id,
        'features': operator_features
    })
  
    # If there is a parent operator, add an edge (operator calls operator)
    if parent_operator_id is not None:
        operator_calledby_operator_edges.append((current_operator_id, parent_operator_id))

    if encode_table_column:
        # Extract tables, columns, predicates, and edges
        if 'Relation Name' in plan_parameters:
            table_name = plan_parameters['Relation Name']
            if table_name not in table_nodes:
                # print(f"db_stats['tables'].keys(): {db_stats['tables'].keys()}")
                relpages, reltuples = db_stats['tables'][table_name]['relpages'], db_stats['tables'][table_name]['reltuples']
                features = [relpages, reltuples]
                table_nodes[table_name] = {
                    'id': len(table_nodes),
                    'features': features
                }
            table_id = table_nodes[table_name]['id']
            table_scannedby_operator_edges.append((table_id, current_operator_id))

        output_list = plan_parameters.get('Output', [])
        for output_item in output_list:
            # print(f"output_item {output_item}")
            cols = extract_columns(output_item)
            for col in cols:
                if col not in column_nodes:
                    table_name = col.split('.')[0]
                    # print(f"table_name: {table_name}, col: {col}")
                    # print(f"db_stats['tables'].keys {db_stats['tables'].keys()}")
                    if table_name not in db_stats['tables']:
                        table_name = table_name[:-2]
                    column_name = col.split('.')[1]
                    column_name = table_name + '.' + column_name
                    # print(f"table_name: {table_name}, column_name: {column_name}")
                    # print(f"db_stats['tables'].keys() {db_stats['tables'].keys()}")
                    avg_width = db_stats['tables'][table_name]['column_features'][column_name]['avg_width']
                    correlation = db_stats['tables'][table_name]['column_features'][column_name]['correlation']
                    n_distinct = db_stats['tables'][table_name]['column_features'][column_name]['n_distinct']
                    null_frac = db_stats['tables'][table_name]['column_features'][column_name]['null_frac']
                    data_type = db_stats['tables'][table_name]['column_features'][column_name]['data_type']
                    one_hot = one_hot_encode_data_type(data_type, data_type_mapping)  # Unique data types: {'character': 0, 'character varying': 1, 'date': 2, 'integer': 3, 'numeric': 4, 'double precision': 5}
                    features = [avg_width, correlation, n_distinct, null_frac] + one_hot
                    column_nodes[col] = {
                        'id': len(column_nodes),
                        'features': features
                    }
                column_id = column_nodes[col]['id']
                # Add edge: column is output by operator
                column_outputby_operator_edges.append((column_id, current_operator_id))

    # Recurse into sub-plans
    if 'Plans' in plan_parameters:
        for sub_plan in plan_parameters['Plans']:
            traverse_operators(statistics, db_stats, data_type_mapping, sub_plan, encode_table_column, operator_nodes, table_nodes, column_nodes, 
                               operator_calledby_operator_edges, table_scannedby_operator_edges, column_outputby_operator_edges, 
                               table_selfloop_table_edges, column_selfloop_column_edges, 
                               operator_id_counter, current_operator_id)


def create_schema_graph(schema, db_stats, data_type_mapping, table_nodes, column_nodes, column_containedby_table_edges, column_referencedby_column_edges):
    for table_name in db_stats['tables'].keys():
        relpages, reltuples = db_stats['tables'][table_name]['relpages'], db_stats['tables'][table_name]['reltuples']
        table_nodes[table_name] = {
            'id': len(table_nodes),
            'features': [relpages, reltuples]  # Placeholder, will be updated later
        }
        for column_name in db_stats['tables'][table_name]['column_features'].keys():
            # column_name = f"{table_name}.{column_name}"
            avg_width = db_stats['tables'][table_name]['column_features'][column_name]['avg_width']
            correlation = db_stats['tables'][table_name]['column_features'][column_name]['correlation']
            n_distinct = db_stats['tables'][table_name]['column_features'][column_name]['n_distinct']
            null_frac = db_stats['tables'][table_name]['column_features'][column_name]['null_frac']
            data_type = db_stats['tables'][table_name]['column_features'][column_name]['data_type']
            one_hot = one_hot_encode_data_type(data_type, data_type_mapping)  # Unique data types: {'character': 0, 'character varying': 1, 'date': 2, 'integer': 3, 'numeric': 4}
            features = [avg_width, correlation, n_distinct, null_frac] + one_hot

            column_nodes[column_name] = {
                'id': len(column_nodes),
                'features': features  # Placeholder, will be updated later
            }
            column_id = column_nodes[column_name]['id']
            table_id = table_nodes[table_name]['id']
            column_containedby_table_edges.append((column_id, table_id))

    for referencing_table, referencing_column, referenced_table, referenced_column in schema['relationships']:
        if isinstance(referencing_column, list) and isinstance(referenced_column, list):
            for referencing_col, referenced_col in zip(referencing_column, referenced_column):
                referee_column_id = column_nodes[f"{referencing_table}.{referencing_col}"]['id']
                referent_column_id = column_nodes[f"{referenced_table}.{referenced_col}"]['id']
                column_referencedby_column_edges.append((referent_column_id, referee_column_id))
        else:
            referee_column_id = column_nodes[f"{referencing_table}.{referencing_column}"]['id']
            referent_column_id = column_nodes[f"{referenced_table}.{referenced_column}"]['id']
            column_referencedby_column_edges.append((referent_column_id, referee_column_id))

# Function to parse the query plan and extract the tables, columns, and predicates
def parse_query_plan(logger, plan, conn, statistics, db_stats, schema, encode_table_column):
    operator_nodes = []    # List of operators with features
    table_nodes = {}       # List of tables with features
    column_nodes = {}      # List of columns with features

    # all edge from bottom to top, while tree is parsed from top to bottom
    operator_calledby_operator_edges = []    
    table_scannedby_operator_edges = []    
    column_outputby_operator_edges = []    
    column_containedby_table_edges = []
    column_referencedby_column_edges = []
    column_selfloop_column_edges = []
    table_selfloop_table_edges = []


    operator_id_counter = [0]  # Using a list to make it mutable in recursion

    unique_data_types = sorted(db_stats['unique_data_types'])
    data_type_mapping = {data_type: i for i, data_type in enumerate(unique_data_types)}

    if encode_table_column:
        create_schema_graph(schema, db_stats, data_type_mapping, table_nodes, column_nodes, column_containedby_table_edges, column_referencedby_column_edges)


    traverse_operators(statistics, db_stats, data_type_mapping, plan, encode_table_column, operator_nodes, table_nodes, column_nodes, 
                       operator_calledby_operator_edges, table_scannedby_operator_edges, column_outputby_operator_edges, 
                       table_selfloop_table_edges, column_selfloop_column_edges, 
                       operator_id_counter)
    
    # for table_name, table_info in table_nodes.items():
    #     table_id = table_info['id']
    #     table_selfloop_table_edges.append((table_id, table_id))

    for column_name, column_info in column_nodes.items():
        column_id = column_info['id']
        column_selfloop_column_edges.append((column_id, column_id))

    return operator_nodes, table_nodes, column_nodes, \
        operator_calledby_operator_edges, table_scannedby_operator_edges, column_outputby_operator_edges, \
        column_containedby_table_edges, column_referencedby_column_edges, column_selfloop_column_edges  

# Function to create the heterogeneous graph from parsed components
def create_hetero_graph(logger, plan, conn, statistics, db_stats, schema, encode_table_column, peakmem, time):
    operator_nodes, table_nodes, column_nodes, \
        operator_calledby_operator_edges, table_scannedby_operator_edges, column_outputby_operator_edges, \
        column_containedby_table_edges, column_referencedby_column_edges, column_selfloop_column_edges  \
        = parse_query_plan(logger, plan, conn, statistics, db_stats, schema, encode_table_column)

    # Create the heterogeneous graph
    data = HeteroData()

    # Assign operator features
    operator_features = [op['features'] for op in operator_nodes]
    data['operator'].x = torch.tensor(operator_features, dtype=torch.float)

    if encode_table_column:
        # Assign table features
        if table_nodes:
            sorted_tables = sorted(table_nodes.items(), key=lambda x: x[1]['id'])
            table_features = [table[1]['features'] for table in sorted_tables]
            data['table'].x = torch.tensor(table_features, dtype=torch.float)
            # logger.debug(f"table_features: {data['table'].x.shape}")
        
        # Assign column features
        if column_nodes:
            sorted_columns = sorted(column_nodes.items(), key=lambda x: x[1]['id'])
            column_features = [column[1]['features'] for column in sorted_columns]
            data['column'].x = torch.tensor(column_features, dtype=torch.float)
            # logger.debug(f"column_features: {data['column'].x .shape}")
        
    if operator_calledby_operator_edges:
        src, dst = zip(*operator_calledby_operator_edges)
        data['operator', 'calledby', 'operator'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    if encode_table_column:
        if table_scannedby_operator_edges:
            src, dst = zip(*table_scannedby_operator_edges)
            data['table', 'scannedby', 'operator'].edge_index = torch.tensor([src, dst], dtype=torch.long)
        
        if column_outputby_operator_edges:
            src, dst = zip(*column_outputby_operator_edges)
            data['column', 'outputby', 'operator'].edge_index = torch.tensor([src, dst], dtype=torch.long)

        if column_referencedby_column_edges:
            src, dst = zip(*column_referencedby_column_edges)
            data['column','referencedby', 'column'].edge_index = torch.tensor([src, dst], dtype=torch.long)

        if column_containedby_table_edges:
            src, dst = zip(*column_containedby_table_edges)
            data['column', 'containedby', 'table'].edge_index = torch.tensor([src, dst], dtype=torch.long)

        # if table_selfloop_table_edges:
        #     src, dst = zip(*table_selfloop_table_edges)
        #     data['table','selfloop', 'table'].edge_index = torch.tensor([src, dst], dtype=torch.long)

        if column_selfloop_column_edges:
            src, dst = zip(*column_selfloop_column_edges)
            data['column', 'selfloop', 'column'].edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Assign the target
    data.y = torch.tensor(np.array([peakmem, time]), dtype=torch.float)
  
    return data


# Establish a connection to the PostgreSQL database
def connect_to_db(DB_CONFIG):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        return None
    

import logging
def get_logger():

    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    fmt = f"[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d]:%(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log

