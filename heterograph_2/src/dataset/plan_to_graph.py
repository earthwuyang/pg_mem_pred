import torch
import re
import psycopg2
from torch_geometric.data import HeteroData
from moz_sql_parser import parse
import numpy as np
from ..utils.database import get_relpages_reltuples, get_table_size, get_columns_info, get_column_features, get_unique_data_types


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
    assert len(one_hot) == 6, "One-hot encoding should have 6 dimensions"
    return one_hot

def extract_columns(string):
    # Regex to extract column names with at least one letter in the table/alias part
    # column_pattern = re.compile(r'\b([A-Za-z]+\.\w+)\b')
    column_pattern = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*\.\w+)\b')
    columns = column_pattern.findall(string)

    return columns

def extract_predicate_features(predicate):
    predicate_length = len(predicate)
    # Count unique columns and operators in the predicate
    columns = extract_columns(predicate)  # Assuming extract_columns can be reused
    unique_columns = len(set(columns))  # Unique columns count

    # Count operators (this is a simple example)
    operator_count = sum(predicate.count(op) for op in ['=', '<>', '>', '<', '>=', '<='])

    return [predicate_length, unique_columns, operator_count, 0]

# Helper function to traverse operators and extract tables, columns, and predicates
def traverse_operators(plan, table_nodes, column_nodes, predicate_nodes, operator_nodes, literal_nodes, numeral_nodes,
                      table_scannedby_operator_edges, predicate_filters_operator_edges, column_outputby_operator_edges,
                      column_connects_predicate_edges, operator_calledby_operator_edges, 
                      literal_connects_predicate_edges, numeral_connects_predicate_edges,
                      literal_selfloop_literal_edges, numeral_selfloop_numeral_edges,
                      operator_id_counter, parent_operator_id=None):
    current_operator_id = operator_id_counter[0]
    operator_id_counter[0] += 1  # Increment the operator ID counter

    expected_column_feature_len = 10  # avg_width, correlation, n_distinct, null_frac, data_type_one_hot_0, data_type_one_hot_1, data_type_one_hot_2, data_type_one_hot_3, data_type_one_hot_4, data_type_one_hot_5
    if 'Plan' in plan:
        plan_parameters = plan.get('Plan', {})
    else:
        plan_parameters = plan

    # Extract operator features
    operator_features = [
        plan_parameters.get('Startup Cost', 0.0),
        plan_parameters.get('Total Cost', 0.0),
        plan_parameters.get('Plan Rows', 0),
        plan_parameters.get('Plan Width', 0)
    ]
    operator_nodes.append({
        'id': current_operator_id,
        'type': plan_parameters.get('Node Type', 'Unknown'),
        'features': operator_features
    })
    # If there is a parent operator, add an edge (operator calls operator)
    if parent_operator_id is not None:
        operator_calledby_operator_edges.append((current_operator_id, parent_operator_id))

    # Extract tables, columns, predicates, and edges
    if 'Relation Name' in plan_parameters:
        table_name = plan_parameters['Relation Name']
        if table_name not in table_nodes:
            table_nodes[table_name] = {
                'id': len(table_nodes),
                'features': [0, 0]  # Placeholder, will be updated later
            }
        table_id = table_nodes[table_name]['id']
        # Add edge: operator involves table
        table_scannedby_operator_edges.append((table_id, current_operator_id))
    else:
        table_id = None

    # Extract columns and predicates from conditions
    involved_columns = set()
    for key in ['Hash Cond', 'Filter', 'Index Cond']:
        if key in plan_parameters:
            condition_str = plan_parameters[key]
            predicate = condition_str.strip()

            cols = extract_columns(condition_str)
            involved_columns.update(cols)

            if predicate not in predicate_nodes:
                predicate_nodes[predicate] = {
                    'id': len(predicate_nodes),
                    'features': [0, 0, 0, 0]  # Placeholder, will be updated later
                }

            predicate_id = predicate_nodes[predicate]['id']

            # Add edge: predicate filters operator
            predicate_filters_operator_edges.append((predicate_id, current_operator_id))

            for col in involved_columns:
                if col not in column_nodes:
                    column_nodes[col] = {
                        'id': len(column_nodes),
                        'features': [0] * expected_column_feature_len  # Placeholder, will be updated later
                    }
                column_id = column_nodes[col]['id']
                # Add edge: predicate connects column
                column_connects_predicate_edges.append((column_id, predicate_id))

    output_list = plan_parameters.get('Output', [])
    for output_item in output_list:
        cols = extract_columns(output_item)
        for col in cols:
            if col not in column_nodes:
                column_nodes[col] = {
                    'id': len(column_nodes),
                    'features': [0] * expected_column_feature_len  # Placeholder, will be updated later
                }
            column_id = column_nodes[col]['id']
            # Add edge: column is output by operator
            column_outputby_operator_edges.append((column_id, current_operator_id))

    # Recurse into sub-plans
    if 'Plans' in plan_parameters:
        for sub_plan in plan_parameters['Plans']:
            traverse_operators(sub_plan, table_nodes, column_nodes, predicate_nodes, operator_nodes,literal_nodes, numeral_nodes,
                      table_scannedby_operator_edges, predicate_filters_operator_edges, column_outputby_operator_edges,
                      column_connects_predicate_edges, operator_calledby_operator_edges, 
                      literal_connects_predicate_edges, numeral_connects_predicate_edges,
                      literal_selfloop_literal_edges, numeral_selfloop_numeral_edges,
                      operator_id_counter, current_operator_id)


# Function to parse the query plan and extract the tables, columns, and predicates
def parse_query_plan(logger, plan, conn, db_stats):

    table_nodes = {}       # table_name -> {'id': int, 'features': [...]}
    column_nodes = {}      # column_name -> {'id': int, 'features': [...]}
    predicate_nodes = {}   # predicate_str -> {'id': int, 'features': [...]}
    operator_nodes = []    # List of operators with features

    literal_nodes = {}     # literal_str -> {'id': int, 'features': [...]}
    numeral_nodes = {}     # numeral_str -> {'id': int, 'features': [...]}
    
    # all edge from bottom to top, while tree is parsed from top to bottom
    table_scannedby_operator_edges = []    
    predicate_filters_operator_edges = []  
    column_outputby_operator_edges = []    
    column_connects_predicate_edges = [] 
    operator_calledby_operator_edges = []    
    table_selfloop_table_edges = []
    column_selfloop_column_edges = []

    literal_connects_predicate_edges = []
    numeral_connects_predicate_edges = []
    literal_selfloop_literal_edges = []
    numeral_selfloop_numeral_edges = []

    

    operator_id_counter = [0]  # Using a list to make it mutable in recursion


    traverse_operators(plan, table_nodes, column_nodes, predicate_nodes, operator_nodes,literal_nodes, numeral_nodes,
                      table_scannedby_operator_edges, predicate_filters_operator_edges, column_outputby_operator_edges,
                      column_connects_predicate_edges, operator_calledby_operator_edges,
                      literal_connects_predicate_edges, numeral_connects_predicate_edges,
                      literal_selfloop_literal_edges, numeral_selfloop_numeral_edges,
                      operator_id_counter)
    
    # add self-loop edges for tables and columns
    for table_name, table_info in table_nodes.items():
        table_id = table_info['id']
        table_selfloop_table_edges.append((table_id, table_id))
    for column_name, column_info in column_nodes.items():
        column_id = column_info['id']
        column_selfloop_column_edges.append((column_id, column_id))


    unique_data_types = sorted(db_stats['unique_data_types'])
    data_type_mapping = {data_type: i for i, data_type in enumerate(unique_data_types)}

    # Now, fetch actual features for tables and columns
    for table_name, table_info in table_nodes.items():
        relpages, reltuples = db_stats['tables'][table_name]['relpages'], db_stats['tables'][table_name]['reltuples']
        table_nodes[table_name]['features'] = [relpages, reltuples]
        
    for column_name in column_nodes:
        table_name = column_name.split('.')[0]
        avg_width = db_stats['tables'][table_name]['column_features'][column_name]['avg_width']
        correlation = db_stats['tables'][table_name]['column_features'][column_name]['correlation']
        n_distinct = db_stats['tables'][table_name]['column_features'][column_name]['n_distinct']
        null_frac = db_stats['tables'][table_name]['column_features'][column_name]['null_frac']
        data_type = db_stats['tables'][table_name]['column_features'][column_name]['data_type']
        one_hot = one_hot_encode_data_type(data_type, data_type_mapping)  # Unique data types: {'character': 0, 'character varying': 1, 'date': 2, 'integer': 3, 'numeric': 4}
        column_nodes[column_name]['features'] = [avg_width, correlation, n_distinct, null_frac] + one_hot


    # Update predicate features: [predicate_length]
    for pred, pred_info in predicate_nodes.items():
        features = extract_predicate_features(pred)
        predicate_nodes[pred]['features'] = features

    return table_nodes, column_nodes, predicate_nodes, operator_nodes, \
                      table_scannedby_operator_edges, predicate_filters_operator_edges, column_outputby_operator_edges, \
                      column_connects_predicate_edges, operator_calledby_operator_edges, \
                      table_selfloop_table_edges, column_selfloop_column_edges

# Function to create the heterogeneous graph from parsed components
def create_hetero_graph(logger, table_nodes, column_nodes, predicate_nodes, operator_nodes,
                      table_scannedby_operator_edges, predicate_filters_operator_edges, column_outputby_operator_edges,
                      column_connects_predicate_edges, operator_calledby_operator_edges, 
                      table_selfloop_table_edges, column_selfloop_column_edges, peakmem, mem_scaler):
    data = HeteroData()

    # Assign operator features
    operator_features = [op['features'] for op in operator_nodes]
    data['operator'].x = torch.tensor(operator_features, dtype=torch.float)
    
    # Assign table features
    sorted_tables = sorted(table_nodes.items(), key=lambda x: x[1]['id'])
    table_features = [table[1]['features'] for table in sorted_tables]
    data['table'].x = torch.tensor(table_features, dtype=torch.float)
    
    

    # Assign column features
    sorted_columns = sorted(column_nodes.items(), key=lambda x: x[1]['id'])
    column_features = [column[1]['features'] for column in sorted_columns]
    
    data['column'].x = torch.tensor(column_features, dtype=torch.float)

    
    # Assign predicate features
    sorted_predicates = sorted(predicate_nodes.items(), key=lambda x: x[1]['id'])
    predicate_features = [predicate[1]['features'] for predicate in sorted_predicates]
    data['predicate'].x = torch.tensor(predicate_features, dtype=torch.float)
    
    
    # Create edge index dictionaries
    # table_scannedby_operator_edges
    if table_scannedby_operator_edges:
        src, dst = zip(*table_scannedby_operator_edges)
        data['table', 'scannedby', 'operator'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # predicate_filters_operator_edges
    if predicate_filters_operator_edges:
        src, dst = zip(*predicate_filters_operator_edges)
        data['predicate', 'filters', 'operator'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # column_outputby_operator_edges
    if column_outputby_operator_edges:
        src, dst = zip(*column_outputby_operator_edges)
        data['column', 'outputby', 'operator'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # column_connects_predicate_edges
    if column_connects_predicate_edges:
        src, dst = zip(*column_connects_predicate_edges)
        data['column', 'connects', 'predicate'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # operator_calledby_operator_edges
    if operator_calledby_operator_edges:
        src, dst = zip(*operator_calledby_operator_edges)
        data['operator', 'calledby', 'operator'].edge_index = torch.tensor([src, dst], dtype=torch.long)

    # table_selfloop_table_edges
    if table_selfloop_table_edges:
        src, dst = zip(*table_selfloop_table_edges)
        data['table', 'selfloop', 'table'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # column_selfloop_column_edges
    if column_selfloop_column_edges:
        src, dst = zip(*column_selfloop_column_edges)
        data['column', 'selfloop', 'column'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # Assign the target
    peakmem = mem_scaler.transform(np.array([peakmem]).reshape(-1, 1)).reshape(-1)
    data.y = torch.tensor(peakmem, dtype=torch.float)
  
    return data


# Establish a connection to the PostgreSQL database
def connect_to_db(DB_CONFIG):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        return None
    

if __name__ == '__main__':
    # Example usage:
    DB_CONFIG = {
        'host': 'localhost',
        'database': 'tpc_h',
        'user': 'wuy',
        'password': ''
    }
    conn = connect_to_db(DB_CONFIG)
    data_type_mapping = {data_type: idx for idx, data_type in enumerate(get_unique_data_types(conn))}
    plan = {
            "Plan": {
                "Node Type": "Aggregate",
                "Strategy": "Plain",
                "Partial Mode": "Simple",
                "Parallel Aware": False,
                "Async Capable": False,
                "Startup Cost": 2.74,
                "Total Cost": 2.75,
                "Plan Rows": 1,
                "Plan Width": 32,
                "Output": [
                "avg((nation.n_regionkey + nation.n_nationkey))"
                ],
                "Plans": [
                {
                    "Node Type": "Hash Join",
                    "Parent Relationship": "Outer",
                    "Parallel Aware": False,
                    "Async Capable": False,
                    "Join Type": "Inner",
                    "Startup Cost": 1.11,
                    "Total Cost": 2.66,
                    "Plan Rows": 16,
                    "Plan Width": 8,
                    "Output": [
                    "nation.n_regionkey",
                    "nation.n_nationkey"
                    ],
                    "Inner Unique": False,
                    "Hash Cond": "(nation.n_regionkey = region.r_regionkey)",
                    "Plans": [
                    {
                        "Node Type": "Seq Scan",
                        "Parent Relationship": "Outer",
                        "Parallel Aware": False,
                        "Async Capable": False,
                        "Relation Name": "nation",
                        "Schema": "public",
                        "Alias": "nation",
                        "Startup Cost": 0.0,
                        "Total Cost": 1.31,
                        "Plan Rows": 20,
                        "Plan Width": 8,
                        "Output": [
                        "nation.n_nationkey",
                        "nation.n_name",
                        "nation.n_regionkey",
                        "nation.n_comment"
                        ],
                        "Filter": "(nation.n_regionkey <> 3)"
                    },
                    {
                        "Node Type": "Hash",
                        "Parent Relationship": "Inner",
                        "Parallel Aware": False,
                        "Async Capable": False,
                        "Startup Cost": 1.06,
                        "Total Cost": 1.06,
                        "Plan Rows": 4,
                        "Plan Width": 4,
                        "Output": [
                        "region.r_regionkey"
                        ],
                        "Plans": [
                        {
                            "Node Type": "Seq Scan",
                            "Parent Relationship": "Outer",
                            "Parallel Aware": False,
                            "Async Capable": False,
                            "Relation Name": "region",
                            "Schema": "public",
                            "Alias": "region",
                            "Startup Cost": 0.0,
                            "Total Cost": 1.06,
                            "Plan Rows": 4,
                            "Plan Width": 4,
                            "Output": [
                            "region.r_regionkey"
                            ],
                            "Filter": "(region.r_regionkey <> 3)"
                        }
                        ]
                    }
                    ]
                }
                ]
            },
            "peakmem": 17220
            }
    table_nodes, column_nodes, predicate_nodes, operator_nodes, \
    operator_involves_table_edges, operator_involves_column_edges, \
    table_contains_column_edges, column_connected_predicate_edges, \
    _ = parse_query_plan(logger, plan, conn, db_stats)
