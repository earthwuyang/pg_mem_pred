import torch
import re
import psycopg2
from torch_geometric.data import HeteroData
from moz_sql_parser import parse
from sklearn.preprocessing import RobustScaler
import numpy as np
import json
import os
from ..utils.database import get_relpages_reltuples, get_table_size, get_columns_info, get_column_features, get_unique_data_types, get_tables


    
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

def one_hot_encode_andornot_type(predicate_type):
    andornot_type_mapping = {
        'AND': 0,
        'OR': 1,
        'NOT': 2,
    }
    one_hot = [0] * 4  # 4 types: AND, OR, NOT, unknown
    if predicate_type in andornot_type_mapping:
        index = andornot_type_mapping[predicate_type]
        one_hot[index] = 1
    else:
        # Assign 'unknown' category
        one_hot[-1] = 1
    assert len(one_hot) == 4, "One-hot encoding should have 4 dimensions"
    return one_hot

def one_hot_encode_operation_type(operation_type):
    operation_type_mapping = {
        '=': 0,
        '<>': 1,
        '>': 2,
        '>=': 3,
        '<': 4,
        '<=': 5,
        'LIKE': 6,
        'UNKNOWN': 7,
    }
    one_hot = [0] * 8  # 8 types: =, <>, >, >=, <, <=, LIKE, UNKNOWN
    if operation_type in operation_type_mapping:
        index = operation_type_mapping[operation_type]
        one_hot[index] = 1
    else:
        # Assign 'unknown' category
        one_hot[-1] = 1
    assert len(one_hot) == 8, "One-hot encoding should have 8 dimensions"
    return one_hot


def extract_columns(string):
    # Regex to extract column names with at least one letter in the table/alias part
    # column_pattern = re.compile(r'\b([A-Za-z]+\.\w+)\b')
    column_pattern = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*\.\w+)\b')
    columns = column_pattern.findall(string)

    return columns

def encode_string_literal(string):
    features = []
    
    # 1. String Length
    str_length = len(string)
    features.append(str_length)
    
    # 2. String Type Indicators
    # Example: Determine if the string is used with LIKE
    if '%' in string or '_' in string:
        features.extend([1, 0, 0])  # LIKE pattern with wildcards
    else:
        features.extend([0, 1, 0])  # Exact match
        # Add more indicators as needed
    
    # 3. Character Composition
    alpha_count = sum(c.isalpha() for c in string)
    digit_count = sum(c.isdigit() for c in string)
    special_count = len(string) - alpha_count - digit_count
    features.extend([alpha_count, digit_count, special_count])
    
    # # 4. Uniqueness and Frequency
    # # Assuming you have access to frequency stats
    # frequency = db_stats.get('string_frequency', {}).get(string, 1)
    # unique_flag = 1 if frequency == 1 else 0
    # features.extend([unique_flag, frequency])
    
    # 5. Embedding Representations
    # Example using a simple character-level embedding (you can use more sophisticated methods)
    embedding_dim = 10
    embedding = np.zeros(embedding_dim)
    for i, char in enumerate(string[:embedding_dim]):
        embedding[i] = ord(char) / 255.0  # Normalize ASCII values
    features.extend(embedding.tolist())
    
    # 6. Hash-Based Features
    hash_value = hash(string) % (10**8)  # Example hash function
    features.append(hash_value)
    
    return features


def parse_predicate(predicate, predicate_nodes, operation_nodes, column_nodes, literal_nodes, numeral_nodes,
                    predicate_filters_operator_edges, operation_connects_predicate_edges, 
                    operation_filters_operator_edges, column_connects_operation_edges,
                    literal_connects_operation_edges, numeral_connects_operation_edges, 
                    predicate_connects_predicate_edges,
                    parent_id):

    def traverse(parsed_dict, predicate_nodes, operation_nodes, column_nodes, literal_nodes, numeral_nodes,
                    predicate_filters_operator_edges, operation_connects_predicate_edges, 
                    operation_filters_operator_edges, column_connects_operation_edges,
                    literal_connects_operation_edges, numeral_connects_operation_edges, predicate_connects_predicate_edges,
                    parent_id = None, parent_is_operator:bool=False):
        
        if 'and' in parsed_dict or 'or' in parsed_dict or 'not' in parsed_dict:
            if 'and' in parsed_dict:
                predicate = 'and'
            elif 'or' in parsed_dict:
                predicate = 'or'
            elif 'not' in parsed_dict:
                predicate = 'not'
            predicate = f"{predicate}_{len(predicate_nodes)}" # to have multiple instances of the same predicate in the graph
            if predicate not in predicate_nodes:  # CAUTION: PITFALL: If the same predicate appears multiple times, it should be added multiple times to the graph
                predicate_nodes[predicate] = {
                    'id': len(predicate_nodes),
                    'features': one_hot_encode_andornot_type(predicate)  # Placeholder, will be updated later
                }

            predicate_id = predicate_nodes[predicate]['id']
            if parent_is_operator:
                predicate_filters_operator_edges.append((predicate_id, parent_id))
            else:
                predicate_connects_predicate_edges.append((predicate_id, parent_id))
            for condition in parsed_dict[predicate.split('_')[0]]:
                traverse(condition, predicate_nodes, operation_nodes, column_nodes, literal_nodes, numeral_nodes,
                        predicate_filters_operator_edges, operation_connects_predicate_edges, 
                        operation_filters_operator_edges, column_connects_operation_edges, 
                        literal_connects_operation_edges, numeral_connects_operation_edges,  predicate_connects_predicate_edges, predicate_id)
        else:
            # It's a comparison operation
            # Identify the operation type
            if 'eq' in parsed_dict:
                op = '='
                left, right = parsed_dict['eq']
            elif 'neq' in parsed_dict:
                op = '<>'
                left, right = parsed_dict['neq']
            elif 'gt' in parsed_dict:
                op = '>'
                left, right = parsed_dict['gt']
            elif 'gte' in parsed_dict:
                op = '>='
                left, right = parsed_dict['gte']
            elif 'lt' in parsed_dict:
                op = '<'
                left, right = parsed_dict['lt']
            elif 'lte' in parsed_dict:
                op = '<='
                left, right = parsed_dict['lte']
            elif 'like' in parsed_dict:
                op = 'LIKE'
                left, right = parsed_dict['like']
            else:
                op = 'UNKNOWN'
            op = f"{op}_{len(operation_nodes)}"  # to have multiple instances of the same operation in the graph
            if op not in operation_nodes:
                operation_nodes[op] = {
                    'id': len(operation_nodes),
                    'features': one_hot_encode_operation_type(op)  # Placeholder, will be updated later
                }
            operation_id = operation_nodes[op]['id']
            if parent_is_operator:
                operation_filters_operator_edges.append((operation_id, parent_id))
            else:
                operation_connects_predicate_edges.append((operation_id, parent_id))
            
            if isinstance(left, str):
                if left not in column_nodes:
                    column_nodes[left] = {
                        'id': len(column_nodes),
                        'features': [0] * 10  # Placeholder, will be updated later
                    }
                column_id = column_nodes[left]['id']
                column_connects_operation_edges.append((column_id, operation_id))

            if isinstance(right, str):
                if right not in literal_nodes:
                    features = encode_string_literal(right)

                    literal_nodes[right] = {
                        'id': len(literal_nodes),
                        'features': features  # Placeholder, will be updated later
                    }
                literal_id = literal_nodes[right]['id']
                literal_connects_operation_edges.append((literal_id, operation_id))
            elif isinstance(right, float) or isinstance(right, int):
                if right not in numeral_nodes:
                    numeral_nodes[right] = {
                        'id': len(numeral_nodes),
                        'features': [right]   # Placeholder, will be updated later
                    }
                numeral_id = numeral_nodes[right]['id']
                numeral_connects_operation_edges.append((numeral_id, operation_id))
            else:    
                # print(f"parsed_dict     {parsed_dict}, right {right}")
                literals = right.get('cast', [dict()])[0].get('literal')
                # print(f"literals     {literals}")
                if literals not in literal_nodes:
                    features = encode_string_literal(literals)

                    literal_nodes[literals] = {
                        'id': len(literal_nodes),
                        'features': features  # Placeholder, will be updated later
                    }
                literal_id = literal_nodes[literals]['id']
                literal_connects_operation_edges.append((literal_id, operation_id))
                # raise ValueError("Invalid right operand type")

    
    predicate = predicate.replace('~~', 'like')
    sql_query = f"select * from dummy_table where {predicate}"

    parsed = parse(sql_query)
    where_clause = parsed.get('where', {})
    # print(f"where_clause     {where_clause}")
    traverse(where_clause, predicate_nodes, operation_nodes, column_nodes, literal_nodes, numeral_nodes,
                    predicate_filters_operator_edges, operation_connects_predicate_edges, 
                    operation_filters_operator_edges, column_connects_operation_edges,
                    literal_connects_operation_edges, numeral_connects_operation_edges, predicate_connects_predicate_edges, parent_id = parent_id, parent_is_operator=True)

# Helper function to traverse operators and extract tables, columns, and predicates
def traverse_operators(db_stats, plan, table_nodes, column_nodes, predicate_nodes, operation_nodes, 
                      operator_nodes, literal_nodes, numeral_nodes,
                      table_scannedby_operator_edges, predicate_filters_operator_edges, column_outputby_operator_edges,
                      column_connects_operation_edges, operator_calledby_operator_edges, operation_filters_operator_edges, 
                      operation_connects_predicate_edges,
                      literal_connects_operation_edges, numeral_connects_operation_edges, column_containedby_table_edges,
                      literal_selfloop_literal_edges, numeral_selfloop_numeral_edges, 
                      table_selfloop_table_edges, column_selfloop_column_edges, predicate_connects_predicate_edges,
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

        # add column to tables for column_containedby_table_edges
        for column in db_stats['tables'][table_name]['column_features']:
            if column not in column_nodes:
                column_nodes[column] = {
                    'id': len(column_nodes),
                    'features': [0] * expected_column_feature_len  # Placeholder, will be updated later
                }
            column_id = column_nodes[column]['id']
            column_containedby_table_edges.append((column_id, table_id))

    else:
        table_id = None

    # Extract columns and predicates from conditions
    involved_columns = set()
    for key in ['Hash Cond', 'Filter', 'Index Cond']:
        if key in plan_parameters:
            condition_str = plan_parameters[key]
            predicate = condition_str.strip()
            parse_predicate(predicate, predicate_nodes, operation_nodes, column_nodes, literal_nodes, numeral_nodes,
                            predicate_filters_operator_edges, operation_connects_predicate_edges, 
                            operation_filters_operator_edges, column_connects_operation_edges,
                            literal_connects_operation_edges, numeral_connects_operation_edges, 
                            predicate_connects_predicate_edges,
                            parent_id=current_operator_id)


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
            traverse_operators(db_stats, sub_plan, table_nodes, column_nodes, predicate_nodes, operation_nodes, 
                        operator_nodes,literal_nodes, numeral_nodes,
                      table_scannedby_operator_edges, predicate_filters_operator_edges, column_outputby_operator_edges,
                      column_connects_operation_edges, operator_calledby_operator_edges, operation_filters_operator_edges, operation_connects_predicate_edges,
                      literal_connects_operation_edges, numeral_connects_operation_edges, column_containedby_table_edges,
                      literal_selfloop_literal_edges, numeral_selfloop_numeral_edges, 
                      table_selfloop_table_edges, column_selfloop_column_edges, predicate_connects_predicate_edges,
                      operator_id_counter, current_operator_id)


# Function to parse the query plan and extract the tables, columns, and predicates
def parse_query_plan(logger, plan, conn, db_stats):

    table_nodes = {}       # table_name -> {'id': int, 'features': [...]}
    column_nodes = {}      # column_name -> {'id': int, 'features': [...]}
    predicate_nodes = {}   # predicate_str -> {'id': int, 'features': [...]}
    operator_nodes = []    # List of operators with features

    operation_nodes = {}   # operation_str -> {'id': int, 'features': [...]}
    literal_nodes = {}     # literal_str -> {'id': int, 'features': [...]}
    numeral_nodes = {}     # numeral_str -> {'id': int, 'features': [...]}
    
    # all edge from bottom to top, while tree is parsed from top to bottom
    table_scannedby_operator_edges = []    
    predicate_filters_operator_edges = []  
    column_outputby_operator_edges = []    
    column_connects_operation_edges = [] 
    operator_calledby_operator_edges = []    
    

    operation_filters_operator_edges = []
    operation_connects_predicate_edges = []
    literal_connects_operation_edges = []
    numeral_connects_operation_edges = []

    predicate_connects_predicate_edges = []
    column_containedby_table_edges = []

    table_selfloop_table_edges = []
    column_selfloop_column_edges = []
    
    literal_selfloop_literal_edges = []
    numeral_selfloop_numeral_edges = []

    operator_id_counter = [0]  # Using a list to make it mutable in recursion


    traverse_operators(db_stats, plan, table_nodes, column_nodes, predicate_nodes, operation_nodes, operator_nodes,literal_nodes, numeral_nodes,
                      table_scannedby_operator_edges, predicate_filters_operator_edges, column_outputby_operator_edges,
                      column_connects_operation_edges, operator_calledby_operator_edges, operation_filters_operator_edges, operation_connects_predicate_edges,
                      literal_connects_operation_edges, numeral_connects_operation_edges, column_containedby_table_edges,
                      literal_selfloop_literal_edges, numeral_selfloop_numeral_edges, 
                      table_selfloop_table_edges, column_selfloop_column_edges, predicate_connects_predicate_edges,
                      operator_id_counter)
    
    # add self-loop edges for tables and columns
    for table_name, table_info in table_nodes.items():
        table_id = table_info['id']
        table_selfloop_table_edges.append((table_id, table_id))
    for column_name, column_info in column_nodes.items():
        column_id = column_info['id']
        column_selfloop_column_edges.append((column_id, column_id))
    for literal_str, literal_info in literal_nodes.items():
        literal_id = literal_info['id']
        literal_selfloop_literal_edges.append((literal_id, literal_id))
    for numeral_str, numeral_info in numeral_nodes.items():
        numeral_id = numeral_info['id']
        numeral_selfloop_numeral_edges.append((numeral_id, numeral_id))


    # print(f"db_stats.keys() {db_stats.keys()}")
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

    if literal_nodes:
        literal_scaler = RobustScaler()
        scaled_literal_features = literal_scaler.fit_transform(np.array([literal_nodes[literal]['features'] for literal in literal_nodes]))
        for i, literal in enumerate(literal_nodes):
            literal_nodes[literal]['features'] = scaled_literal_features[i].tolist()

    if numeral_nodes:
        numeral_scaler = RobustScaler()
        scaled_numeral_features = numeral_scaler.fit_transform(np.array([numeral_nodes[numeral]['features'] for numeral in numeral_nodes]).reshape(-1, 1)).reshape(-1)
        for i, numeral in enumerate(numeral_nodes):
            numeral_nodes[numeral]['features'] = scaled_numeral_features[i].tolist()

    if table_nodes:
        # normalize table features and column features
        table_scaler = RobustScaler()
        scaled_table_features = table_scaler.fit_transform(np.array([table_nodes[table]['features'] for table in table_nodes]))
        for i, table in enumerate(table_nodes):
            table_nodes[table]['features'] = scaled_table_features[i].tolist()

    if column_nodes:
        column_scaler = RobustScaler()
        scaled_column_features = column_scaler.fit_transform(np.array([column_nodes[column]['features'] for column in column_nodes]))
        for i, column in enumerate(column_nodes):
            column_nodes[column]['features'] = scaled_column_features[i].tolist()



    # # Update predicate features: [predicate_length]
    # for pred, pred_info in predicate_nodes.items():
    #     predicate_length = len(pred)
    #     predicate_nodes[pred]['features'] = [predicate_length]
 
    return table_nodes, column_nodes, predicate_nodes, operation_nodes, operator_nodes, literal_nodes, numeral_nodes, \
                      table_scannedby_operator_edges, predicate_filters_operator_edges, column_outputby_operator_edges, \
                      column_connects_operation_edges, operator_calledby_operator_edges, operation_filters_operator_edges, operation_connects_predicate_edges,  \
                      literal_connects_operation_edges, numeral_connects_operation_edges, column_containedby_table_edges, \
                      literal_selfloop_literal_edges, numeral_selfloop_numeral_edges,  \
                      table_selfloop_table_edges, column_selfloop_column_edges, predicate_connects_predicate_edges

# Function to create the heterogeneous graph from parsed components
def create_hetero_graph(logger, 
                table_nodes, column_nodes, predicate_nodes, operation_nodes, operator_nodes,literal_nodes, numeral_nodes, 
                      table_scannedby_operator_edges, predicate_filters_operator_edges, column_outputby_operator_edges, 
                      column_connects_operation_edges, operator_calledby_operator_edges, operation_filters_operator_edges, operation_connects_predicate_edges,  
                      literal_connects_operation_edges, numeral_connects_operation_edges, column_containedby_table_edges,
                      literal_selfloop_literal_edges, numeral_selfloop_numeral_edges,  
                      table_selfloop_table_edges, column_selfloop_column_edges, predicate_connects_predicate_edges, peakmem, mem_scaler):
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
    
    # Assign operation features
    sorted_operations = sorted(operation_nodes.items(), key=lambda x: x[1]['id'])
    operation_features = [operation[1]['features'] for operation in sorted_operations]
    data['operation'].x = torch.tensor(operation_features, dtype=torch.float)
    
    # Assign literal features
    sorted_literals = sorted(literal_nodes.items(), key=lambda x: x[1]['id'])
    literal_features = [literal[1]['features'] for literal in sorted_literals]
    data['literal'].x = torch.tensor(literal_features, dtype=torch.float)
    
    # Assign numeral features
    sorted_numerals = sorted(numeral_nodes.items(), key=lambda x: x[1]['id'])
    numeral_features = [numeral[1]['features'] for numeral in sorted_numerals]
    data['numeral'].x = torch.tensor(numeral_features, dtype=torch.float)


    
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

    # literal_selfloop_literal_edges
    if literal_selfloop_literal_edges:
        src, dst = zip(*literal_selfloop_literal_edges)
        data['literal', 'selfloop', 'literal'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # numeral_selfloop_numeral_edges
    if numeral_selfloop_numeral_edges:
        src, dst = zip(*numeral_selfloop_numeral_edges)
        data['numeral', 'selfloop', 'numeral'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # column_connects_operation_edges
    if column_connects_operation_edges:
        src, dst = zip(*column_connects_operation_edges)
        data['column', 'connects', 'operation'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # operation_filters_operator_edges
    if operation_filters_operator_edges:
        src, dst = zip(*operation_filters_operator_edges)
        data['operation', 'filters', 'operator'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # operation_connects_predicate_edges
    if operation_connects_predicate_edges:
        src, dst = zip(*operation_connects_predicate_edges)
        data['operation', 'connects', 'predicate'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # literal_connects_operation_edges
    if literal_connects_operation_edges:
        src, dst = zip(*literal_connects_operation_edges)
        data['literal', 'connects', 'operation'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # numeral_connects_operation_edges
    if numeral_connects_operation_edges:
        src, dst = zip(*numeral_connects_operation_edges)
        data['numeral', 'connects', 'operation'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # predicate_connects_predicate_edges
    if predicate_connects_predicate_edges:
        src, dst = zip(*predicate_connects_predicate_edges)
        data['predicate', 'connects', 'predicate'].edge_index = torch.tensor([src, dst], dtype=torch.long)

    if column_containedby_table_edges:
        src, dst = zip(*column_containedby_table_edges)
        data['column', 'containedby', 'table'].edge_index = torch.tensor([src, dst], dtype=torch.long)

    peakmem = mem_scaler.transform(np.array([peakmem]).reshape(-1, 1)).reshape(-1)
    
    # Assign the target
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


if __name__ == '__main__':
    # Example usage:
    DB_CONFIG = {
        'host': 'localhost',
        'database': 'tpc_h',
        'user': 'wuy',
        'password': ''
    }
    conn = connect_to_db(DB_CONFIG)
    db_stats = get_db_stats('tpch_sf1')
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
    
    plan ={
            "Plan": {
                "Node Type": "Gather Merge",
                "Parallel Aware": False,
                "Async Capable": False,
                "Startup Cost": 40072.66,
                "Total Cost": 40079.66,
                "Plan Rows": 60,
                "Plan Width": 12,
                "Output": ["customer.c_custkey", "orders.o_orderkey", "orders.o_orderdate"],
                "Workers Planned": 2,
                "Plans": [
                {
                    "Node Type": "Sort",
                    "Parent Relationship": "Outer",
                    "Parallel Aware": False,
                    "Async Capable": False,
                    "Startup Cost": 39072.64,
                    "Total Cost": 39072.71,
                    "Plan Rows": 30,
                    "Plan Width": 12,
                    "Output": ["customer.c_custkey", "orders.o_orderkey", "orders.o_orderdate"],
                    "Sort Key": ["orders.o_orderdate DESC"],
                    "Plans": [
                    {
                        "Node Type": "Hash Join",
                        "Parent Relationship": "Outer",
                        "Parallel Aware": True,
                        "Async Capable": False,
                        "Join Type": "Inner",
                        "Startup Cost": 4366.32,
                        "Total Cost": 39071.90,
                        "Plan Rows": 30,
                        "Plan Width": 12,
                        "Output": ["customer.c_custkey", "orders.o_orderkey", "orders.o_orderdate"],
                        "Inner Unique": True,
                        "Hash Cond": "(orders.o_custkey = customer.c_custkey)",
                        "Plans": [
                        {
                            "Node Type": "Seq Scan",
                            "Parent Relationship": "Outer",
                            "Parallel Aware": True,
                            "Async Capable": False,
                            "Relation Name": "orders",
                            "Schema": "public",
                            "Alias": "orders",
                            "Startup Cost": 0.00,
                            "Total Cost": 33907.50,
                            "Plan Rows": 304021,
                            "Plan Width": 12,
                            "Output": [
                            "orders.o_orderkey",
                            "orders.o_custkey",
                            "orders.o_orderstatus",
                            "orders.o_totalprice",
                            "orders.o_orderdate",
                            "orders.o_orderpriority",
                            "orders.o_clerk",
                            "orders.o_shippriority",
                            "orders.o_comment"
                            ],
                            "Filter": "(orders.o_orderstatus = 'F'::bpchar)"
                        },
                        {
                            "Node Type": "Hash",
                            "Parent Relationship": "Inner",
                            "Parallel Aware": True,
                            "Async Capable": False,
                            "Startup Cost": 4366.25,
                            "Total Cost": 4366.25,
                            "Plan Rows": 6,
                            "Plan Width": 4,
                            "Output": ["customer.c_custkey"],
                            "Plans": [
                            {
                                "Node Type": "Seq Scan",
                                "Parent Relationship": "Outer",
                                "Parallel Aware": True,
                                "Async Capable": False,
                                "Relation Name": "customer",
                                "Schema": "public",
                                "Alias": "customer",
                                "Startup Cost": 0.00,
                                "Total Cost": 4366.25,
                                "Plan Rows": 6,
                                "Plan Width": 4,
                                "Output": ["customer.c_custkey"],
                                "Filter": "((customer.c_name)::text like 'Alice%'::text)"
                            }
                            ]
                        }
                        ]
                    }
                    ]
                }
                ]
            }
            }



    logger = get_logger()
    table_nodes, column_nodes, predicate_nodes, operation_nodes, operator_nodes,literal_nodes, numeral_nodes, \
        table_scannedby_operator_edges, predicate_filters_operator_edges, column_outputby_operator_edges, \
        column_connects_operation_edges, operator_calledby_operator_edges, operation_filters_operator_edges, operation_connects_predicate_edges,  \
        literal_connects_operation_edges, numeral_connects_operation_edges, column_containedby_table_edges, \
        literal_selfloop_literal_edges, numeral_selfloop_numeral_edges,  \
        table_selfloop_table_edges, column_selfloop_column_edges, predicate_connects_predicate_edges = parse_query_plan(logger, plan, conn, db_stats)


    print(f"table_nodes: {table_nodes}")
    print(f"column_nodes: {column_nodes}")
    print(f"predicate_nodes: {predicate_nodes}")
    print(f"operation_nodes: {operation_nodes}")
    print(f"operator_nodes: {operator_nodes}")
    print(f"literal_nodes: {literal_nodes}")
    print(f"numeral_nodes: {numeral_nodes}")

    print(f"table_scannedby_operator_edges: {table_scannedby_operator_edges}")
    print(f"predicate_filters_operator_edges: {predicate_filters_operator_edges}")
    print(f"column_outputby_operator_edges: {column_outputby_operator_edges}")
    print(f"column_connects_operation_edges: {column_connects_operation_edges}")
    print(f"operator_calledby_operator_edges: {operator_calledby_operator_edges}")
    print(f"operation_filters_operator_edges: {operation_filters_operator_edges}")
    print(f"operation_connects_predicate_edges: {operation_connects_predicate_edges}")
    print(f"literal_connects_operation_edges: {literal_connects_operation_edges}")
    print(f"numeral_connects_operation_edges: {numeral_connects_operation_edges}")
    print(f"column_containedby_table_edges: {column_containedby_table_edges}")
    print(f"literal_selfloop_literal_edges: {literal_selfloop_literal_edges}")
    print(f"numeral_selfloop_numeral_edges: {numeral_selfloop_numeral_edges}")
    print(f"table_selfloop_table_edges: {table_selfloop_table_edges}")
    print(f"column_selfloop_column_edges: {column_selfloop_column_edges}")
    print(f"predicate_connects_predicate_edges: {predicate_connects_predicate_edges}")