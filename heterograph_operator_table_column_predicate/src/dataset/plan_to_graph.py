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
            predicate = f"{predicate}_{len(predicate_nodes)}"
            if predicate not in predicate_nodes:  # CAUTION: PITFALL: If the same predicate appears multiple times, it should be added multiple times to the graph
                predicate_nodes[predicate] = {
                    'id': len(predicate_nodes),
                    'features': one_hot_encode_andornot_type(predicate)  
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

            op = f"{op}_{len(operation_nodes)}"
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
                assert left in column_nodes, "Left operand should be a column name"
                # if left not in column_nodes:
                #     column_nodes[left] = {
                #         'id': len(column_nodes),
                #         'features': [0] * 10  # Placeholder, will be updated later
                #     }
                column_id = column_nodes[left]['id']
                column_connects_operation_edges.append((column_id, operation_id))

            if isinstance(right, str):
                column_id = column_nodes[right]['id']
                column_connects_operation_edges.append((column_id, operation_id))
            elif isinstance(right, int) or isinstance(right, float):
                if right not in numeral_nodes:
                    numeral_nodes[right] = {
                        'id': len(numeral_nodes),
                        'features': [right]   # Placeholder, will be updated later
                    }
                numeral_id = numeral_nodes[right]['id']
                numeral_connects_operation_edges.append((numeral_id, operation_id))
            else:
                assert isinstance(right, dict), 'Right operand should be a string literal or a numeral, or a dict such as function cast "{"cast": [{"literal": "string_literal"}]}'
                # print(f"parsed_dict     {parsed_dict}, right {right}")
                literals = right.get('cast', [dict()])[0].get('literal')
                # print(f"literals     {literals}")
                if literals not in literal_nodes:
                    features = encode_string_literal(literals)

                    literal_nodes[literals] = {
                        'id': len(literal_nodes),
                        'features': features 
                    }
                literal_id = literal_nodes[literals]['id']
                literal_connects_operation_edges.append((literal_id, operation_id))
                # raise ValueError("Invalid right operand type")


    sql_query = f"select * from dummy_table where {predicate}"
    parsed = parse(sql_query)
    where_clause = parsed.get('where', {})
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
                      column_referencedby_column_edges,
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
        table_id = table_nodes[table_name]['id']
        # Add edge: operator involves table
        table_scannedby_operator_edges.append((table_id, current_operator_id))
    else:
        table_id = None

    output_list = plan_parameters.get('Output', [])
    for output_item in output_list:
        cols = extract_columns(output_item)
        for col in cols:
            column_id = column_nodes[col]['id']
            # Add edge: column is output by operator
            column_outputby_operator_edges.append((column_id, current_operator_id))
    
    # Extract columns and predicates from conditions
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

    # Recurse into sub-plans
    if 'Plans' in plan_parameters:
        for sub_plan in plan_parameters['Plans']:
            traverse_operators(db_stats, sub_plan, table_nodes, column_nodes, predicate_nodes, operation_nodes, 
                        operator_nodes,literal_nodes, numeral_nodes,
                      table_scannedby_operator_edges, predicate_filters_operator_edges, column_outputby_operator_edges,
                      column_connects_operation_edges, operator_calledby_operator_edges, operation_filters_operator_edges, operation_connects_predicate_edges,
                      literal_connects_operation_edges, numeral_connects_operation_edges, column_containedby_table_edges,
                      column_referencedby_column_edges,
                      literal_selfloop_literal_edges, numeral_selfloop_numeral_edges, 
                      table_selfloop_table_edges, column_selfloop_column_edges, predicate_connects_predicate_edges,
                      operator_id_counter, current_operator_id)


def create_schema_graph(schema, db_stats, data_type_mapping, table_nodes, column_nodes, column_containedby_table_edges, column_referencedby_column_edges):
  
    for table_name in schema['columns']:
        relpages, reltuples = db_stats['tables'][table_name]['relpages'], db_stats['tables'][table_name]['reltuples']
        table_nodes[table_name] = {
            'id': len(table_nodes),
            'features': [relpages, reltuples]  # Placeholder, will be updated later
        }
        for column_name in schema['columns'][table_name]:
            column_name = f"{table_name}.{column_name}"
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
def parse_query_plan(logger, plan, conn, db_stats, schema):

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
    column_referencedby_column_edges = []

    table_selfloop_table_edges = []
    column_selfloop_column_edges = []
    
    literal_selfloop_literal_edges = []
    numeral_selfloop_numeral_edges = []

    operator_id_counter = [0]  # Using a list to make it mutable in recursion

    # print(f"db_stats.keys() {db_stats.keys()}")
    unique_data_types = sorted(db_stats['unique_data_types'])
    data_type_mapping = {data_type: i for i, data_type in enumerate(unique_data_types)}

    create_schema_graph(schema, db_stats, data_type_mapping, table_nodes, column_nodes, column_containedby_table_edges, column_referencedby_column_edges)

    traverse_operators(db_stats, plan, table_nodes, column_nodes, predicate_nodes, operation_nodes, operator_nodes,literal_nodes, numeral_nodes,
                      table_scannedby_operator_edges, predicate_filters_operator_edges, column_outputby_operator_edges,
                      column_connects_operation_edges, operator_calledby_operator_edges, operation_filters_operator_edges, operation_connects_predicate_edges,
                      literal_connects_operation_edges, numeral_connects_operation_edges, column_containedby_table_edges, column_referencedby_column_edges,
                      literal_selfloop_literal_edges, numeral_selfloop_numeral_edges, 
                      table_selfloop_table_edges, column_selfloop_column_edges, predicate_connects_predicate_edges,
                      operator_id_counter)
    
    # add self-loop edges for tables and columns
    # for table_name, table_info in table_nodes.items():
    #     table_id = table_info['id']
    #     table_selfloop_table_edges.append((table_id, table_id))
    for column_name, column_info in column_nodes.items():
        column_id = column_info['id']
        column_selfloop_column_edges.append((column_id, column_id))
    for literal_str, literal_info in literal_nodes.items():
        literal_id = literal_info['id']
        literal_selfloop_literal_edges.append((literal_id, literal_id))
    for numeral_str, numeral_info in numeral_nodes.items():
        numeral_id = numeral_info['id']
        numeral_selfloop_numeral_edges.append((numeral_id, numeral_id))

 
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
                      table_selfloop_table_edges, column_selfloop_column_edges, predicate_connects_predicate_edges, peakmem, time):
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
    if sorted_predicates:
        predicate_features = [predicate[1]['features'] for predicate in sorted_predicates]
        data['predicate'].x = torch.tensor(predicate_features, dtype=torch.float)
    else:
        data['predicate'].x = torch.empty((0, 4))
    
    # Assign operation features
    sorted_operations = sorted(operation_nodes.items(), key=lambda x: x[1]['id'])
    if sorted_operations:
        operation_features = [operation[1]['features'] for operation in sorted_operations]
        data['operation'].x = torch.tensor(operation_features, dtype=torch.float)
    else:
        data['operation'].x = torch.empty((0, 8))
    
    # Assign literal features
    sorted_literals = sorted(literal_nodes.items(), key=lambda x: x[1]['id'])
    if sorted_literals:
        literal_features = [literal[1]['features'] for literal in sorted_literals]
        data['literal'].x = torch.tensor(literal_features, dtype=torch.float)
    else:
        data['literal'].x = torch.empty((0, 18))

    # Assign numeral features
    sorted_numerals = sorted(numeral_nodes.items(), key=lambda x: x[1]['id'])
    if sorted_numerals:
        numeral_features = [numeral[1]['features'] for numeral in sorted_numerals]
        data['numeral'].x = torch.tensor(np.array(numeral_features), dtype=torch.float)
    else: 
        data['numeral'].x = torch.empty((0, 1))

    
    # Create edge index dictionaries
    # operator_calledby_operator_edges
    if operator_calledby_operator_edges:
        src, dst = zip(*operator_calledby_operator_edges)
        data['operator', 'calledby', 'operator'].edge_index = torch.tensor([src, dst], dtype=torch.long)

    # table_scannedby_operator_edges
    if table_scannedby_operator_edges:
        src, dst = zip(*table_scannedby_operator_edges)
        data['table', 'scannedby', 'operator'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # column_outputby_operator_edges
    if column_outputby_operator_edges:
        src, dst = zip(*column_outputby_operator_edges)
        data['column', 'outputby', 'operator'].edge_index = torch.tensor([src, dst], dtype=torch.long)

    if column_containedby_table_edges:
        src, dst = zip(*column_containedby_table_edges)
        data['column', 'containedby', 'table'].edge_index = torch.tensor([src, dst], dtype=torch.long)

    # column_selfloop_column_edges
    if column_selfloop_column_edges:
        src, dst = zip(*column_selfloop_column_edges)
        data['column', 'selfloop', 'column'].edge_index = torch.tensor([src, dst], dtype=torch.long)

    # predicate_filters_operator_edges
    if predicate_filters_operator_edges:
        src, dst = zip(*predicate_filters_operator_edges)
        data['predicate', 'filters', 'operator'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    

    # # table_selfloop_table_edges
    # if table_selfloop_table_edges:
    #     src, dst = zip(*table_selfloop_table_edges)
    #     data['table', 'selfloop', 'table'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    


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
    logger = get_logger()

    # print(f"db_stats.keys() {db_stats.keys()}")
    unique_data_types = sorted(db_stats['unique_data_types'])
    data_type_mapping = {data_type: i for i, data_type in enumerate(unique_data_types)}

    table_nodes, column_nodes, predicate_nodes, operation_nodes, operator_nodes,literal_nodes, numeral_nodes, \
        table_scannedby_operator_edges, predicate_filters_operator_edges, column_outputby_operator_edges, \
        column_connects_operation_edges, operator_calledby_operator_edges, operation_filters_operator_edges, operation_connects_predicate_edges,  \
        literal_connects_operation_edges, numeral_connects_operation_edges, \
        literal_selfloop_literal_edges, numeral_selfloop_numeral_edges,  \
        table_selfloop_table_edges, column_selfloop_column_edges, predicate_connects_predicate_edges = parse_query_plan(logger, plan, conn, db_stats, data_type_mapping)


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
    print(f"literal_selfloop_literal_edges: {literal_selfloop_literal_edges}")
    print(f"numeral_selfloop_numeral_edges: {numeral_selfloop_numeral_edges}")
    print(f"table_selfloop_table_edges: {table_selfloop_table_edges}")
    print(f"column_selfloop_column_edges: {column_selfloop_column_edges}")
    print(f"predicate_connects_predicate_edges: {predicate_connects_predicate_edges}")