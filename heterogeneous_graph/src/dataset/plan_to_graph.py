import torch
import re
import psycopg2
from torch_geometric.data import HeteroData


def get_relpages_reltuples(conn, table_name):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT relpages, reltuples FROM pg_class WHERE relname = %s;", (table_name,))
            result = cur.fetchone()
            relpages = result[0]
            reltuples = result[1]
        return relpages, reltuples
    except Exception as e:
        print(f"Error fetching relpages and reltuples for {table_name}: {e}")
        return 0, 0

# Fetch table size
def get_table_size(conn, table_name):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_total_relation_size(%s);", (table_name,))
            table_size = cur.fetchone()[0]  # Size in bytes
        return table_size
    except Exception as e:
        print(f"Error fetching table size for {table_name}: {e}")
        return 0

# Fetch number of columns and their data types
def get_columns_info(conn, table_name):
    try:
        with conn.cursor() as cur:
            # Fetch column names and data types
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = %s;
            """, (table_name,))
            columns = cur.fetchall()  # List of tuples: (column_name, data_type)
        
        return columns
    except Exception as e:
        print(f"Error fetching columns info for {table_name}: {e}")
        return []

# Fetch average width of a column
def get_column_features(conn, table_name, column_name):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT avg_width, correlation, n_distinct, null_frac
                FROM pg_stats
                WHERE tablename = %s AND attname = %s;
            """, (table_name, column_name))
            result = cur.fetchone()
            avg_width = result[0] if result and result[0] is not None else 0
            correlation = result[1] if result and result[1] is not None else 0
            n_distinct = result[2] if result and result[2] is not None else 0
            null_frac = result[3] if result and result[3] is not None else 0
            column_features = [avg_width, correlation, n_distinct, null_frac]
        return column_features
    except Exception as e:
        print(f"Error fetching column features for {table_name}.{column_name}: {e}")
        return [0, 0, 0, 0]


# One-hot encode data types with an 'unknown' category
def one_hot_encode_data_type(data_type, data_type_mapping):
    one_hot = [0] * (len(data_type_mapping) + 1)  # +1 for 'unknown'
    if data_type in data_type_mapping:
        index = data_type_mapping[data_type]
        one_hot[index] = 1
    else:
        # Assign 'unknown' category
        one_hot[-1] = 1
    return one_hot

# Fetch all unique data types from the database to create a mapping
def get_unique_data_types(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT data_type
                FROM information_schema.columns
                WHERE table_schema = 'public';
            """)
            data_types = [row[0] for row in cur.fetchall()]
        return data_types
    except Exception as e:
        print(f"Error fetching unique data types: {e}")
        return []


# ---------------------- Helper Functions ---------------------- #
def extract_columns(string):
    # Regex to extract column names with at least one letter in the table/alias part
    # column_pattern = re.compile(r'\b([A-Za-z]+\.\w+)\b')
    column_pattern = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*\.\w+)\b')
    columns = column_pattern.findall(string)

    return columns

# Helper function to traverse operators and extract tables, columns, and predicates
def traverse_operators(plan, table_nodes, column_nodes, predicate_nodes, operator_nodes,
                      table_scannedby_operator_edges, predicate_filters_operator_edges, column_outputby_operator_edges,
                      column_connects_predicate_edges, operator_calledby_operator_edges, 
                      operator_id_counter, global_node_id_counter, parent_operator_id=None):
    current_global_id = global_node_id_counter[0]
    current_operator_id = global_node_id_counter[0]
    operator_id_counter[0] += 1  # Increment the operator ID counter

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
        'global_id': global_node_id_counter[0],
        'type': plan_parameters.get('Node Type', 'Unknown'),
        'features': operator_features
    })
    global_node_id_counter[0] += 1
    # If there is a parent operator, add an edge (operator calls operator)
    if parent_operator_id is not None:
        operator_calledby_operator_edges.append((current_operator_id, parent_operator_id))

    # Extract tables, columns, predicates, and edges
    if 'Relation Name' in plan_parameters:
        table_name = plan_parameters['Relation Name']
        if table_name not in table_nodes:
            table_nodes[table_name] = {
                'id': len(table_nodes),
                'global_id': global_node_id_counter[0],
                'features': [0, 0]  # Placeholder, will be updated later
            }
            global_node_id_counter[0] += 1
        table_id = table_nodes[table_name]['global_id']
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
                    'global_id': global_node_id_counter[0],
                    'features': [0]  # Placeholder, will be updated later
                }
                global_node_id_counter[0] += 1

            predicate_id = predicate_nodes[predicate]['global_id']

            # Add edge: predicate filters operator
            predicate_filters_operator_edges.append((predicate_id, current_operator_id))

            for col in involved_columns:
                if col not in column_nodes:
                    column_nodes[col] = {
                        'id': len(column_nodes),
                        'global_id': global_node_id_counter[0],
                        'features': [0, 0]  # Placeholder, will be updated later
                    }
                    global_node_id_counter[0] += 1
                column_id = column_nodes[col]['global_id']
                # Add edge: predicate connects column
                column_connects_predicate_edges.append((column_id, predicate_id))

    output_list = plan_parameters.get('Output', [])
    for output_item in output_list:
        cols = extract_columns(output_item)
        for col in cols:
            if col not in column_nodes:
                column_nodes[col] = {
                    'id': len(column_nodes),
                    'global_id': global_node_id_counter[0],
                    'features': [0] * 9  # Placeholder, will be updated later
                }
                global_node_id_counter[0] += 1
            column_id = column_nodes[col]['global_id']
            # Add edge: column is output by operator
            column_outputby_operator_edges.append((column_id, current_operator_id))

    # Recurse into sub-plans
    if 'Plans' in plan_parameters:
        for sub_plan in plan_parameters['Plans']:
            traverse_operators(sub_plan, table_nodes, column_nodes, predicate_nodes, operator_nodes,
                      table_scannedby_operator_edges, predicate_filters_operator_edges, column_outputby_operator_edges,
                      column_connects_predicate_edges, operator_calledby_operator_edges, 
                      operator_id_counter, global_node_id_counter, current_operator_id)


# Function to parse the query plan and extract the tables, columns, and predicates
def parse_query_plan(plan, conn, data_type_mapping):
    table_nodes = {}       # table_name -> {'id': int, 'features': [...]}
    column_nodes = {}      # column_name -> {'id': int, 'features': [...]}
    predicate_nodes = {}   # predicate_str -> {'id': int, 'features': [...]}
    operator_nodes = []    # List of operators with features
    
    # all edge from bottom to top, while tree is parsed from top to bottom
    table_scannedby_operator_edges = []    
    predicate_filters_operator_edges = []  
    column_outputby_operator_edges = []    
    column_connects_predicate_edges = [] 
    operator_calledby_operator_edges = []    

    

    operator_id_counter = [0]  # Using a list to make it mutable in recursion
    global_node_id_counter = [0]

    traverse_operators(plan, table_nodes, column_nodes, predicate_nodes, operator_nodes,
                      table_scannedby_operator_edges, predicate_filters_operator_edges, column_outputby_operator_edges,
                      column_connects_predicate_edges, operator_calledby_operator_edges, 
                      operator_id_counter, global_node_id_counter)
    
    # Debug: Print extracted components
    # print(f"table_nodes: {table_nodes}")
    # print(f"column_nodes: {column_nodes}")
    # print(f"predicate_nodes: {predicate_nodes}")
    # print(f"operator_nodes: {operator_nodes}")

    # print(f"table_scannedby_operator_edges: {table_scannedby_operator_edges}")
    # print(f"predicate_filters_operator_edges: {predicate_filters_operator_edges}")
    # print(f"column_outputby_operator_edges: {column_outputby_operator_edges}")
    # print(f"column_connects_predicate_edges: {column_connects_predicate_edges}")
    # print(f"operator_calledby_operator_edges: {operator_calledby_operator_edges}")
    # exit()


    # Now, fetch actual features for tables and columns
    for table_name, table_info in table_nodes.items():
        relpages, reltuples = get_relpages_reltuples(conn, table_name)
        print(f"relpages: {relpages}, reltuples: {reltuples}")
        table_nodes[table_name]['features'] = [relpages, reltuples]
        
        # Update column features
        columns = get_columns_info(conn, table_name)
        for column_name, data_type in columns:
            full_column_name = f"{table_name}.{column_name}"
            if full_column_name in column_nodes:
                avg_width, correlation, n_distinct, null_frac = get_column_features(conn, table_name, column_name)
                one_hot = one_hot_encode_data_type(data_type, data_type_mapping)  # Unique data types: {'character': 0, 'character varying': 1, 'date': 2, 'integer': 3, 'numeric': 4}
                column_nodes[full_column_name]['features'] = [avg_width, correlation, n_distinct, null_frac] + one_hot

    # Update predicate features: [predicate_length]
    for pred, pred_info in predicate_nodes.items():
        # print(f"pred: {pred}, pred_info: {pred_info}")
        predicate_length = len(pred)
        predicate_nodes[pred]['features'] = [predicate_length]
    # exit()
    return table_nodes, column_nodes, predicate_nodes, operator_nodes, \
                      table_scannedby_operator_edges, predicate_filters_operator_edges, column_outputby_operator_edges, \
                      column_connects_predicate_edges, operator_calledby_operator_edges

# Function to create the heterogeneous graph from parsed components
def create_hetero_graph(table_nodes, column_nodes, predicate_nodes, operator_nodes,
                      table_scannedby_operator_edges, predicate_filters_operator_edges, column_outputby_operator_edges,
                      column_connects_predicate_edges, operator_calledby_operator_edges, peakmem):
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
    
    # Validate column_features length
    expected_column_features_length = len(column_features[0]) if column_features else 0
    for idx, features in enumerate(column_features):
        if len(features) != expected_column_features_length:
            print(f"Warning: Column {sorted_columns[idx][0]} has incorrect feature length {len(features)} (expected {expected_column_features_length}).")
            # Pad with zeros if necessary
            if len(features) < expected_column_features_length:
                padded_features = features + [0] * (expected_column_features_length - len(features))
                column_features[idx] = padded_features
            else:
                # Truncate if necessary
                column_features[idx] = features[:expected_column_features_length]
    
    data['column'].x = torch.tensor(column_features, dtype=torch.float)
    
    # Assign predicate features
    sorted_predicates = sorted(predicate_nodes.items(), key=lambda x: x[1]['id'])
    predicate_features = [predicate[1]['features'] for predicate in sorted_predicates]
    data['predicate'].x = torch.tensor(predicate_features, dtype=torch.float)
    
    # table_scannedby_operator_edges, predicate_filters_operator_edges, column_outputby_operator_edges,
    # column_connects_predicate_edges, operator_calledby_operator_edges
    
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
    
    # Assign the target
    data.y = torch.tensor([peakmem], dtype=torch.float)
    
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
    _ = parse_query_plan(plan, conn, data_type_mapping)
