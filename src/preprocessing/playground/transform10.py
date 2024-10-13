import json
from functools import reduce
from operator import mul
import collections
import re
from types import SimpleNamespace
from tqdm import tqdm
import sys
sys.path.append('/home/wuy/DB/memory_prediction/zsce')
from collect_db_stats import collect_db_statistics

# ======================================
# Step 1: Load Database Statistics
# ======================================

# Path to the database statistics JSON file
database_stats_file = '/home/wuy/DB/pg_mem_data/tpch_sf1/database_stats.json'

# Load database statistics
with open(database_stats_file, 'r') as f:
    database_stats = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

# Initialize mappings
column_id_mapping = dict()
table_id_mapping = dict()
partial_column_name_mapping = collections.defaultdict(set)

# Map table names to their sizes
table_sizes = {table_stat.relname: table_stat.reltuples for table_stat in database_stats.table_stats}

# Generate column_id_mapping and partial_column_name_mapping
for i, column_stat in enumerate(database_stats.column_stats):
    table = column_stat.tablename
    column = column_stat.attname
    column_stat.table_size = table_sizes.get(table, 0)
    column_id_mapping[(table, column)] = i
    partial_column_name_mapping[column].add(table)

# Generate table_id_mapping
for i, table_stat in enumerate(database_stats.table_stats):
    table = table_stat.relname
    table_id_mapping[table] = i

# ======================================
# Step 2: Define Helper Functions
# ======================================

def parse_output_columns_refined(output_list, column_id_mapping):
    parsed_columns = []
    for item in output_list:
        aggregation = 'None'
        columns = []

        agg_match = re.match(r'(\w+)\((.+)\)', item.strip())
        if agg_match:
            agg_func = agg_match.group(1).upper()
            expr = agg_match.group(2)
            if agg_func in {'SUM', 'AVG', 'COUNT', 'MIN', 'MAX'}:
                aggregation = agg_func
        else:
            expr = item  # No aggregation

        pattern = re.compile(r'\"?(\w+)\"?\."?(\w+)\"?')
        matches = pattern.findall(expr)

        for table, column in matches:
            key = (table, column)
            if key in column_id_mapping:
                columns.append(column_id_mapping[key])
            else:
                raise ValueError(f"Column mapping not found for {table}.{column}")

        parsed_columns.append({'aggregation': aggregation, 'columns': columns})

    return parsed_columns


import re

# Assuming a column to ID mapping
column_id_mapping = {
    ('partsupp', 'ps_supplycost'): 0
}

def parse_filter(filter_expression):
    print(f"filter_expression: {filter_expression}")
    
    # Compound filter regex (e.g., "col1 = 5 AND col2 > 3")
    compound_regex = re.compile(r'\(\s*(.*?)\s*(AND|OR|NOT)\s*(.*?)\s*\)')
    
    # Simple filter regex (e.g., "partsupp.ps_supplycost <= 48.34")
    simple_regex = re.compile(r'(\w+\.\w+)\s*([<>!=]=?|IS NULL|IS NOT NULL)\s*([\d\.\-]+|\'.*?\'|\".*?\")')

    # Check for compound filters
    match = compound_regex.match(filter_expression)
    if match:
        left_part = parse_filter(match.group(1).strip())
        right_part = parse_filter(match.group(3).strip())
        return {
            "column": None,
            "operator": match.group(2),
            "literal": None,
            "literal_feature": None,
            "children": [left_part, right_part]
        }

    # Check for simple filters
    match_simple = simple_regex.match(filter_expression)
    print(f"match_simple: {match_simple}")
    if match_simple:
        column_name = match_simple.group(1)
        operator = match_simple.group(2)
        literal = match_simple.group(3).strip('"\'')

        # Look up the column ID from column_id_mapping
        table_column = tuple(column_name.split('.'))
        column_id = column_id_mapping.get(table_column)

        # Convert literal to appropriate type (int or float)
        try:
            if '.' in literal:
                literal = float(literal)
            else:
                literal = int(literal)
        except ValueError:
            # Keep it as a string if conversion fails
            pass

        return {
            "column": column_id,  # Use the mapped column ID
            "operator": operator,
            "literal": literal,
            "literal_feature": 0,  # Default feature index
            "children": []  # No children for simple filters
        }

    return None  # If unable to parse




def transform_node_refined(node, column_id_mapping, table_id_mapping):
    transformed = {
        'plain_content': [],
        'plan_parameters': {
            'op_name': node.get('Node Type', 'Unknown'),
            'est_startup_cost': node.get('Startup Cost', 0.0),
            'est_cost': node.get('Total Cost', 0.0),
            'est_card': float(node.get('Plan Rows', 1)),
            'est_width': float(node.get('Plan Width', 0)),
            'act_startup_cost': float(node.get('Actual Startup Time', 0.0)) if 'Actual Startup Time' in node else 0.0,
            'act_time': float(node.get('Actual Total Time', 0.0)) if 'Actual Total Time' in node else 0.0,
            'act_card': float(node.get('Actual Rows', 0.0)) if 'Actual Rows' in node else 0.0,
            'output_columns': parse_output_columns_refined(node.get('Output', []), column_id_mapping),
            'est_children_card': 0.0,
            'act_children_card': 0.0,
            'workers_planned': node.get('Workers Planned', 0),
            # 'filter_columns': None  # To hold the filter structure
        },
        'children': []
    }

    # Parse filter column if it exists
    if 'Filter' in node:
        filter_structure = parse_filter(node['Filter'])
        print(f"filter_structure: {filter_structure}")
        while 1:pass
        if filter_structure:
            transformed['plan_parameters']['filter_columns'] = filter_structure

    # Parse join conditions if 'Hash Cond' exists
    if 'Hash Cond' in node:
        transformed['plan_parameters']['join_conds'] = node['Hash Cond']

    # Process child nodes recursively
    children = node.get('Plans', [])
    for child in children:
        transformed_child = transform_node_refined(child, column_id_mapping, table_id_mapping)
        transformed['children'].append(transformed_child)

    # Calculate est_children_card and act_children_card
    if children:
        est_card_product = reduce(mul, [float(child.get('Plan Rows', 1)) for child in children], 1.0)
        act_card_product = 1.0
        for child in transformed['children']:
            act_card = child['plan_parameters'].get('act_card', 1.0)
            act_card_product *= act_card
        transformed['plan_parameters']['est_children_card'] = est_card_product
        transformed['plan_parameters']['act_children_card'] = act_card_product
    else:
        transformed['plan_parameters']['est_children_card'] = 1.0
        transformed['plan_parameters']['act_children_card'] = 1.0

    return transformed

def transform_plan_refined(plan, column_id_mapping, table_id_mapping):
    transformed_plan = transform_node_refined(plan['Plan'], column_id_mapping, table_id_mapping)

    # Extract 'peakmem' from the plan if it exists
    transformed_plan['peakmem'] = plan.get('peakmem', 0)

    return transformed_plan

# ======================================
# Step 3: Main Function
# ======================================

def main(mode):
    json_file_path = f'/home/wuy/DB/pg_mem_data/tpch_sf1/{mode}_plans.json'

    with open(json_file_path, 'r') as f:
        plans = json.load(f)

    transformed_plans = []
    for idx, plan in tqdm(enumerate(plans), total=len(plans), desc=f'Processing {mode} plans'):
        try:
            transformed_plan = transform_plan_refined(plan, column_id_mapping, table_id_mapping)
            transformed_plans.append(transformed_plan)
        except ValueError as e:
            print(f"Error processing plan {idx + 1}: {e}")
            continue

    database = 'tpch_sf1'
    conn_params = {
        "dbname": database,
        "user": "wuy",
        "password": "",
        "host": "localhost"
    }

    output_structure = {
        'parsed_plans': transformed_plans, 
        'database_stats': collect_db_statistics(conn_params), 
        'run_kwargs': {'hardware': 'qh1'}
    }

    output_file_path = f'/home/wuy/DB/pg_mem_data/tpch_sf1/zsce/{mode}_plans.json'
    with open(output_file_path, 'w') as f:
        json.dump(output_structure, f, indent=4, ensure_ascii=False)

    print(f"Transformed plans for '{mode}' have been saved to {output_file_path}")

# ======================================
# Step 4: Execute the Transformation
# ======================================

if __name__ == '__main__':
    for mode in ['tiny']:
        main(mode)
