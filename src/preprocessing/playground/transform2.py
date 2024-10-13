import json
from functools import reduce
from operator import mul
import collections
import re
from types import SimpleNamespace

# ================================
# Step 1: Load Database Statistics
# ================================

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
table_sizes = dict()
for table_stat in database_stats.table_stats:
    table_sizes[table_stat.relname] = table_stat.reltuples

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

def load_explain_plan(json_str):
    """Load and parse PostgreSQL's EXPLAIN JSON plan."""
    return json.loads(json_str)

def parse_output_columns(output_list, column_id_mapping, table_id_mapping):
    """
    Parse the Output field, map (table, column) to column_id.

    :param output_list: List of output strings from the plan.
    :param column_id_mapping: Dict mapping (table, column) to column_id.
    :param table_id_mapping: Dict mapping table names to table IDs.
    :return: List of dictionaries with aggregation and column_ids.
    """
    parsed_columns = []
    for item in output_list:
        aggregation = 'None'
        columns = []

        # Detect aggregation functions
        agg_match = re.match(r'(\w+)\((.+)\)', item.strip())
        if agg_match:
            agg_func = agg_match.group(1).upper()
            expr = agg_match.group(2)
            if agg_func in {'SUM', 'AVG', 'COUNT', 'MIN', 'MAX'}:
                aggregation = agg_func
            else:
                aggregation = 'None'  # Unsupported or unknown aggregation
        else:
            expr = item  # No aggregation

        # Extract (table, column) pairs from the expression
        # This regex matches patterns like table.column or "table"."column"
        pattern = re.compile(r'\"?(\w+)\"?\."?(\w+)\"?')
        matches = pattern.findall(expr)

        for table, column in matches:
            key = (table, column)
            if key in column_id_mapping:
                columns.append(column_id_mapping[key])
            else:
                raise ValueError(f"Column mapping not found for {table}.{column}")

        # If no aggregation and columns are directly referenced
        if aggregation == 'None' and not columns:
            # Attempt to find columns without aggregation
            simple_pattern = re.compile(r'\"?(\w+)\"?\."?(\w+)\"?')
            simple_matches = simple_pattern.findall(expr)
            for table, column in simple_matches:
                key = (table, column)
                if key in column_id_mapping:
                    columns.append(column_id_mapping[key])
                else:
                    raise ValueError(f"Column mapping not found for {table}.{column}")

        parsed_columns.append({'aggregation': aggregation, 'columns': columns})

    return parsed_columns

def transform_node(node, column_id_mapping, table_id_mapping):
    """
    Transform a single node to the target format.

    :param node: The node dictionary from EXPLAIN JSON.
    :param column_id_mapping: Dict mapping (table, column) to column_id.
    :param table_id_mapping: Dict mapping table names to table IDs.
    :return: Transformed node dictionary.
    """
    transformed = {
        'plain_content': [],  # Keep empty as per target format
        'plan_parameters': {
            'op_name': node.get('Node Type', 'Unknown'),
            'est_startup_cost': node.get('Startup Cost', 0.0),
            'est_cost': node.get('Total Cost', 0.0),
            'est_card': float(node.get('Plan Rows', 1)),
            'est_width': float(node.get('Plan Width', 0)),
            # Uncomment these if actual execution metrics are available
            # 'act_startup_cost': float(node.get('Actual Startup Time', 0.0)),
            # 'act_time': float(node.get('Actual Total Time', 0.0)),
            # 'act_card': float(node.get('Actual Rows', 0.0)),
            'output_columns': parse_output_columns(node.get('Output', []), column_id_mapping, table_id_mapping),
            # 'act_children_card': 0.0,  # To be calculated
            'est_children_card': 0.0,  # To be calculated
            'workers_planned': node.get('Workers Planned', 0)
        },
        'children': []
    }

    # Process child nodes recursively
    children = node.get('Plans', [])
    for child in children:
        transformed_child = transform_node(child, column_id_mapping, table_id_mapping)
        transformed['children'].append(transformed_child)

    # Calculate est_children_card and act_children_card
    if children:
        est_card_product = reduce(mul, [float(child.get('Plan Rows', 1)) for child in children], 1.0)
        # If actual cardinalities are present, calculate product; else, set to 1.0
        act_card_product = 1.0
        for child in transformed['children']:
            act_card = child['plan_parameters'].get('act_card', 1.0)
            act_card_product *= act_card
        transformed['plan_parameters']['est_children_card'] = est_card_product
        # transformed['plan_parameters']['act_children_card'] = act_card_product
    else:
        transformed['plan_parameters']['est_children_card'] = 1.0
        # transformed['plan_parameters']['act_children_card'] = 1.0

    return transformed

def transform_plan(explain_json, column_id_mapping, table_id_mapping):
    """
    Transform the entire query plan.

    :param explain_json: The parsed EXPLAIN JSON dictionary.
    :param column_id_mapping: Dict mapping (table, column) to column_id.
    :param table_id_mapping: Dict mapping table names to table IDs.
    :return: Transformed plan dictionary.
    """
    plan = explain_json.get('Plan', {})
    transformed_plan = transform_node(plan, column_id_mapping, table_id_mapping)
    return transformed_plan

# ======================================
# Step 3: Update parse_output_columns
# ======================================

def parse_output_columns_refined(output_list, column_id_mapping, table_id_mapping):
    """
    Enhanced parse_output_columns function using column_id_mapping.

    :param output_list: List of output strings from the plan.
    :param column_id_mapping: Dict mapping (table, column) to column_id.
    :param table_id_mapping: Dict mapping table names to table IDs.
    :return: List of dictionaries with aggregation and column_ids.
    """
    parsed_columns = []
    for item in output_list:
        aggregation = 'None'
        columns = []

        # Detect aggregation functions using regex
        agg_match = re.match(r'(\w+)\((.+)\)', item.strip())
        if agg_match:
            agg_func = agg_match.group(1).upper()
            expr = agg_match.group(2)
            if agg_func in {'SUM', 'AVG', 'COUNT', 'MIN', 'MAX'}:
                aggregation = agg_func
            else:
                aggregation = 'None'  # Unsupported or unknown aggregation
        else:
            expr = item  # No aggregation

        # Extract (table, column) pairs from the expression
        # This regex matches patterns like table.column or "table"."column"
        pattern = re.compile(r'\"?(\w+)\"?\."?(\w+)\"?')
        matches = pattern.findall(expr)

        for table, column in matches:
            key = (table, column)
            # print(f"key is {key}")
            # print()
            # print(f"column_id_mapping is {column_id_mapping}")
            if key in column_id_mapping:
                columns.append(column_id_mapping[key])
            else:
                raise ValueError(f"Column mapping not found for {table}.{column}")

        # If no aggregation and columns are directly referenced
        if aggregation == 'None' and not columns:
            # Attempt to find columns without aggregation
            simple_pattern = re.compile(r'\"?(\w+)\"?\."?(\w+)\"?')
            simple_matches = simple_pattern.findall(expr)
            for table, column in simple_matches:
                key = (table, column)
                if key in column_id_mapping:
                    columns.append(column_id_mapping[key])
                else:
                    raise ValueError(f"Column mapping not found for {table}.{column}")

        parsed_columns.append({'aggregation': aggregation, 'columns': columns})

    return parsed_columns

# ======================================
# Step 4: Update transform_node Function
# ======================================

def transform_node_refined(node, column_id_mapping, table_id_mapping):
    """
    Refined transform_node function using column_id_mapping.

    :param node: The node dictionary from EXPLAIN JSON.
    :param column_id_mapping: Dict mapping (table, column) to column_id.
    :param table_id_mapping: Dict mapping table names to table IDs.
    :return: Transformed node dictionary.
    """
    transformed = {
        'plain_content': [],  # Keep empty as per target format
        'plan_parameters': {
            'op_name': node.get('Node Type', 'Unknown'),
            'est_startup_cost': node.get('Startup Cost', 0.0),
            'est_cost': node.get('Total Cost', 0.0),
            'est_card': float(node.get('Plan Rows', 1)),
            'est_width': float(node.get('Plan Width', 0)),
            # Uncomment these if actual execution metrics are available
            # 'act_startup_cost': float(node.get('Actual Startup Time', 0.0)),
            # 'act_time': float(node.get('Actual Total Time', 0.0)),
            # 'act_card': float(node.get('Actual Rows', 0.0)),
            'output_columns': parse_output_columns_refined(node.get('Output', []), column_id_mapping, table_id_mapping),
            # 'act_children_card': 0.0,  # To be calculated
            'est_children_card': 0.0,  # To be calculated
            'workers_planned': node.get('Workers Planned', 0)
        },
        'children': []
    }

    # Process child nodes recursively
    children = node.get('Plans', [])
    for child in children:
        transformed_child = transform_node_refined(child, column_id_mapping, table_id_mapping)
        transformed['children'].append(transformed_child)

    # Calculate est_children_card and act_children_card
    if children:
        est_card_product = reduce(mul, [float(child.get('Plan Rows', 1)) for child in children], 1.0)
        # If actual cardinalities are present, calculate product; else, set to 1.0
        act_card_product = 1.0
        for child in transformed['children']:
            act_card = child['plan_parameters'].get('act_card', 1.0)
            act_card_product *= act_card
        transformed['plan_parameters']['est_children_card'] = est_card_product
        # transformed['plan_parameters']['act_children_card'] = act_card_product
    else:
        transformed['plan_parameters']['est_children_card'] = 1.0
        # transformed['plan_parameters']['act_children_card'] = 1.0

    return transformed

# ======================================
# Step 5: Update transform_plan Function
# ======================================

def transform_plan_refined(explain_json, column_id_mapping, table_id_mapping):
    """
    Refined transform_plan function.

    :param explain_json: The parsed EXPLAIN JSON dictionary.
    :param column_id_mapping: Dict mapping (table, column) to column_id.
    :param table_id_mapping: Dict mapping table names to table IDs.
    :return: Transformed plan dictionary.
    """
    plan = explain_json.get('Plan', {})
    transformed_plan = transform_node_refined(plan, column_id_mapping, table_id_mapping)
    return transformed_plan

# ======================================
# Step 6: Main Function
# ======================================

def main():
    # Path to the EXPLAIN JSON file
    json_file_path = '/home/wuy/DB/pg_mem_data/tpch_sf1/tiny_plans.json'  # Update with your actual file path

    # Load the EXPLAIN JSON plan
    with open(json_file_path, 'r') as f:
        plans = json.load(f)

    # Process each plan
    transformed_plans = []
    for idx, plan in enumerate(plans):
        print(f"Processing plan {idx + 1}/{len(plans)}")
        transformed_plan = transform_plan_refined(plan, column_id_mapping, table_id_mapping)
        transformed_plans.append(transformed_plan)


    # Wrap the transformed plans in the desired structure
    output_structure = {'parsed_plans': transformed_plans}

    # Save the transformed plans to a JSON file
    output_file_path = '/home/wuy/DB/pg_mem_data/tpch_sf1/transformed_plans.json'  # Update with your desired output path
    with open(output_file_path, 'w') as f:
        json.dump(output_structure, f, indent=4, ensure_ascii=False)

    print(f"Transformed plans have been saved to {output_file_path}")

    # Optionally, print the transformed plans
    # print(json.dumps(output_structure, indent=4, ensure_ascii=False))

if __name__ == '__main__':
    main()
