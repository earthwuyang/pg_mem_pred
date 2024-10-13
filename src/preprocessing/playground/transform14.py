import json
from functools import reduce
from operator import mul
import collections
import re
from types import SimpleNamespace
from tqdm import tqdm
from moz_sql_parser import parse
import sys

# ======================================
# Step 1: Load Database Statistics
# ======================================
# Add path to your local module if needed
sys.path.append('/home/wuy/DB/memory_prediction/zsce')
from collect_db_stats import collect_db_statistics

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
# Predicate Node Class
# ======================================

class PredicateNode:
    def __init__(self, text, children):
        self.text = text
        self.children = children
        self.column = None
        self.operator = None
        self.literal = None
        self.filter_feature = None

    def __str__(self):
        return self.to_tree_rep(depth=0)

    def to_dict(self):
        return dict(
            column=self.column,
            operator=str(self.operator),
            literal=self.literal,
            literal_feature=self.filter_feature,
            children=[c.to_dict() for c in self.children]
        )

    def parse_lines(self):
        # This method parses the predicate line
        keywords = [w.strip() for w in self.text.split(' ') if len(w.strip()) > 0]
        if all([k == 'AND' for k in keywords]):
            self.operator = 'AND'
        elif all([k == 'OR' for k in keywords]):
            self.operator = 'OR'
        else:
            op_mapping = {
                '= ANY': 'IN',
                '=': '=',
                '>=': '>=',
                '>': '>',
                '<=': '<=',
                '<': '<',
                '<>': '!=',
                'IS NULL': 'IS NULL',
                'IS NOT NULL': 'IS NOT NULL',
                'LIKE': 'LIKE',
                'NOT LIKE': 'NOT LIKE'
            }
            for op_str, op in op_mapping.items():
                if op_str in self.text:
                    self.operator = op
                    left, right = self.text.split(op_str)
                    self.column = left.strip()
                    self.literal = right.strip()
                    break

    def to_tree_rep(self, depth=0):
        rep_text = '\n' + ''.join(['\t'] * depth)
        rep_text += self.text
        for c in self.children:
            rep_text += c.to_tree_rep(depth=depth + 1)
        return rep_text

# ======================================
# Predicate Parsing Functions
# ======================================

def parse_recursively(filter_cond, offset):
    node_text = ''
    children = []
    while True:
        if offset >= len(filter_cond):
            return PredicateNode(node_text, children), offset

        if filter_cond[offset] == '(':
            child_node, offset = parse_recursively(filter_cond, offset + 1)
            children.append(child_node)
        elif filter_cond[offset] == ')':
            return PredicateNode(node_text, children), offset
        else:
            node_text += filter_cond[offset]
        offset += 1

def parse_filter(filter_cond):
    parse_tree, _ = parse_recursively(filter_cond, offset=0)
    assert len(parse_tree.children) == 1
    parse_tree = parse_tree.children[0]
    parse_tree.parse_lines()
    return parse_tree

# ======================================
# Define Helper Functions
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
        },
        'children': []
    }

    # Parse filter column if it exists
    if 'Filter' in node:
        filter_structure = parse_filter(node['Filter'])
        if filter_structure:
            transformed['plan_parameters']['filter_columns'] = filter_structure.to_dict()

    # Parse join conditions from 'Hash Cond' if it exists
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

    output_structure = {'parsed_plans': transformed_plans, 'database_stats': collect_db_statistics(conn_params), 'run_kwargs': {'hardware': 'qh1'}}

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
