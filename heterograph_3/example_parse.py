import re
import networkx as nx
from typing import Any, Dict, List, Tuple

# Define node and edge types
NODE_TYPES = ['Operator', 'Table', 'Column', 'Predicate', 'Operation', 'Literal', 'Numeral']
EDGE_TYPES = [
    ('Operator', 'CALLS', 'Operator'),
    ('Operator', 'INVOLVES', 'Table'),
    ('Operator', 'OUTPUTS', 'Column'),
    ('Operator', 'FILTERS', 'Predicate'),
    ('Predicate', 'USES', 'Operation'),
    ('Operation', 'USES_COLUMN', 'Column'),
    ('Operation', 'USES_LITERAL', 'Literal'),
    ('Table', 'HAS_COLUMN', 'Column'),
    # Add more edge types as needed
]

# Initialize the graph
graph = nx.MultiDiGraph()

# Helper functions for one-hot encoding (as defined in your original code)
def one_hot_encode_data_type(data_type, data_type_mapping):
    one_hot = [0] * (len(data_type_mapping) + 1)  # +1 for 'unknown'
    if data_type in data_type_mapping:
        index = data_type_mapping[data_type]
        one_hot[index] = 1
    else:
        one_hot[-1] = 1
    return one_hot

def one_hot_encode_andornot_type(predicate_type):
    andornot_type_mapping = {
        'AND': 0,
        'OR': 1,
        'NOT': 2,
    }
    one_hot = [0] * 4  # AND, OR, NOT, UNKNOWN
    if predicate_type.upper() in andornot_type_mapping:
        index = andornot_type_mapping[predicate_type.upper()]
        one_hot[index] = 1
    else:
        one_hot[-1] = 1
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
    one_hot = [0] * 8  # 8 types
    if operation_type.upper() in operation_type_mapping:
        index = operation_type_mapping[operation_type.upper()]
        one_hot[index] = 1
    else:
        one_hot[-1] = 1
    return one_hot

def extract_columns(string):
    column_pattern = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*\.\w+)\b')
    columns = column_pattern.findall(string)
    return columns

# Enhanced parse_predicate function
def parse_predicate(predicate: str, graph: nx.MultiDiGraph, parent_operator_id: Any, db_stats: Dict[str, Any]):
    """
    Parses a predicate string and updates the graph with Predicate, Operation, Column, and Literal nodes.
    """
    # This is a placeholder for actual predicate parsing logic.
    # You might need to use a SQL parser or write a more robust parser.
    # For simplicity, we'll handle simple binary operations and logical connectors.

    # Example predicate: "nation.n_regionkey = region.r_regionkey AND nation.n_nationkey <> 3"

    tokens = re.split(r'\s+(AND|OR|NOT)\s+', predicate, flags=re.IGNORECASE)
    stack = []
    current_predicate_id = None

    i = 0
    while i < len(tokens):
        token = tokens[i].strip()
        if token.upper() in ['AND', 'OR', 'NOT']:
            predicate_node = token.upper()
            graph.add_node(predicate_node, type='Predicate', features=one_hot_encode_andornot_type(predicate_node))
            if current_predicate_id:
                graph.add_edge(predicate_node, current_predicate_id, type='USES')
            current_predicate_id = predicate_node
            i += 1
        else:
            # Handle binary operations
            match = re.match(r'(\w+\.\w+)\s*(=|<>|>|>=|<|<=|LIKE)\s*(\w+\.\w+|\d+)', token)
            if match:
                left, op, right = match.groups()
                operation_node = f"Operation_{left}_{op}_{right}"
                graph.add_node(operation_node, type='Operation', features=one_hot_encode_operation_type(op))
                graph.add_edge(operation_node, 'Predicate', type='USES')  # Connect operation to current predicate

                # Add Column node
                graph.add_node(left, type='Column')
                graph.add_edge(left, operation_node, type='USES_COLUMN')
                graph.add_node(right, type='Column' if '.' in right else 'Literal')
                if '.' in right:
                    graph.add_edge(right, operation_node, type='USES_COLUMN')
                elif right.isdigit():
                    graph.add_node(right, type='Numeral')
                    graph.add_edge(right, operation_node, type='USES_LITERAL')
                else:
                    graph.add_node(right, type='Literal')
                    graph.add_edge(right, operation_node, type='USES_LITERAL')

                # Connect operation to parent operator
                graph.add_edge(operation_node, parent_operator_id, type='FILTERS')

            i += 1
def traverse_operators(plan: Dict[str, Any], graph: nx.MultiDiGraph, db_stats: Dict[str, Any], parent_operator_id: Any = None):
    """
    Recursively traverses the execution plan and updates the graph with nodes and edges.
    """
    operator_type = plan.get('Node Type', 'Unknown')
    operator_id = f"Operator_{len(graph.nodes)}"
    operator_features = [
        plan.get('Startup Cost', 0.0),
        plan.get('Total Cost', 0.0),
        plan.get('Plan Rows', 0),
        plan.get('Plan Width', 0)
    ]
    graph.add_node(operator_id, type='Operator', operator_type=operator_type, features=operator_features)

    if parent_operator_id:
        graph.add_edge(operator_id, parent_operator_id, type='CALLS')

    # If the operator involves a table
    relation_name = plan.get('Relation Name')
    if relation_name:
        table_id = f"Table_{relation_name}"
        graph.add_node(table_id, type='Table', features=[
            db_stats['tables'][relation_name]['relpages'],
            db_stats['tables'][relation_name]['reltuples']
        ])
        graph.add_edge(table_id, operator_id, type='INVOLVES')

        # Connect table to its columns
        for column_name in db_stats['tables'][relation_name]['column_features']:
            column_full_name = column_name  # Corrected line
            column_id = f"Column_{column_full_name}"
            graph.add_node(column_id, type='Column', features=db_stats['tables'][relation_name]['column_features'][column_full_name])
            graph.add_edge(column_id, table_id, type='HAS_COLUMN')

    # Extract and parse predicates
    for key in ['Hash Cond', 'Filter', 'Index Cond']:
        if key in plan:
            predicate_str = plan[key]
            parse_predicate(predicate_str, graph, operator_id, db_stats)

    # Handle output columns
    output_list = plan.get('Output', [])
    for output_item in output_list:
        cols = extract_columns(output_item)
        for col in cols:
            column_id = f"Column_{col}"
            if not graph.has_node(column_id):
                graph.add_node(column_id, type='Column', features=[0]*10)  # Placeholder features
            graph.add_edge(column_id, operator_id, type='OUTPUTS')

    # Recurse into sub-plans
    for sub_plan in plan.get('Plans', []):
        traverse_operators(sub_plan, graph, db_stats, parent_operator_id=operator_id)


# Function to parse the entire query plan
def parse_query_plan(plan: Dict[str, Any], db_stats: Dict[str, Any]) -> nx.MultiDiGraph:
    """
    Parses the PostgreSQL EXPLAIN JSON plan and constructs a heterogeneous graph.
    """
    graph = nx.MultiDiGraph()
    traverse_operators(plan['Plan'], graph, db_stats)
    return graph

# Example usage
if __name__ == "__main__":
    import json

    # Sample plan (as provided)
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

    # Sample db_stats (You need to provide actual statistics)
    db_stats = {
        'unique_data_types': ['character', 'character varying', 'date', 'integer', 'numeric'],
        'tables': {
            'nation': {
                'relpages': 10,
                'reltuples': 100,
                'column_features': {
                    'nation.n_nationkey': {
                        'avg_width': 4.0,
                        'correlation': 1.0,
                        'n_distinct': 25,
                        'null_frac': 0.0,
                        'data_type': 'integer'
                    },
                    'nation.n_name': {
                        'avg_width': 25.0,
                        'correlation': 0.0,
                        'n_distinct': 25,
                        'null_frac': 0.0,
                        'data_type': 'character varying'
                    },
                    'nation.n_regionkey': {
                        'avg_width': 4.0,
                        'correlation': 1.0,
                        'n_distinct': 5,
                        'null_frac': 0.0,
                        'data_type': 'integer'
                    },
                    'nation.n_comment': {
                        'avg_width': 152.0,
                        'correlation': 0.0,
                        'n_distinct': 25,
                        'null_frac': 0.0,
                        'data_type': 'character varying'
                    },
                }
            },
            'region': {
                'relpages': 5,
                'reltuples': 25,
                'column_features': {
                    'region.r_regionkey': {
                        'avg_width': 4.0,
                        'correlation': 1.0,
                        'n_distinct': 5,
                        'null_frac': 0.0,
                        'data_type': 'integer'
                    }
                }
            }
        }
    }

    # Parse the plan into a graph
    graph = parse_query_plan(plan, db_stats)

    # Visualize the graph (optional)
    import matplotlib.pyplot as plt

    pos = nx.spring_layout(graph, k=0.5, iterations=50)
    node_colors = [node[1]['type'] for node in graph.nodes(data=True)]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 20))

    # Aggregate edge labels
    aggregated_edge_labels = {}
    for u, v, data in graph.edges(data=True):
        edge_type = data.get('type', '')
        if (u, v) in aggregated_edge_labels:
            aggregated_edge_labels[(u, v)] += f", {edge_type}"
        else:
            aggregated_edge_labels[(u, v)] = edge_type


    # Define color mapping for node types
    color_map = {
        'Operator': 'lightblue',
        'Table': 'lightgreen',
        'Column': 'orange',
        'Predicate': 'red',
        'Operation': 'yellow',
        'Literal': 'pink',
        'Numeral': 'purple'
    }

    # Assign colors based on node types
    node_colors = [color_map.get(data['type'], 'grey') for _, data in graph.nodes(data=True)]

    # Generate positions for all nodes
    pos = nx.spring_layout(graph, k=0.5, iterations=50)

    # Draw nodes with colors
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=1500)

    # Draw edges without labels first
    nx.draw_networkx_edges(graph, pos, arrowstyle='->', arrowsize=20)

    # Draw node labels
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold')

    # Draw aggregated edge labels
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels=aggregated_edge_labels, font_color='black', font_size=8
    )

    # Display the graph
    plt.axis('off')
    plt.savefig('example_parse.png')
