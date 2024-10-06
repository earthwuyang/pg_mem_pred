def parse_plan(plan, statistics, parent=None, nodes=None, edges=None, node_id=0):
    """
    Recursively parse the JSON plan and build node features and edges.
    
    Args:
        plan (dict): The JSON plan.
        parent (int): The parent node ID.
        nodes (list): List to store node features.
        edges (list): List to store edge connections.
        node_id (int): Unique identifier for nodes.
        
    Returns:
        node_id (int): Updated node ID.
    """
    if nodes is None:
        nodes = []
    if edges is None:
        edges = []
        
    current_id = node_id
    # print(f"plan: {plan}")
    if 'Plan' in plan:
        plan_node = plan['Plan']
    else: 
        plan_node = plan
    
    # Extract features from the current node
    features = extract_features(plan_node, statistics)
    nodes.append(features)
    
    # If there is a parent, add an edge
    if parent is not None:
        edges.append([parent, current_id])
    
    # Initialize children if not present
    children = plan_node.get('Plans', [])
    
    # Recursively parse children
    for child in children:
        node_id += 1
        node_id = parse_plan(child, statistics, parent=current_id, nodes=nodes, edges=edges, node_id=node_id)
    
    return node_id

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

    for key in statistics:
        
        # if key in ['Output', 'Hash Cond', 'Filter', 'Index Cond', 'Join Filter', 'Recheck Cond', 'Merge Cond', 'One-Time Filter']:
        #     continue
        if key in ['Startup Cost', 'Total Cost', 'Plan Rows', 'Plan Width', 'Workers Planned', 'Node Type', 'Join Type', 'Strategy', 'Partial Mode', 'Parent Relationship', 'Scan Direction']:
 
            if statistics[key]['type'] == 'numerical':
                value = (plan_node.get(key, 0.0) - statistics[key]['center']) / statistics[key]['scale']
                feature_vector.append(value)
            elif statistics[key]['type'] == 'categorical':
                value = plan_node.get(key, 'unknown')
                # print(f"key: {key}, value is {value}")
                one_hot_features = one_hot_encode(statistics[key]['value_dict'].get(value, statistics[key]['no_vals']), statistics[key]['no_vals']+1)  # unknown will map to an extra number in the directory
                feature_vector.extend(one_hot_features)
                # feature_vector.append(statistics[key]['value_dict'].get(value, statistics[key]['no_vals'])) # unknown will map to an extra number in the directory
    # print(f"len(feature_vector): {len(feature_vector)}")
    return feature_vector


    numerical_features = {
        'Startup Cost': plan_node.get('Startup Cost', 0.0),
        'Total Cost': plan_node.get('Total Cost', 0.0),
        'Plan Rows': plan_node.get('Plan Rows', 0.0),
        'Plan Width': plan_node.get('Plan Width', 0.0),
        'Workers Planned': plan_node.get('Workers Planned', 0.0)
    }
    # # Numerical features
    # numerical_features = [
    #     plan_node.get('Startup Cost', 0.0),
    #     plan_node.get('Total Cost', 0.0),
    #     plan_node.get('Plan Rows', 0.0),
    #     plan_node.get('Plan Width', 0.0),
    #     plan_node.get('Workers Planned', 0.0)
    # ]
    # feature_vector.extend(numerical_features)
    
    # Categorical features: Node Type, Join Type, etc.
    categorical_features = [
        plan_node.get('Node Type', ''),
        plan_node.get('Join Type', ''),
        plan_node.get('Strategy', ''),
        plan_node.get('Partial Mode', ''),
        plan_node.get('Parent Relationship', ''),
        plan_node.get('Scan Direction', ''),
        plan_node.get('Filter', ''),
        plan_node.get('Hash Cond', ''),
        plan_node.get('Index Cond', ''),
        plan_node.get('Join Filter', '')
    ]
    
    # Convert categorical features to numerical via one-hot encoding or other encoding schemes
    # For simplicity, we'll use a basic encoding: assign a unique integer to each category
    # In practice, you might want to use more sophisticated encoding methods
    categorical_dict = {
        'Node Type': {},
        'Join Type': {},
        'Strategy': {},
        'Partial Mode': {},
        'Parent Relationship': {},
        'Scan Direction': {},
        'Filter': {},
        'Hash Cond': {},
        'Index Cond': {},
        'Join Filter': {}
    }
    
    # This dictionary should be built based on your dataset to map categories to integers
    # For demonstration, we'll assign arbitrary integers
    # You should replace this with a consistent encoding based on your dataset
    for i, cat in enumerate(categorical_features):
        if cat not in categorical_dict[list(categorical_dict.keys())[i]]:
            categorical_dict[list(categorical_dict.keys())[i]][cat] = len(categorical_dict[list(categorical_dict.keys())[i]])
        feature_vector.append(categorical_dict[list(categorical_dict.keys())[i]][cat])
    
    return feature_vector


if __name__ == '__main__':
    plan={
    "Plan": {
        "Node Type": "Aggregate",
        "Strategy": "Plain",
        "Partial Mode": "Finalize",
        "Parallel Aware": False,
        "Async Capable": False,
        "Startup Cost": 37988.88,
        "Total Cost": 37988.89,
        "Plan Rows": 1,
        "Plan Width": 8,
        "Output": [
        "count(*)"
        ],
        "Plans": [
        {
            "Node Type": "Gather",
            "Parent Relationship": "Outer",
            "Parallel Aware": False,
            "Async Capable": False,
            "Startup Cost": 37988.67,
            "Total Cost": 37988.88,
            "Plan Rows": 2,
            "Plan Width": 8,
            "Output": [
            "(PARTIAL count(*))"
            ],
            "Workers Planned": 2,
            "Single Copy": False,
            "Plans": [
            {
                "Node Type": "Aggregate",
                "Strategy": "Plain",
                "Partial Mode": "Partial",
                "Parent Relationship": "Outer",
                "Parallel Aware": False,
                "Async Capable": False,
                "Startup Cost": 36988.67,
                "Total Cost": 36988.68,
                "Plan Rows": 1,
                "Plan Width": 8,
                "Output": [
                "PARTIAL count(*)"
                ],
                "Plans": [
                {
                    "Node Type": "Hash Join",
                    "Parent Relationship": "Outer",
                    "Parallel Aware": False,
                    "Async Capable": False,
                    "Join Type": "Inner",
                    "Startup Cost": 992.68,
                    "Total Cost": 36987.24,
                    "Plan Rows": 572,
                    "Plan Width": 0,
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
                        "Startup Cost": 0.0,
                        "Total Cost": 35470.0,
                        "Plan Rows": 199826,
                        "Plan Width": 4,
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
                        "Filter": "((orders.o_totalprice >= 171954.30918958847) AND (orders.o_orderpriority <> '4-NOT SPECIFIED'::bpchar))"
                    },
                    {
                        "Node Type": "Hash",
                        "Parent Relationship": "Inner",
                        "Parallel Aware": False,
                        "Async Capable": False,
                        "Startup Cost": 987.32,
                        "Total Cost": 987.32,
                        "Plan Rows": 429,
                        "Plan Width": 4,
                        "Output": [
                        "customer.c_custkey"
                        ],
                        "Plans": [
                        {
                            "Node Type": "Index Scan",
                            "Parent Relationship": "Outer",
                            "Parallel Aware": False,
                            "Async Capable": False,
                            "Scan Direction": "Forward",
                            "Index Name": "customer_pkey",
                            "Relation Name": "customer",
                            "Schema": "public",
                            "Alias": "customer",
                            "Startup Cost": 0.42,
                            "Total Cost": 987.32,
                            "Plan Rows": 429,
                            "Plan Width": 4,
                            "Output": [
                            "customer.c_custkey"
                            ],
                            "Index Cond": "(customer.c_custkey >= 133044)",
                            "Filter": "((customer.c_nationkey <= 3) AND (customer.c_acctbal <= 748.9779954851595))"
                        }
                        ]
                    }
                    ]
                }
                ]
            }
            ]
        }
        ]
    },
    "peakmem": 134636,
    "time": 0.156626
    }
    import json
    with open('/home/wuy/DB/pg_mem_data/tpch_sf1/statistics_workload_combined.json') as f:
        statistics = json.load(f)

    nodes, edges = [], []
    parse_plan(plan, statistics)