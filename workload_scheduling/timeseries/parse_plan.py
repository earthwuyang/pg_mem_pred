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
    for key in ['Startup Cost', 'Total Cost', 'Plan Rows', 'Plan Width', 'Node Type']:
        if statistics[key]['type'] == 'numeric':
            value = ( plan_node[key] - statistics[key]['center']) / statistics[key]['scale']
            feature_vector.append(value)
        elif statistics[key]['type'] == 'categorical':
            value = plan_node.get(key, 'unknown')
            one_hot_features = one_hot_encode(statistics[key]['value_dict'].get(value, statistics[key]['no_vals']), statistics[key]['no_vals']+1)  # unknown will map to an extra number in the directory
            feature_vector.extend(one_hot_features)
    return feature_vector
