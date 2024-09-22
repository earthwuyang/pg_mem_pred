def parse_plan(plan, parent=None, nodes=None, edges=None, node_id=0):
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
    features = extract_features(plan_node)
    nodes.append(features)
    
    # If there is a parent, add an edge
    if parent is not None:
        edges.append([parent, current_id])
    
    # Initialize children if not present
    children = plan_node.get('Plans', [])
    
    # Recursively parse children
    for child in children:
        node_id += 1
        node_id = parse_plan(child, parent=current_id, nodes=nodes, edges=edges, node_id=node_id)
    
    return node_id

def extract_features(plan_node):
    """
    Extract relevant features from a plan node.
    
    Args:
        plan_node (dict): A single plan node from the JSON.
        
    Returns:
        feature_vector (list): A list of numerical features.
    """
    # Define which features to extract
    feature_vector = []
    
    # Numerical features
    numerical_features = [
        plan_node.get('Startup Cost', 0.0),
        plan_node.get('Total Cost', 0.0),
        plan_node.get('Plan Rows', 0.0),
        plan_node.get('Plan Width', 0.0),
        plan_node.get('Workers Planned', 0.0)
    ]
    feature_vector.extend(numerical_features)
    
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