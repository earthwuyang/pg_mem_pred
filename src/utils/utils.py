import json

def load_json(file_path):
    """
    Load the JSON execution plans from a file.
    
    Args:
        file_path (str): Path to the JSON file containing execution plans.
        
    Returns:
        list: A list of execution plan dictionaries.
    """
    file_path = '/home/wuy/DB/pg_mem_data/tpch/tiny_plans.json'   # for debug
    with open(file_path, 'r') as f:
        plans = json.load(f)
    return plans