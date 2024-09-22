import os
import json
from torch_geometric.loader import DataLoader
from .dataset import PlanGraphDataset

def load_plans(file_path):
    """
    Load the JSON execution plans from a file.
    
    Args:
        file_path (str): Path to the JSON file containing execution plans.
        
    Returns:
        list: A list of execution plan dictionaries.
    """
    with open(file_path, 'r') as f:
        plans = json.load(f)
    return plans

def load_data(dataset_dir, train_dataset, test_dataset):
    
    train_file = os.path.join(dataset_dir, train_dataset, 'train_plans.json')
    val_file = os.path.join(dataset_dir, train_dataset, 'val_plans.json')
    test_file = os.path.join(dataset_dir, test_dataset, 'test_plans.json')

    train_plans = load_plans(train_file)
    val_plans = load_plans(val_file)
    test_plans = load_plans(test_file)
    
    return train_plans, val_plans, test_plans

def get_dataloaders(dataset_dir, train_dataset, test_dataset, batch_size=1, num_workers=0):
    train_plans, val_plans, test_plans = load_data(dataset_dir, train_dataset, test_dataset)

    train_dataset = PlanGraphDataset(train_plans)
    val_dataset = PlanGraphDataset(val_plans)
    test_dataset = PlanGraphDataset(test_plans)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    num_workers = 0
    batch_size = 1
    dataset_dir ="/home/wuy/DB/pg_mem_data"
    train_dataset = 'tpch'
    test_dataset = 'tpch'
    train_loader, val_loader, test_loader = get_dataloaders(dataset_dir, train_dataset, test_dataset, batch_size, num_workers)
    for i in val_loader.dataset[:10]:
        print(i)