import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np



def load_json(json_file):
    # json_file = '/home/wuy/DB/pg_mem_data/tpch_sf1/tiny_plans.json' # for debug
    with open(json_file, 'r') as f:
        query_plans = json.load(f)
    return query_plans

    
class ZSCEPlanDataset(Dataset):
    def __init__(self, dataset_dir, dataset, mode, debug):
        super(ZSCEPlanDataset, self).__init__()
        self.dataset_list = []
        for ds in dataset:
            json_file_path = os.path.join(dataset_dir, ds, f'{mode}_plans.json')
            if debug:
                json_file_path = os.path.join(dataset_dir, ds, f'tiny_plans.json')
            plans = load_json(json_file_path)
            self.dataset_list.extend(plans)
    
    def __len__(self):
        return len(self.dataset_list)
    
    def __getitem__(self, idx):
        return self.dataset_list[idx]



