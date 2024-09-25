from torch_geometric.data import Dataset, Data
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import torch
import json
import numpy as np

from .parse_plan import parse_plan

class PlanGraphDataset(Dataset):
    def __init__(self, json_plans, statistics, transform=None, pre_transform=None):
        super(PlanGraphDataset, self).__init__(None, transform, pre_transform)
        self.json_plans = json_plans
        self.scaler = StandardScaler()
        self.statistics = statistics
        # self.mem_scaler = FunctionTransformer(lambda x: (x-self.statistics['peakmem']['center']) / self.statistics['peakmem']['scale'], validate=True)
        self.mem_scaler = FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True)

    def len(self):
        return len(self.json_plans)

    def get(self, idx):
        plan = self.json_plans[idx]
        
        # Parse the plan into nodes and edges
        nodes = []
        edges = []
        parse_plan(plan, self.statistics, nodes=nodes, edges=edges)
        
        # Convert lists to tensors
        x = torch.tensor(nodes, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # # Apply scaling to features
        # x = torch.tensor(self.scaler.fit_transform(x.numpy()), dtype=torch.float)
        
        
        # Get the label (peakmem)
        peakmem = plan.get('peakmem', 0.0)
        # scaled_peakmem = (peakmem- self.statistics['peakmem']['center']) / self.statistics['peakmem']['scale']
        # y = torch.tensor(scaled_peakmem, dtype=torch.float)

        log_peakmem = self.mem_scaler.transform(np.array([peakmem]).reshape(-1, 1)).reshape(-1)
        y = torch.tensor(log_peakmem, dtype=torch.float)

        
        data = Data(x=x, edge_index=edge_index, y=y)
        return data
    
