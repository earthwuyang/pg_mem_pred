from torch_geometric.data import Dataset, Data
from sklearn.preprocessing import StandardScaler
import torch
import json

from .parse_plan import parse_plan

class PlanGraphDataset(Dataset):
    def __init__(self, json_plans, transform=None, pre_transform=None):
        super(PlanGraphDataset, self).__init__(None, transform, pre_transform)
        self.json_plans = json_plans
        self.scaler = StandardScaler()

    def len(self):
        return len(self.json_plans)

    def get(self, idx):
        plan = self.json_plans[idx]
        
        # Parse the plan into nodes and edges
        nodes = []
        edges = []
        parse_plan(plan, nodes=nodes, edges=edges)
        
        # Convert lists to tensors
        x = torch.tensor(nodes, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Apply scaling to features
        x = torch.tensor(self.scaler.fit_transform(x.numpy()), dtype=torch.float)
        
        # Get the label (peakmem)
        y = torch.tensor(plan.get('peakmem', 0.0), dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        return data
    
