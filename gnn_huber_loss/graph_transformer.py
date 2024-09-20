import json
import torch
from torch_geometric.data import Data, Dataset, DataLoader
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GCNConv, global_add_pool, global_mean_pool
from sklearn.preprocessing import StandardScaler
import networkx as nx
from tqdm import tqdm
from metrics import compute_metrics

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




# Example usage:
# Assuming 'plans.json' contains a list of execution plans like the one provided
json_plans = load_plans('../tpch_data/explain_json_plans.json')  # Replace with your actual file path
print(f"num plans: {len(json_plans)}")

# Create the dataset
dataset = PlanGraphDataset(json_plans)

# Split into training and testing sets (e.g., 80% train, 20% test)
train_size = int(0.5 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])



batch_size = 10240
# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=10)


class GrapnTransformer(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, heads=4):
        super(GrapnTransformer, self).__init__()
        torch.manual_seed(42)
        
        # TransformerConv with multi-head attention
        self.conv1 = TransformerConv(num_node_features, hidden_channels // heads, heads=heads)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels // heads, heads=heads)
        self.fc = torch.nn.Linear(hidden_channels, 1)  # Regression output

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First TransformerConv layer with multi-head attention
        x = self.conv1(x, edge_index)
        x = x.relu()

        # Second TransformerConv layer with multi-head attention
        x = self.conv2(x, edge_index)
        x = x.relu()

        # Global mean pooling
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # Final linear layer
        x = self.fc(x)

        return x.squeeze()  # [batch_size]


# Initialize the model, loss, and optimizer
num_node_features = len(dataset[0].x[0])  # Number of features per node
hidden_channels = 64
# model = TransformerConv(num_node_features, hidden_channels)
model = GrapnTransformer(num_node_features, hidden_channels)
print(model)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)


class HuberLoss(torch.nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, pred, target):
        error = torch.abs(pred - target)
        cond = error < self.delta
        squared_loss = 0.5 * error ** 2
        linear_loss = self.delta * (error - 0.5 * self.delta)
        return torch.where(cond, squared_loss, linear_loss).mean()
def train():
    model.train()
    total_loss = 0
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def test(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            metrics = compute_metrics(data.y.cpu().numpy(), out.detach().cpu().numpy())
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset), metrics

# Training loop
epochs = 100
for epoch in range(1, epochs + 1):
    loss = train()
    train_loss = loss
    test_loss, metrics = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    print(f"Test Metrics: {metrics}")
    scheduler.step(test_loss)
    # Save model
    torch.save(model.state_dict(), f'model_graph_transformer.pth')


