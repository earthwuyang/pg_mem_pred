import json
import torch
import os
from torch_geometric.data import Data, Dataset, DataLoader
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GINConv,
    GATConv,
    TransformerConv,
    global_add_pool,
    global_mean_pool,
)
from sklearn.preprocessing import RobustScaler
import networkx as nx
import numpy as np
from collections import defaultdict
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from tqdm import tqdm

from metrics import compute_metrics

# ----------------------------
# Data Preprocessing Components
# ----------------------------

class PlanGraphDataset(Dataset):
    def __init__(self, json_plans, categorical_maps, numerical_scaler, transform=None, pre_transform=None):
        super(PlanGraphDataset, self).__init__(None, transform, pre_transform)
        self.json_plans = json_plans
        self.categorical_maps = categorical_maps
        self.numerical_scaler = numerical_scaler

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
        
        # Apply scaling to numerical features
        x = self.numerical_scaler.transform(x.numpy())
        x = torch.tensor(x, dtype=torch.float)
        
        # Get additional statistical features
        histogram = compute_histogram(x).astype(np.float32)
        sample_bits = compute_sample_bits(x).astype(np.float32)
        
        # Concatenate additional features
        x = torch.cat([x, torch.tensor(histogram), torch.tensor(sample_bits)], dim=1)
        
        # Get the label (peakmem)
        y = torch.tensor(plan.get('peakmem', 0.0), dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        return data

def parse_plan(plan, parent=None, nodes=None, edges=None, node_id=0, categorical_maps=None):
    """
    Recursively parse the JSON plan and build node features and edges.
    
    Args:
        plan (dict): The JSON plan.
        parent (int): The parent node ID.
        nodes (list): List to store node features.
        edges (list): List to store edge connections.
        node_id (int): Unique identifier for nodes.
        categorical_maps (dict): Predefined mappings for categorical features.
        
    Returns:
        node_id (int): Updated node ID.
    """
    if nodes is None:
        nodes = []
    if edges is None:
        edges = []
    
    current_id = node_id
    
    if 'Plan' in plan:
        plan_node = plan['Plan']
    else: 
        plan_node = plan
    
    # Extract features from the current node
    features = extract_features(plan_node, categorical_maps)
    nodes.append(features)
    
    # If there is a parent, add an edge
    if parent is not None:
        edges.append([parent, current_id])
    
    # Initialize children if not present
    children = plan_node.get('Plans', [])
    
    # Recursively parse children
    for child in children:
        node_id += 1
        node_id = parse_plan(child, parent=current_id, nodes=nodes, edges=edges, node_id=node_id, categorical_maps=categorical_maps)
    
    return node_id

def extract_features(plan_node, categorical_maps):
    """
    Extract relevant features from a plan node.
    
    Args:
        plan_node (dict): A single plan node from the JSON.
        categorical_maps (dict): Predefined mappings for categorical features.
        
    Returns:
        feature_vector (list): A list of numerical and encoded categorical features.
    """
    # Define which features to extract
    numerical_features = [
        plan_node.get('Startup Cost', 0.0),
        plan_node.get('Total Cost', 0.0),
        plan_node.get('Plan Rows', 0.0),
        plan_node.get('Plan Width', 0.0),
        plan_node.get('Workers Planned', 0.0)
    ]
    
    # Categorical features
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
    
    # One-Hot Encode categorical features
    one_hot_features = []
    for i, cat in enumerate(categorical_features):
        feature_name = list(categorical_maps.keys())[i]
        mapping = categorical_maps[feature_name]
        encoded = torch.nn.functional.one_hot(torch.tensor(mapping.get(cat, 0)), num_classes=len(mapping)).float()
        one_hot_features.append(encoded)
    
    if one_hot_features:
        one_hot_features = torch.cat(one_hot_features).numpy().tolist()
    else:
        one_hot_features = []
    
    # Combine numerical and categorical features
    feature_vector = numerical_features + one_hot_features
    
    return feature_vector

def compute_histogram(x, bins=10):
    """
    Compute histogram features for each node's numerical features.
    
    Args:
        x (np.ndarray): Node feature matrix.
        bins (int): Number of histogram bins.
        
    Returns:
        histogram_features (np.ndarray): Histogram features.
    """
    histogram_features = []
    for feature_idx in range(x.shape[1]):
        hist, _ = np.histogram(x[:, feature_idx], bins=bins, range=(np.min(x[:, feature_idx]), np.max(x[:, feature_idx])))
        histogram_features.extend(hist)
    return np.array(histogram_features)

def compute_sample_bits(x, bits=8):
    """
    Compute sample bits features (e.g., quantization) for each node's numerical features.
    
    Args:
        x (np.ndarray): Node feature matrix.
        bits (int): Number of bits for quantization.
        
    Returns:
        sample_bits_features (np.ndarray): Sample bits features.
    """
    # Example: Quantize each numerical feature into discrete bins based on bits
    sample_bits_features = []
    for feature in x.T:
        quantized = np.floor(feature / (np.max(feature) - np.min(feature) + 1e-5) * (2 ** bits - 1))
        sample_bits_features.extend(quantized.astype(int))
    return np.array(sample_bits_features)

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

def build_categorical_maps(json_plans):
    """
    Build categorical mappings for all categorical features based on the dataset.
    
    Args:
        json_plans (list): List of execution plan dictionaries.
        
    Returns:
        dict: A dictionary mapping feature names to category-to-integer mappings.
    """
    categorical_features = [
        'Node Type',
        'Join Type',
        'Strategy',
        'Partial Mode',
        'Parent Relationship',
        'Scan Direction',
        'Filter',
        'Hash Cond',
        'Index Cond',
        'Join Filter'
    ]
    
    categorical_maps = {feature: {} for feature in categorical_features}
    
    for plan in json_plans:
        stack = [plan]
        while stack:
            current = stack.pop()
            if 'Plan' in current:
                node = current['Plan']
            else:
                node = current
            for i, feature in enumerate(categorical_features):
                value = node.get(feature, '')
                if value not in categorical_maps[feature]:
                    categorical_maps[feature][value] = len(categorical_maps[feature])
            stack.extend(node.get('Plans', []))
    
    return categorical_maps

# ----------------------------
# Model Definitions
# ----------------------------

class GINModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_layers=2):
        super(GINModel, self).__init__()
        torch.manual_seed(42)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            nn = torch.nn.Sequential(
                torch.nn.Linear(num_node_features if i == 0 else hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU()
            )
            self.convs.append(GINConv(nn))
        self.fc = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x.squeeze()

class GATModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, heads=4, num_layers=2):
        super(GATModel, self).__init__()
        torch.manual_seed(42)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                GATConv(
                    in_channels=num_node_features if i == 0 else hidden_channels * heads,
                    out_channels=hidden_channels,
                    heads=heads,
                    concat=True
                )
            )
        self.fc = torch.nn.Linear(hidden_channels * heads, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x.squeeze()

class GraphTransformerModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, heads=4, num_layers=2):
        super(GraphTransformerModel, self).__init__()
        torch.manual_seed(42)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                TransformerConv(
                    in_channels=num_node_features if i == 0 else hidden_channels,
                    out_channels=hidden_channels // heads,
                    heads=heads
                )
            )
        self.fc = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x.squeeze()

# ----------------------------
# Metrics
# ----------------------------

# def q_error(y_true, y_pred):
#     """
#     Calculate the q-error metric.
    
#     Args:
#         y_true (np.ndarray): True target values.
#         y_pred (np.ndarray): Predicted target values.
        
#     Returns:
#         float: The average q-error.
#     """
#     epsilon = 1e-6  # To prevent division by zero
#     q_errors = np.maximum(y_pred / (y_true + epsilon), (y_true + epsilon) / (y_pred + epsilon))
#     return np.mean(q_errors)

# def compute_metrics(y_true, y_pred):
#     """
#     Compute evaluation metrics.
    
#     Args:
#         y_true (np.ndarray): True target values.
#         y_pred (np.ndarray): Predicted target values.
        
#     Returns:
#         dict: Dictionary containing metrics.
#     """
#     q_err = q_error(y_true, y_pred)
#     mse = np.mean((y_true - y_pred) ** 2)
#     return {'q_error': q_err, 'mse': mse}

# ----------------------------
# Training and Evaluation
# ----------------------------

def train_model(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0
    for data in tqdm(train_loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def evaluate_model(model, criterion, loader, device):
    model.eval()
    total_loss = 0
    all_y_true = []
    all_y_pred = []
    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating"):
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs
            all_y_true.append(data.y.cpu().numpy())
            all_y_pred.append(out.cpu().numpy())
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    metrics = compute_metrics(y_true, y_pred)
    return total_loss / len(loader.dataset), metrics

# ----------------------------
# Hyperparameter Tuning with Optuna
# ----------------------------

def objective(trial, train_loader, valid_loader, num_node_features, device):
    # Suggest hyperparameters
    model_type = trial.suggest_categorical('model_type', ['GIN', 'GAT', 'GraphTransformer'])
    hidden_channels = trial.suggest_int('hidden_channels', 32, 128)
    num_layers = trial.suggest_int('num_layers', 2, 4)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
    heads = trial.suggest_int('heads', 2, 8) if model_type in ['GAT', 'GraphTransformer'] else None
    
    # Initialize model
    if model_type == 'GIN':
        model = GINModel(num_node_features, hidden_channels, num_layers=num_layers)
    elif model_type == 'GAT':
        model = GATModel(num_node_features, hidden_channels, heads=heads, num_layers=num_layers)
    elif model_type == 'GraphTransformer':
        model = GraphTransformerModel(num_node_features, hidden_channels, heads=heads, num_layers=num_layers)
    else:
        raise ValueError("Unknown model type")
    
    model = model.to(device)
    
    # Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    epochs = 50
    for epoch in range(epochs):
        train_loss = train_model(model, optimizer, criterion, train_loader, device)
        val_loss, metrics = evaluate_model(model, criterion, valid_loader, device)
        
        # Report intermediate objective value
        trial.report(metrics['q_error'], epoch)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return metrics['q_error']

def perform_hyperparameter_tuning(train_loader, valid_loader, num_node_features, device):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_loader, valid_loader, num_node_features, device), n_trials=50)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Q-Error: {trial.value}")
    print("  Best hyperparameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    return study.best_trial.params

# ----------------------------
# Main Execution
# ----------------------------

def main():
    # Load plans
    json_plans = load_plans('../tpch_data/explain_json_plans.json')  # Replace with your actual file path
    print(f"Number of plans: {len(json_plans)}")
    
    # Build categorical mappings
    categorical_maps = build_categorical_maps(json_plans)
    
    # Initialize numerical scaler
    # To fit the scaler, we need to extract all numerical features first
    numerical_features = []
    numerical_features_file = 'numerical_features.npy'
    if os.path.exists(numerical_features_file):
        numerical_features = np.load(numerical_features_file)
    else:
        print(f"Parsing plans...")
        for plan in tqdm(json_plans):
            nodes = []
            parse_plan(plan, nodes=nodes, edges=[], categorical_maps=categorical_maps)
            numerical_features.append(nodes)
        numerical_features = np.concatenate(numerical_features, axis=0)
        # save numerical_features to file
        np.save('numerical_features.npy', numerical_features)
    
    scaler = RobustScaler()
    scaler.fit(numerical_features)
    
    # Create the dataset
    dataset = PlanGraphDataset(json_plans, categorical_maps, scaler)
    print(f"split dataset")
    # Split into training, validation, and testing sets (e.g., 70% train, 15% val, 15% test)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    
    # Define DataLoaders
    batch_size = 128  # Adjust based on your system's memory
    num_workers = 4    # Adjust based on your CPU cores
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameter Tuning
    num_node_features = dataset[0].x.shape[1]
    best_params = perform_hyperparameter_tuning(train_loader, val_loader, num_node_features, device)
    
    # Initialize the best model with the optimal hyperparameters
    model_type = best_params['model_type']
    hidden_channels = best_params['hidden_channels']
    num_layers = best_params['num_layers']
    learning_rate = best_params['learning_rate']
    weight_decay = best_params['weight_decay']
    heads = best_params.get('heads', None)
    
    if model_type == 'GIN':
        model = GINModel(num_node_features, hidden_channels, num_layers=num_layers)
    elif model_type == 'GAT':
        model = GATModel(num_node_features, hidden_channels, heads=heads, num_layers=num_layers)
    elif model_type == 'GraphTransformer':
        model = GraphTransformerModel(num_node_features, hidden_channels, heads=heads, num_layers=num_layers)
    else:
        raise ValueError("Unknown model type")
    
    model = model.to(device)
    
    # Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    epochs = 100
    best_val_q_error = float('inf')
    for epoch in range(1, epochs + 1):
        train_loss = train_model(model, optimizer, criterion, train_loader, device)
        val_loss, val_metrics = evaluate_model(model, criterion, val_loader, device)
        scheduler.step(val_loss)
        
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Q-Error: {val_metrics["q_error"]:.4f}')
        
        # Save the best model
        if val_metrics['q_error'] < best_val_q_error:
            best_val_q_error = val_metrics['q_error']
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Best model saved with Q-Error: {best_val_q_error:.4f}")
    
    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate on the test set
    test_loss, test_metrics = evaluate_model(model, criterion, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test Q-Error: {test_metrics['q_error']:.4f}")

if __name__ == "__main__":
    main()
