import re
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader  # Updated import
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.data import Batch
import numpy as np
import json
from metrics import compute_metrics

from dataloader import PlanDataset
from tqdm import tqdm
from model import GraphGINModel
from torch.utils.data import random_split


# Validation Function
def validate(model, dataset, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_actuals = []
    with torch.no_grad():
        for data in tqdm(loader, desc="Validation"):
            data = data.to(device)
            output = model(data).squeeze()
            loss = criterion(output, data.y.squeeze())
            total_loss += loss.item()
            all_predictions.extend(output.cpu().numpy())
            all_actuals.extend(data.y.squeeze().cpu().numpy())
    avg_loss = total_loss / len(loader)
    all_actuals = dataset.scaler.inverse_transform(np.array(all_actuals).reshape(-1, 1)).flatten()
    all_predictions = dataset.scaler.inverse_transform(np.array(all_predictions).reshape(-1, 1)).flatten()
    metrics = compute_metrics(all_actuals, all_predictions)
    return avg_loss, metrics

# Testing Function
def test(model, dataset, loader, device):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for data in tqdm(loader, desc="Testing"):
            data = data.to(device)
            output = model(data).squeeze()
            predictions.extend(output.cpu().numpy())
            actuals.extend(data.y.squeeze().cpu().numpy())
        actuals = dataset.scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
        predictions = dataset.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        metrics = compute_metrics(actuals, predictions)
    return metrics

# Training Function
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data).squeeze()
        loss = criterion(output, data.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss

# Training Loop with Early Stopping and Model Saving
def train(model, dataset, train_loader, val_loader, optimizer, criterion, epochs=100, patience=20, save_path='best_model.pth'):
    best_median_qe = float('inf')
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(epochs):
        # Training Phase
        avg_train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation Phase
        val_loss, val_metrics = validate(model, dataset, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Metrics: {val_metrics}")
        
        # check for improvement in validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), save_path)
            print(f"Validation loss improved to {best_val_loss:.4f}. Model saved.")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} consecutive epoch(s).")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                early_stop = True
                break  # Exit the training loop
        # # Check for improvement in median Q-error
        # current_median_qe = val_metrics.get('qerror_50 (Median)', float('inf'))
        # if current_median_qe < best_median_qe:
        #     best_median_qe = current_median_qe
        #     epochs_no_improve = 0
        #     # Save the best model
        #     torch.save(model.state_dict(), save_path)
        #     print(f"Validation median Q-error improved to {best_median_qe:.4f}. Model saved.")
        # else:
        #     epochs_no_improve += 1
        #     print(f"No improvement in median Q-error for {epochs_no_improve} consecutive epoch(s).")
        #     if epochs_no_improve >= patience:
        #         print("Early stopping triggered.")
        #         early_stop = True
        #         break  # Exit the training loop
        
    if not early_stop:
        print(f"Training completed after {epochs} epochs.")
    else:
        print(f"Early stopping after {epoch+1} epochs.")
    
    # Load the best model before testing
    model.load_state_dict(torch.load(save_path))
    print("Best model loaded for testing.")


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--skip_train', action='store_true', help='Skip training and only test the model')
args = parser.parse_args()
# Initialize Model, Optimizer, and Loss Function
num_node_features = 4  # [Node Type, Log(Cardinality), Num of Columns, Limit]
hidden_dim = 32
max_epoch = 1000
patience = 20
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
train_dataset='tpcds'


test_dataset = 'tpcds'


model = GraphGINModel(num_node_features, hidden_dim)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


# Create Dataset and DataLoader
dataset = PlanDataset(train_dataset)

import pickle
# save dataset.scaler
with open(f'{train_dataset}_scaler.pkl', 'wb') as f:
    pickle.dump(dataset.scaler, f)

print(f"Scaler saved to {train_dataset}_scaler.pkl")

# Define split sizes
total_size = len(dataset)
train_size = int(0.9 * total_size)
val_size = int(0.05 * total_size)
test_size = total_size - train_size - val_size

# Perform the split
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(seed)  # For reproducibility
)

print(f"Total samples: {total_size}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Create DataLoaders
batch_size = 6000  # Adjust batch size as needed (60000 is typically too large)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Train the Model with Early Stopping and Model Saving
train(model, dataset, train_loader, val_loader, optimizer, criterion, epochs=max_epoch, patience=patience, save_path='best_model.pth')

# Test the model on the test set
test_metrics = test(model, dataset, test_loader, device)

print(f"Test metrics: {test_metrics}")
