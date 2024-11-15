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
    time_preds = []
    mem_preds = []
    time_labels = []
    mem_labels = []
    with torch.no_grad():
        for data in tqdm(loader, desc="Validation"):
            data = data.to(device)
            output = model(data).squeeze()
            time_pred = output[:, 0].cpu()
            mem_pred = output[:, 1:].cpu().flatten()
            labels = torch.tensor(data.y).to(device)
            loss = criterion(output, labels)
            total_loss += loss.item()
            time_label = labels[:, 0]
            mem_label = labels[:, 1:].flatten()
            time_preds.extend(time_pred)
            mem_preds.extend(mem_pred)
            time_labels.extend(time_label)
            mem_labels.extend(mem_label)
    avg_loss = total_loss / len(loader)
    time_preds = torch.tensor(time_preds)
    mem_preds = torch.tensor(mem_preds)
    time_labels = torch.tensor(time_labels)
    mem_labels = torch.tensor(mem_labels)
    time_preds = time_preds * statistics['time']['scale'] + statistics['time']['center']
    mem_preds = mem_preds * statistics['mem']['scale'] + statistics['mem']['center']
    time_labels = time_labels * statistics['time']['scale'] + statistics['time']['center']
    mem_labels = mem_labels * statistics['mem']['scale'] + statistics['mem']['center']
    time_metrics = compute_metrics(time_labels, time_preds)
    mem_metrics = compute_metrics(mem_labels.flatten(), mem_preds.flatten())
    metrics = {'time': time_metrics,'mem': mem_metrics}
    return avg_loss, metrics

# Testing Function
def test(model, dataset, loader, device, statistics):
    model.eval()
    time_preds = []
    mem_preds = []
    time_labels = []
    mem_labels = []
    with torch.no_grad():
        for data in tqdm(loader, desc="Testing"):
            data = data.to(device)
            output = model(data).squeeze()
            time_pred = output[:, 0].cpu()
            mem_pred = output[:, 1:].cpu().flatten()
            labels = torch.tensor(data.y)
            time_label = labels[:, 0]
            mem_label = labels[:, 1:].flatten()
            time_preds.extend(time_pred)
            mem_preds.extend(mem_pred)
            time_labels.extend(time_label)
            mem_labels.extend(mem_label)
    time_preds = np.array(time_preds)
    mem_preds = np.array(mem_preds)
    time_labels = np.array(time_labels)
    mem_labels = np.array(mem_labels)
    time_preds = time_preds * statistics['time']['scale'] + statistics['time']['center']
    mem_preds = mem_preds * statistics['mem']['scale'] + statistics['mem']['center']
    time_labels = time_labels * statistics['time']['scale'] + statistics['time']['center']
    mem_labels = mem_labels * statistics['mem']['scale'] + statistics['mem']['center']
    time_metrics = compute_metrics(time_labels, time_preds)
    mem_metrics = compute_metrics(mem_labels.flatten(), mem_preds.flatten())
    metrics = {'time': time_metrics,'mem': mem_metrics}
    return metrics

# Training Function
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data).squeeze()
        labels = torch.tensor(data.y).to(device)
        loss = criterion(output, labels)
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
output_dim = 1 + 10 # time, 10*mem
max_epoch = 1000
patience = 20
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
train_dataset='tpch'


test_dataset = 'tpch'


model = GraphGINModel(num_node_features, hidden_dim, output_dim)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

with open('statistics.json', 'r') as f:
    statistics = json.load(f)

# Create Dataset and DataLoader
dataset = PlanDataset(statistics, train_dataset)

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
test_metrics = test(model, dataset, test_loader, device, statistics)

print(f"Test metrics: {test_metrics}")
