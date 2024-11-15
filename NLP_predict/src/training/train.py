import torch
from tqdm import tqdm
from torch.optim import Adam
import os
import pickle
import numpy as np
import json
from sklearn.preprocessing import RobustScaler, MinMaxScaler, FunctionTransformer
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
import functools
from time import time
import logging
from src.dataset.dataset import load_json
from src.models.GIN import GIN
from src.models.GAT import GAT
from src.models.GCN import GCN
from src.models.GraphTransformer import GraphTransformer
# from src.models.TreeTransformer import TreeTransformer
from src.models.model import Model
# from src.models.zero_shot_models.specific_models.postgres_zero_shot import PostgresZeroShotModel
from src.training.metrics import compute_metrics
from src.dataset.dataset import QueryPlanDataset
# from src.dataset.zsce_plan_dataset import ZSCEPlanDataset
# from src.dataset.zsce_plan_collator import plan_collator


# Define the early stopping class (redefined for clarity)
class EarlyStopping:
    def __init__(self, logger, patience, best_model_path, verbose=False):
        self.logger = logger
        self.best_model_path = best_model_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                self.logger.info(f'Validation loss decreased to {val_loss:.4f}. Resetting patience.')
            # save model
            torch.save(model.state_dict(), self.best_model_path)
        else:
            self.counter += 1
            if self.verbose:
                self.logger.info(f'Validation loss did not decrease. Patience counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

def validate_model(model_name, model, val_loader, criterion, statistics, device, mem_pred, time_pred, dataset):
    # Validation
    model.eval()
    val_loss = 0
    preds = []
    labels = []
    torch.cuda.synchronize()
    start_time = time()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val:"):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            output = model(x).squeeze()
            loss = criterion(output.squeeze(), y)
            val_loss += loss.item()
            mem_loss = criterion(output, y)
           
            val_loss += loss.item()
            preds.append(output.cpu().numpy())
            labels.append(y.cpu().numpy())
    avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    preds = dataset.scaler.inverse_transform(preds.reshape(-1, 1)).reshape(-1)
    labels = dataset.scaler.inverse_transform(labels.reshape(-1, 1)).reshape(-1)
    metrics={}
    metrics['peakmem'] = compute_metrics(labels, preds)
    return avg_val_loss, metrics

def train_epoch(logger, model_name, model, optimizer, criterion, train_loader, val_loader, early_stopping, epochs, device, statistics, mem_pred, time_pred, dataset):
    logger.info("Training begins")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Train:"):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            optimizer.zero_grad()
            loss = criterion(output.squeeze(), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        avg_val_loss, metrics = validate_model(model_name, model, val_loader, criterion, statistics, device, mem_pred, time_pred, dataset)
        
        logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        if mem_pred:
            logger.info(f"peakmem metrics={metrics['peakmem']}")
        if time_pred:
            logger.info(f"time metrics={metrics['time']}")

        # Early stopping
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    logger.info("Training ends")

# Define training function (redefined for clarity)
def train_model(logger, args, statistics):

    dataset_dir = args.data_dir
    train_dataset = args.train_dataset[0]
    val_dataset = args.val_dataset
    test_dataset = args.test_dataset
    batch_size = args.batch_size
    num_workers = args.num_workers

    with open(args.db_config) as f:
        conn_info = json.load(f)


    if args.model == 'zsce':
        args.batch_size = 1


    train_dataset = QueryPlanDataset(args.dataset[0])

    # train test split
    from sklearn.model_selection import train_test_split
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    # test_dataset = QueryPlanDataset(args.test_dataset, train_dataset.scaler)
    # val_dataset = test_dataset
    val_dataset = train_dataset
    test_dataset = train_dataset
    logger.info(f"train_dataset: {type(train_dataset)}")
    batch_size = 128
    num_workers = 10
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = Model(768)
    
    model = model.to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.L1Loss()
        
    # encode_table_column_flag = '_encode_table_column' if args.encode_table_column else ''
    # best_model_path = os.path.join(best_model_dir, f"{args.model}_{'_'.join(args.train_dataset)}{encode_table_column_flag}_{'mem' if args.mem_pred else ''}_{'time' if args.time_pred else ''}_best.pth")
    best_model_path = 'checkpoint.pt'
    early_stopping = EarlyStopping(logger, args.patience, best_model_path, verbose=True)

    if not args.skip_train:
        begin = time()
        train_epoch(logger, args.model, model, optimizer, criterion, train_loader, val_loader, early_stopping, args.epochs, args.device, statistics, 
                    args.mem_pred, args.time_pred, train_dataset)
        time_elapse = time()-begin
        print(f"training takes {time_elapse} seconds, equivalent to {time_elapse/3600:.4f} hours")

    model.load_state_dict(torch.load(best_model_path))
    avg_test_loss, metrics = validate_model(args.model, model, test_loader, criterion, statistics, args.device, args.mem_pred, args.time_pred, train_dataset)
    logger.info(f"Test Loss={avg_test_loss:.4f}")
    if args.mem_pred:
        logger.info(f"peakmem metrics={metrics['peakmem']}")
    if args.time_pred:
        logger.info(f"time metrics={metrics['time']}")
