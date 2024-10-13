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

from src.dataset.dataset import load_json
from src.models.GIN import GIN
from src.models.GAT import GAT
from src.models.GCN import GCN
from src.models.GraphTransformer import GraphTransformer
# from src.models.TreeTransformer import TreeTransformer
from src.models.TreeLSTM import TreeLSTM
from src.models.HeteroGraphRGCN import HeteroGraphRGCN
from src.models.HeteroGraphHGT import HeteroGraphHGT
from src.models.HeteroGraphConv import HeteroGraphConv
from src.models.HeteroGraphSAGE import HeteroGraphSAGE
# from src.models.zero_shot_models.specific_models.postgres_zero_shot import PostgresZeroShotModel
from src.training.metrics import compute_metrics
from src.dataset.dataset import QueryPlanDataset
from src.dataset.zsce_plan_dataset import ZSCEPlanDataset
from src.dataset.zsce_plan_collator import plan_collator

MODELS = {
    'GIN': GIN,
    'GAT': GAT,
    'GCN': GCN,
    'GraphTransformer': GraphTransformer,
    # 'TreeTransformer': TreeTransformer,
    'TreeLSTM': TreeLSTM,
    'HeteroGraphRGCN': HeteroGraphRGCN,
    'HeteroGraphHGT': HeteroGraphHGT,
    'HeteroGraphConv': HeteroGraphConv,
    'HeteroGraphSAGE': HeteroGraphSAGE,
    # 'zsce': PostgresZeroShotModel
}

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

def validate_model(model_name, model, val_loader, criterion, statistics, device, mem_pred, time_pred):
    # Validation
    model.eval()
    val_loss = 0
    mem_preds = []
    time_preds = []
    memories = []
    times = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val:"):
            batch = batch.to(device)
            if model_name.startswith('Hetero'):
                # Ensure that 'operator' node type exists in the batch
                if 'operator' not in batch.x_dict:
                    continue  # Skip batches without 'operator' nodes
                out_mem, out_time = model(batch.x_dict, batch.edge_index_dict, batch['operator'].batch)
            else: # homogeneous graph
                out_mem, out_time = model(batch)
            labels = batch.y.reshape(-1,2)
            mem_loss = criterion(out_mem, labels[:, 0])
            time_loss = criterion(out_time, labels[:, 1])
            loss = 0
            if mem_pred:
                loss += mem_loss
            if time_pred:
                loss += time_loss
            val_loss += loss.item()
            mem_preds.extend(out_mem.cpu())
            time_preds.extend(out_time.cpu())
            memories.extend(labels[:, 0].cpu())
            times.extend(labels[:, 1].cpu())
    avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    mem_preds = np.array(mem_preds) * statistics['peakmem']['scale'] + statistics['peakmem']['center']
    time_preds = np.array(time_preds) * statistics['time']['scale'] + statistics['time']['center']
    memories = np.array(memories) * statistics['peakmem']['scale'] + statistics['peakmem']['center']
    times = np.array(times) * statistics['time']['scale'] + statistics['time']['center']
    metrics={}
    metrics['peakmem'] = compute_metrics(memories, mem_preds)
    metrics['time'] = compute_metrics(times, time_preds)
    return avg_val_loss, metrics

def train_epoch(logger, model_name, model, optimizer, criterion, train_loader, val_loader, early_stopping, epochs, device, statistics, mem_pred, time_pred):
    logger.info("Training begins")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Train:"):
            batch = batch.to(device)
            optimizer.zero_grad()
            if model_name.startswith('Hetero'):
                # Ensure that 'operator' node type exists in the batch
                if 'operator' not in batch.x_dict:
                    continue  # Skip batches without 'operator' nodes
                out_mem, out_time = model(batch.x_dict, batch.edge_index_dict, batch['operator'].batch)
            else: # homogeneous graph
                out_mem, out_time = model(batch)
            labels = batch.y.reshape(-1,2)
            mem_loss = criterion(out_mem, labels[:, 0])
            time_loss = criterion(out_time, labels[:, 1])
            loss = 0
            if mem_pred:
                loss += mem_loss
            if time_pred:
                loss += time_loss   
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        avg_val_loss, metrics = validate_model(model_name, model, val_loader, criterion, statistics, device, mem_pred, time_pred)
        
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
def train_model(logger, args):

    dataset_dir = args.dataset_dir
    train_dataset = args.train_dataset
    test_dataset = args.test_dataset
    batch_size = args.batch_size
    num_workers = args.num_workers

    statistics_file_path = os.path.join(args.dataset_dir, args.train_dataset[0], 'statistics_workload_combined.json')  # CAUTION
    with open(statistics_file_path, 'r') as f:
        statistics = json.load(f)

    database_stats_file_path = os.path.join(args.dataset_dir, args.train_dataset[0], 'database_stats.json')  # CAUTION
    with open(database_stats_file_path) as f:
        db_statistics = json.load(f)

    if args.model == 'zsce':
        args.batch_size = 1

    # Load the dataset
    if not args.skip_train:
        if args.model == 'zsce':
            traindataset = ZSCEPlanDataset(dataset_dir, train_dataset, 'train', args.debug)
            collate_fn = functools.partial(plan_collator, feature_statistics=statistics, db_statistics=db_statistics)
            train_loader = TorchDataLoader(traindataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
            for batch in train_loader:
                print(batch)
                while 1:pass
            while 1:pass
        else:
            traindataset = QueryPlanDataset(logger, args.model, args.encode_table_column, dataset_dir, train_dataset, 'train', statistics, args.debug)
            logger.info('Train dataset size: {}'.format(len(traindataset)))
            valdataset = QueryPlanDataset(logger, args.model, args.encode_table_column, dataset_dir, train_dataset, 'val', statistics, args.debug)
            logger.info('Val dataset size: {}'.format(len(valdataset)))
        
            train_loader = DataLoader(traindataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            val_loader = DataLoader(valdataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testdataset = QueryPlanDataset(logger, args.model, args.encode_table_column, dataset_dir, test_dataset, 'test', statistics, args.debug)  

    logger.info('Test dataset size: {}'.format(len(testdataset)))
    
    test_loader = DataLoader(testdataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if args.model.startswith('Hetero'):
        # Initialize the model
        # Determine the number of unique data types for one-hot encoding
        # Assuming all graphs have the same data_type_mapping
        sample_graph = test_loader.dataset[0]
        num_operator_features = sample_graph.x_dict['operator'].shape[1]
        num_table_features = sample_graph.x_dict['table'].shape[1] if 'table' in sample_graph.x_dict else None
        num_column_features = sample_graph.x_dict['column'].shape[1] if 'column' in sample_graph.x_dict else None
        model = MODELS[args.model](
            hidden_channels=args.hidden_dim, out_channels=1, num_layers=args.num_layers, encode_table_column=args.encode_table_column, 
            num_operator_features=num_operator_features, num_table_features=num_table_features, num_column_features=num_column_features, dropout=args.dropout)
    else: 
        sample_graph = test_loader.dataset[0]
        num_node_features = sample_graph.x.shape[1]
        model = MODELS[args.model](hidden_channels=args.hidden_dim, out_channels=1, num_layers = args.num_layers, num_node_features=num_node_features, dropout=args.dropout)
    model = model.to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.L1Loss()
        
    best_model_dir = 'checkpoints'
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)
    encode_table_column_flag = '_encode_table_column' if args.encode_table_column else ''
    best_model_path = os.path.join(best_model_dir, f"{args.model}_{'_'.join(args.train_dataset)}{encode_table_column_flag}.pth")
    early_stopping = EarlyStopping(logger, args.patience, best_model_path, verbose=True)

    if not args.skip_train:
        train_epoch(logger, args.model, model, optimizer, criterion, train_loader, val_loader, early_stopping, args.epochs, args.device, statistics, 
                    args.mem_pred, args.time_pred)

    model.load_state_dict(torch.load(best_model_path))
    avg_test_loss, metrics = validate_model(args.model, model, test_loader, criterion, statistics, args.device, args.mem_pred, args.time_pred)
    logger.info(f"Test Loss={avg_test_loss:.4f}")
    if args.mem_pred:
        logger.info(f"peakmem metrics={metrics['peakmem']}")
    if args.time_pred:
        logger.info(f"time metrics={metrics['time']}")
