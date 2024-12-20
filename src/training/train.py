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
from src.models.TreeLSTM import TreeLSTM
from src.models.HeteroGraphRGCN import HeteroGraphRGCN
from src.models.HeteroGraphHGT import HeteroGraphHGT
from src.models.HeteroGraphConv import HeteroGraphConv
from src.models.HeteroGraphSAGE import HeteroGraphSAGE
# from src.models.zero_shot_models.specific_models.postgres_zero_shot import PostgresZeroShotModel
from src.training.metrics import compute_metrics
from src.dataset.dataset import QueryPlanDataset
# from src.dataset.zsce_plan_dataset import ZSCEPlanDataset
# from src.dataset.zsce_plan_collator import plan_collator

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

def heuristic_peak_memory(estimated_rows, row_width, scaling_factor=1.0):
    """
    Estimate peak memory using a heuristic based on estimated rows and row width.

    Args:
        estimated_rows (torch.Tensor or np.ndarray): Estimated number of rows.
        row_width (torch.Tensor or np.ndarray): Width of each row in bytes.
        scaling_factor (float): Scaling factor to adjust the heuristic.

    Returns:
        torch.Tensor or np.ndarray: Estimated peak memory.
    """
    return estimated_rows * row_width * scaling_factor


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

def validate_model(model_name, model, val_loader, criterion, statistics, device, mem_pred, time_pred, scaling_factor=1.0):
    # Validation
    model.eval()
    val_loss = 0
    mem_preds = []
    time_preds = []
    heuristic_mem_preds = []
    memories = []
    times = []
    torch.cuda.synchronize()
    start_time = time()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val:"):
            batch = batch.to(device)
            heuristic_mem = heuristic_peak_memory(batch.plan_rows, batch.plan_width, scaling_factor=scaling_factor)
            heuristic_mem_preds.extend(heuristic_mem.cpu())

            if model_name.startswith('Hetero'):
                # Ensure that 'operator' node type exists in the batch
                if 'operator' not in batch.x_dict:
                    continue  # Skip batches without 'operator' nodes
                out_mem, out_time = model(batch.x_dict, batch.edge_index_dict, batch['operator'].batch)
            else: # homogeneous graph
                out_mem, out_time = model(batch)
            end_time = time()
            total_time = end_time - start_time
            # logging.info(f"Time taken for one batch: {total_time}")
            # while 1:pass

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

    # Compute metrics for heuristic
    heuristic_metrics = {}
    heuristic_metrics['peakmem'] = compute_metrics(memories, heuristic_mem_preds)
    metrics['heuristic_peakmem'] = heuristic_metrics['peakmem']
    

    return avg_val_loss, metrics

def train_epoch(logger, model_name, model, optimizer, criterion, train_loader, val_loader, early_stopping, epochs, device, statistics, mem_pred, time_pred, scaling_factor=1.0):
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
            # logger.debug(f"mem_loss={mem_loss.item()}, time_loss={time_loss.item()}")
            loss = 0
            if mem_pred:
                loss += mem_loss
            if time_pred:
                loss += time_loss   
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        avg_val_loss, metrics = validate_model(model_name, model, val_loader, criterion, statistics, device, mem_pred, time_pred, scaling_factor)
        
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
    train_dataset = args.train_dataset
    val_dataset = args.val_dataset
    test_dataset = args.test_dataset
    batch_size = args.batch_size
    num_workers = args.num_workers
    scaling_factor = args.scaling_factor

    with open(args.db_config) as f:
        conn_info = json.load(f)


    if args.model == 'zsce':
        args.batch_size = 1

    not_cross_dataset = len(args.train_dataset) ==1

    # Load the dataset
    if not args.skip_train:
        traindataset = QueryPlanDataset(logger, args.model, args.encode_table_column, dataset_dir, train_dataset, 'train', statistics, args.debug, conn_info, not_cross_dataset)
        logger.info('Train dataset size: {}'.format(len(traindataset)))

        valdataset = QueryPlanDataset(logger, args.model, args.encode_table_column, dataset_dir, val_dataset, 'val', statistics, args.debug, conn_info, not_cross_dataset)
        logger.info('Val dataset size: {}'.format(len(valdataset)))
    
        train_loader = DataLoader(traindataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        val_loader = DataLoader(valdataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testdataset = QueryPlanDataset(logger, args.model, args.encode_table_column, dataset_dir, test_dataset, 'test', statistics, args.debug, conn_info, not_cross_dataset)

    logger.info('Test dataset size: {}'.format(len(testdataset)))
    
    test_loader = DataLoader(testdataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if args.model.startswith('Hetero'):
        # Initialize the model
        # Determine the number of unique data types for one-hot encoding
        # Assuming all graphs have the same data_type_mapping
        # sample_graph = train_loader.dataset[1]
        num_operator_features = 23 # sample_graph.x_dict['operator'].shape[1]
        num_table_features = 2 #sample_graph.x_dict['table'].shape[1] if 'table' in sample_graph.x_dict else None
        num_column_features = 11 #sample_graph.x_dict['column'].shape[1] if 'column' in sample_graph.x_dict else None
        logger.info(f"num_operator_features={num_operator_features}, num_table_features={num_table_features}, num_column_features={num_column_features}")
        model = MODELS[args.model](
            hidden_channels=args.hidden_dim, out_channels=1, num_layers=args.num_layers, encode_table_column=args.encode_table_column, 
            num_operator_features=num_operator_features, num_table_features=num_table_features, num_column_features=num_column_features, dropout=args.dropout)
    else: 
        sample_graph = test_loader.dataset[0]
        num_node_features = sample_graph.x.shape[1]
        model = MODELS[args.model](hidden_channels=args.hidden_dim, out_channels=1, num_layers = args.num_layers, num_node_features=num_node_features, dropout=args.dropout)
    
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params}")
    # 计算模型的总大小
    total_size = 0
    for param in model.parameters():
        total_size += param.numel() * param.element_size()  # numel() 是元素总数, element_size() 是每个元素的字节数
    
    # 转换为 KB, MB
    logger.info(f"Model total size: {total_size / 1024:.2f} KB")
    logger.info(f"Model total size: {total_size / (1024 ** 2):.2f} MB")

    model = model.to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.L1Loss()
        
    best_model_dir = 'checkpoints'
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)
    encode_table_column_flag = '_encode_table_column' if args.encode_table_column else ''
    best_model_path = os.path.join(best_model_dir, f"{args.model}_{'_'.join(args.train_dataset)}{encode_table_column_flag}_{'mem' if args.mem_pred else ''}_{'time' if args.time_pred else ''}_best.pth")
    early_stopping = EarlyStopping(logger, args.patience, best_model_path, verbose=True)

    if not args.skip_train:
        begin = time()
        train_epoch(logger, args.model, model, optimizer, criterion, train_loader, val_loader, early_stopping, args.epochs, args.device, statistics, 
                    args.mem_pred, args.time_pred, scaling_factor=scaling_factor)
        time_elapse = time()-begin
        print(f"training takes {time_elapse} seconds, equivalent to {time_elapse/3600:.4f} hours")

    model.load_state_dict(torch.load(best_model_path))
    avg_test_loss, metrics = validate_model(args.model, model, test_loader, criterion, statistics, args.device, args.mem_pred, args.time_pred, scaling_factor)
    logger.info(f"Test Loss={avg_test_loss:.4f}")
    if args.mem_pred:
        logger.info(f"peakmem metrics={metrics['peakmem']}")
    if args.time_pred:
        logger.info(f"time metrics={metrics['time']}")

    logging.info(f"heuristic metrics={metrics['heuristic_peakmem']}")
