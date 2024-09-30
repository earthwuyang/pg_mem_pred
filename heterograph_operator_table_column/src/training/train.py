import torch
from tqdm import tqdm
from torch.optim import Adam
import os
import pickle
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler, FunctionTransformer
from torch_geometric.loader import DataLoader
from src.dataset.dataset import load_json
from src.models.HeteroGraph import HeteroGraph
from src.training.metrics import compute_metrics
from src.dataset.dataset import QueryPlanDataset



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

def validate_model(model, val_loader, criterion, scalers, device, mem_pred, time_pred):
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
            if 'operator' not in batch.x_dict:
                continue  # Skip batches without 'operator' nodes
            out_mem, out_time = model(batch.x_dict, batch.edge_index_dict, batch['operator'].batch)
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
    mem_preds = scalers['peakmem'].inverse_transform(np.array(mem_preds).reshape(-1, 1)).reshape(-1)
    time_preds = scalers['time'].inverse_transform(np.array(time_preds).reshape(-1, 1)).reshape(-1)
    memories = scalers['peakmem'].inverse_transform(np.array(memories).reshape(-1, 1)).reshape(-1)
    times = scalers['time'].inverse_transform(np.array(times).reshape(-1, 1)).reshape(-1)
    metrics={}
    metrics['peakmem'] = compute_metrics(memories, mem_preds)
    metrics['time'] = compute_metrics(times, time_preds)
    return avg_val_loss, metrics

def train_epoch(logger, model, optimizer, criterion, train_loader, val_loader, early_stopping, epochs, device, scalers, mem_pred, time_pred):
    logger.info("Training begins")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Train:"):
            batch = batch.to(device)
            optimizer.zero_grad()
            # Ensure that 'operator' node type exists in the batch
            if 'operator' not in batch.x_dict:
                continue  # Skip batches without 'operator' nodes
            # print(f"Epoch {epoch+1}, Batch 'table' features shape: {batch.x_dict['table'].shape}")
            # Pass 'batch_operator' to the model
            out_mem, out_time = model(batch.x_dict, batch.edge_index_dict, batch['operator'].batch)
            labels = batch.y.reshape(-1,2)
            mem_loss = criterion(out_mem, labels[:, 0])
            time_loss = criterion(out_time, labels[:, 1])
            # print(f"mem_loss: {mem_loss}, time_loss: {time_loss}")
            # while 1:pass
            loss = 0
            if mem_pred:
                loss += mem_loss
            if time_pred:
                loss += time_loss   
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        avg_val_loss, metrics = validate_model(model, val_loader, criterion, scalers, device, mem_pred, time_pred)
        
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


class LogRobustScaler(): # first robustscaler then log
    def __init__(self):
        self.scaler = RobustScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        Y = self.scaler.transform(X)
        Y_log = np.log1p(Y)
        return Y_log
    
    def fit_transform(self, X, y=None):
        Y=self.scaler.fit_transform(X)
        Y_log = np.log1p(Y)
        return Y_log

    def inverse_transform(self, X):
        X_exp = np.expm1(X)  # exp(X) - 1 to reverse log(1 + X)
        return self.scaler.inverse_transform(X_exp)  # inverse the robust scaling


def get_scalers(dataset_dir, train_dataset, mode):
    scaler_pickle_path = os.path.join('data', f'{train_dataset}_scaler.pkl')
    if not os.path.exists(os.path.dirname(scaler_pickle_path)):
        os.makedirs(os.path.dirname(scaler_pickle_path))
    if os.path.exists(scaler_pickle_path):
        with open(scaler_pickle_path, 'rb') as f:
            scalers = pickle.load(f)
    else:
        json_file_path = os.path.join(dataset_dir, train_dataset, f'{mode}_plans.json')
        plans = load_json(json_file_path)
        peakmem_list = []
        time_list = []
        for plan in plans:
            peakmem_list.append(plan['peakmem'])
            time_list.append(plan['time'])
        scalers = {}
        scalers['peakmem'] = LogRobustScaler()
        scalers['peakmem'].fit_transform(np.array(peakmem_list).reshape(-1, 1)).reshape(-1)
        scalers['time'] = LogRobustScaler()
        scalers['time'].fit_transform(np.array(time_list).reshape(-1, 1)).reshape(-1)
        with open(scaler_pickle_path, 'wb') as f:
            pickle.dump(scalers, f)
    return scalers

# Define training function (redefined for clarity)
def train_model(logger, args):

    dataset_dir = args.dataset_dir
    train_dataset = args.train_dataset
    test_dataset = args.test_dataset
    batch_size = args.batch_size
    num_workers = args.num_workers

    scalers = get_scalers(dataset_dir, train_dataset, 'train')
    # Load the dataset
    if not args.skip_train:
        traindataset = QueryPlanDataset(logger, dataset_dir, train_dataset, 'train', scalers, args.debug)
        logger.info('Train dataset size: {}'.format(len(traindataset)))
        valdataset = QueryPlanDataset(logger, dataset_dir, train_dataset, 'val', scalers, args.debug)
        logger.info('Val dataset size: {}'.format(len(valdataset)))
        train_loader = DataLoader(traindataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        val_loader = DataLoader(valdataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testdataset = QueryPlanDataset(logger, dataset_dir, test_dataset, 'test', scalers, args.debug)  

    logger.info('Test dataset size: {}'.format(len(testdataset)))
    
    test_loader = DataLoader(testdataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Initialize the model
    # Determine the number of unique data types for one-hot encoding
    # Assuming all graphs have the same data_type_mapping
    sample_graph = test_loader.dataset[0]
    num_column_features = sample_graph.x_dict['column'].shape[1]  # [avg_width] + one-hot encoded data types

    # output dim is 2 for peakmem and time
    model = HeteroGraph(hidden_channels=args.hidden_dim, out_channels=1, num_layers=args.num_layers, num_column_features=num_column_features)
    model = model.to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.L1Loss()


    # avg_test_loss, metrics = validate_model(model, test_loader, criterion, mem_scaler)
    # logger.info(f"before train, model metrics Test Loss={avg_test_loss:.4f}, metrics={metrics}")
        
    best_model_path = 'best_model.pth'
    early_stopping = EarlyStopping(logger, args.patience, best_model_path, verbose=True)

    if not args.skip_train:
        train_epoch(logger, model, optimizer, criterion, train_loader, val_loader, early_stopping, args.epochs, args.device, scalers, 
                    args.mem_pred, args.time_pred)

    model.load_state_dict(torch.load(best_model_path))
    avg_test_loss, metrics = validate_model(model, test_loader, criterion, scalers, args.device, args.mem_pred, args.time_pred)
    logger.info(f"Test Loss={avg_test_loss:.4f}, \npeakmem metrics={metrics['peakmem']}, \ntime metrics={metrics['time']}")
