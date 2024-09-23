import torch
from tqdm import tqdm
from torch.optim import Adam
from src.models.HeteroGraph import HeteroGraph
from src.dataset.dataloader import get_loaders
from src.training.metrics import compute_metrics


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

def validate_model(model, val_loader, criterion):
    # Validation
    model.eval()
    val_loss = 0
    preds = []
    trues = []
    with torch.no_grad():
        for batch in val_loader:
            if 'operator' not in batch.x_dict:
                continue  # Skip batches without 'operator' nodes
            out = model(batch.x_dict, batch.edge_index_dict, batch['operator'].batch)
            loss = criterion(out, batch.y.squeeze())
            val_loss += loss.item()
            preds.append(out)
            trues.append(batch.y.squeeze())
    avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    metrics = compute_metrics(torch.cat(trues).cpu().numpy(), torch.cat(preds).cpu().numpy())
    return avg_val_loss, metrics

def train_epoch(logger, model, optimizer, criterion, train_loader, val_loader, early_stopping, epochs):
    logger.info("Training begins")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            # Ensure that 'operator' node type exists in the batch
            if 'operator' not in batch.x_dict:
                continue  # Skip batches without 'operator' nodes
            # print(f"Epoch {epoch+1}, Batch 'table' features shape: {batch.x_dict['table'].shape}")
            # Pass 'batch_operator' to the model
            out = model(batch.x_dict, batch.edge_index_dict, batch['operator'].batch)
            loss = criterion(out, batch.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        avg_val_loss, metrics = validate_model(model, val_loader, criterion)
        
        logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, metrics={metrics}")

        # Early stopping
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    logger.info("Training ends")


# Define training function (redefined for clarity)
def train_model(logger, args):

    train_loader, val_loader, test_loader = get_loaders(logger, args.dataset_dir, args.train_dataset, args.test_dataset, args.batch_size, args.num_workers)

    # Initialize the model
    # Determine the number of unique data types for one-hot encoding
    # Assuming all graphs have the same data_type_mapping
    sample_graph = train_loader.dataset[0]
    num_column_features = sample_graph.x_dict['column'].shape[1]  # [avg_width] + one-hot encoded data types

    model = HeteroGraph(hidden_channels=32, out_channels=1, num_column_features=num_column_features)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.L1Loss()

    
    best_model_path = 'best_model.pth'
    early_stopping = EarlyStopping(logger, args.patience, best_model_path, verbose=True)

    train_epoch(logger, model, optimizer, criterion, train_loader, val_loader, early_stopping, args.epochs)

    model.load_state_dict(torch.load(best_model_path))
    avg_test_loss, metrics = validate_model(model, test_loader, criterion)
    logger.info(f"Test Loss={avg_test_loss:.4f}, metrics={metrics}")
    