from tqdm import tqdm
import torch
from tqdm import tqdm
from .metrics import compute_metrics

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in train_loader:        
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)   

def evaluate_model(model, loader, criterion, device, statistics, mem_scaler):
    model.eval()
    total_loss = 0
    out_list = []
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y)
        total_loss += loss.item() * data.num_graphs
        out_list.append(out.detach().cpu())
    out = torch.cat(out_list, dim=0)
    labels = data.y.cpu()
    # center = statistics['peakmem']['center']
    # scale = statistics['peakmem']['scale']
    # out = (out * scale) + center
    # out = out.numpy()
    # labels = (labels * scale) + center
    # labels = labels.numpy()
    out = mem_scaler.inverse_transform(out.reshape(-1, 1)).reshape(-1)
    labels = mem_scaler.inverse_transform(labels.reshape(-1, 1)).reshape(-1)
    metrics = compute_metrics(labels, out)
    return total_loss / len(loader.dataset), metrics

def train_model(logger, model, train_loader, val_loader, optimizer, criterion, device, num_epochs, checkpoint_path, statistics, mem_scaler):
    logger.info('Training begins')

    model.train()
    total_loss = 0
    # early stopping
    best_val_loss = float('inf')
    patience = 20
    no_improvement = 0
    best_model = None
    for epoch in range(1, num_epochs+1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, metrics = evaluate_model(model, val_loader, criterion, device, statistics, mem_scaler)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            no_improvement = 0
            logger.debug(f"no_improvement: {no_improvement}")
            #save model
            torch.save(best_model.state_dict(), checkpoint_path)
        else:
            no_improvement += 1
            logger.debug(f"no_improvement: {no_improvement}")
        if no_improvement == patience:
            logger.info(f'Early stopping at epoch {epoch}')
            break
        if no_improvement == patience:
            logger.info(f'No improvement for {no_improvement} epochs, stopping training')
            break
        logger.info(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Metrics: {metrics}')
    return best_model
