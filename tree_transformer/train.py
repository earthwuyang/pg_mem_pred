from model import SimpleTransformerModel
import torch
from torch.utils.data import random_split

from dataloader import PlanDataset
import torch.optim as optim
import logging
import torch.nn as nn
from metrics import compute_metrics
import argparse
from datetime import datetime
import os
import random
import json

def get_logger(logfile):

    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    fmt = f"[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d]:%(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    fh=logging.FileHandler(logfile)
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    log.addHandler(ch)
    log.addHandler(fh)
    return log

def validate_model(model, dataloader, loss_fn, device, memory_scaler):
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i, (features, peakmem) in enumerate(dataloader):
                features = features.to(device)
                features = features.permute(1, 0, 2)
                peakmem = peakmem.to(device)
                output = model(features)
                loss = loss_fn(output.squeeze(), peakmem)
                running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        original_memory = memory_scaler.inverse_transform(peakmem.cpu().reshape(-1, 1)).reshape(-1)
        output =memory_scaler.inverse_transform(output.cpu().squeeze().reshape(-1, 1)).reshape(-1)
        metrics = compute_metrics(original_memory, output)
        return avg_loss, metrics

def train_epoch(model, dataloader, optimizer, loss_fn, device, epochs=1000, lr=0.001):

    model.train()
    running_loss = 0.0
    for i, (features, peakmem) in enumerate(dataloader):
        # Zero gradients    
        features = features.to(device)
        features = features.permute(1, 0, 2)
        peakmem = peakmem.to(device)
        # Forward pass
        output = model(features)
        # print(f"output.shape: {output.shape}")
        optimizer.zero_grad()
        # Compute loss
        loss = loss_fn(output.squeeze(), peakmem)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss

def train_model(logger, args, model, train_dataloader, val_dataloader, optimizer, loss_fn, device, memory_scaler, best_model_path, epochs=1000, lr=0.001):
    patience = 20

    best_qerror = float('inf')
    epochs_no_improve = 0
    early_stop = False

    
    if not args.skip_train:
        logger.info(f"Training model begins for {epochs} epochs with learning rate {lr}")
        for epoch in range(epochs):
            avg_epoch_loss = train_epoch(model, train_dataloader, optimizer, loss_fn, device, epochs=epochs, lr=lr)
            avg_val_loss, metrics = validate_model(model, val_dataloader, loss_fn, device, memory_scaler)
            median_qerror = metrics['qerror_50 (Median)']
            if best_qerror > median_qerror:
                best_qerror = median_qerror
                epochs_no_improve = 0
                logger.info(f"epochs_no_improve: {epochs_no_improve}")
                torch.save(model.state_dict(), best_model_path)
            else:
                epochs_no_improve += 1
                logger.info(f"epochs_no_improve: {epochs_no_improve}")
                early_stop = epochs_no_improve >= patience
                if early_stop:
                    break
            logger.info(f"Epoch {epoch+1}/{epochs} - Train loss: {avg_epoch_loss:.4f} - Val loss: {avg_val_loss:.4f} \nMetrics: {metrics}")
    return early_stop

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train_dataset', type=str, default='tpch', help='Dataset to use (tpch, tpcds)')
    argparser.add_argument('--test_dataset', type=str, default='tpch', help='Dataset to test (tpch, tpcds)')
    argparser.add_argument('--skip_train', action='store_true', help='Skip training')
    args = argparser.parse_args()

    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"train_{args.train_dataset}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')}.log")
    logger = get_logger(log_file)

    random.seed(1)
    torch.manual_seed(1)

    train_plan_file = f'../{args.train_dataset}_data/train_plans.json'
    test_plan_file = f'../{args.test_dataset}_data/val_plans.json'

    statistics_file = f'../{args.train_dataset}_data/statistics_workload_combined.json'
    with open(statistics_file, 'r') as f:
        statistics = json.load(f)

    train_dataset = PlanDataset(logger, train_plan_file, statistics)
    test_dataset = PlanDataset(logger, test_plan_file, statistics)

    total_train_dataset = train_dataset

    train_size = int(0.9 * len(total_train_dataset))
    val_size = len(total_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
            dataset=train_dataset,
            lengths=[train_size, val_size],
            generator=torch.Generator().manual_seed(1)
        )

    batch_size = 10240
    max_epochs = 1000
    lr = 0.1



    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=10)


    # Initialize model
    input_dim = 7  # Number of input features
    hidden_dim = 128
    output_dim = 1  # Predicting a single value: peakmem

    

    model = SimpleTransformerModel(input_dim, hidden_dim, output_dim)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    best_model_path = os.path.join(checkpoint_dir, f"best_plan_net_{args.train_dataset}.pth")
    # Train the model
    early_stop = train_model(logger, args, model, train_dataloader, val_dataloader, optimizer, loss_fn, device, total_train_dataset.memory_scaler, best_model_path, epochs=max_epochs, lr=lr)
    if early_stop:
        logger.info(f"Early stopping after {patience} epochs without improvement")

    # Test the model
    if args.skip_train and not os.path.exists(best_model_path):
        logger.info(f"Skipping test because model {best_model_path} does not exist")
    else:
        model.load_state_dict(torch.load(best_model_path))
        _, metrics = validate_model(model, test_dataloader, loss_fn, device, total_train_dataset.memory_scaler)
        logger.info(f"Test metrics: {metrics}")


if __name__ == '__main__':
    main()