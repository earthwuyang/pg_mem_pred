
import argparse
import torch
import logging
import os
from datetime import datetime
import numpy as np
import random
import json
from torch_geometric.loader import DataLoader

from src.models.GIN import GIN
from src.models.GAT import GAT
from src.models.GCN import GCN
from src.models.GraphTransformer import GraphTransformer
from src.models.TreeTransformer import TreeTransformer
from src.models.TreeLSTM import TreeLSTM

from src.utils.utils import load_json
from src.datasets.dataset import PlanGraphDataset

from src.training.train import train_model, evaluate_model
from src.training.metrics import compute_metrics

MODELS = {
    'GIN': GIN,
    'GAT': GAT,
    'GCN': GCN,
    'GraphTransformer': GraphTransformer,
    'TreeTransformer': TreeTransformer,
    'TreeLSTM': TreeLSTM  # currently not supported
}

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', type=str, default='/home/wuy/DB/pg_mem_data', help='dataset directory ')
    parser.add_argument('--train_dataset', type=str, default='tpch_sf1', help='dataset name (default: tpch). train and validation will use the same dataset')
    parser.add_argument('--test_dataset', type=str, default='tpch_sf1', help='dataset name (default: tpch)')
    parser.add_argument('--skip_train', action='store_true', default=False, help='skip training')
    
    parser.add_argument('--model_name', type=str, default='GIN', help='model name')
    parser.add_argument('--hidden_channels', type=int, default=64, help='number of hidden channels')
    
    parser.add_argument('--batch_size', type=int, default=10240, help='batch size')
    parser.add_argument('--num_workers', type=int, default=10, help='number of workers')
    parser.add_argument('--epochs', type=int, default=10000, help='number of epochs')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)

    args = parser.parse_args()

    log_dir = f"./logs/{args.model_name}_{args.train_dataset}/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file_path = os.path.join(log_dir, log_file)
    logger = get_logger(log_file_path)

    logger.info(f"args: {args}")

    statistics_file_path = os.path.join(args.dataset_dir, args.train_dataset, 'statistics_workload_combined.json')
    with open(statistics_file_path, 'r') as f:
        statistics = json.load(f)
    
    if not args.skip_train:
        train_file = os.path.join(args.dataset_dir, args.train_dataset, 'train_plans.json')
        val_file = os.path.join(args.dataset_dir, args.train_dataset, 'val_plans.json')

        train_plans = load_json(train_file)
        val_plans = load_json(val_file)

        train_dataset = PlanGraphDataset(train_plans, statistics)
        logger.info(f"train dataset size: {len(train_dataset)}")
        val_dataset = PlanGraphDataset(val_plans, statistics)
        logger.info(f"val dataset size: {len(val_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    test_file = os.path.join(args.dataset_dir, args.test_dataset, 'test_plans.json')
    test_plans = load_json(test_file)
    test_dataset = PlanGraphDataset(test_plans, statistics)
    logger.info(f"test dataset size: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    num_node_features = len(train_loader.dataset[0].x[0])  # Number of features per node
    model = MODELS[args.model_name](num_node_features, args.hidden_channels)
        
    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')
    model = model.to(device)

    # Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    criterion = torch.nn.MSELoss()

    checkpoint_dir ='./checkpoints/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f"{args.model_name}_{args.train_dataset}.pth")

    mem_scaler = train_loader.dataset.mem_scaler

    if not args.skip_train:
        train_model(logger, model, train_loader, val_loader, optimizer, criterion, device, args.epochs, checkpoint_path, statistics, mem_scaler)

    logger.info(f"reload best model from '{checkpoint_path}' and evaluate on test set")
    model.load_state_dict(torch.load(checkpoint_path))
    test_loss, metrics = evaluate_model(model, test_loader, criterion, device, statistics, mem_scaler)
    logger.info(f'Test metrics on {args.test_dataset}, trained on {args.train_dataset}: {metrics}')
        

