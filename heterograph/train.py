import psycopg2
import torch

from torch.optim import Adam
import copy
import re
import random
import numpy as np
import json
import argparse
import logging
from tqdm import tqdm
import os
from datetime import datetime

from src.training.train import train_model



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



if __name__ == "__main__":# Set random seed for reproducibility
    
    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/home/wuy/DB/pg_mem_data', help='dataset directory')
    parser.add_argument('--train_dataset', type=str, default='tpch_sf1', help='dataset name. train and validation will use the same dataset')
    parser.add_argument('--test_dataset', type=str, default='tpch_sf1', help='dataset name')
    parser.add_argument('--skip_train', action='store_true', default=False, help='skip training')
    parser.add_argument('--epochs', type=int, default=10000, help='number of epochs')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--hidden_dim', type=int, default=32, help='hidden dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='number of layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--num_workers', type=int, default=10, help='number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use for training')
    parser.add_argument('--mem_pred', action='store_true', default=True, help='predict memory')
    parser.add_argument('--time_pred', action='store_true', default=False, help='predict time')
    args = parser.parse_args()

    assert args.mem_pred or args.time_pred, "At least one of --mem_pred (default True) and --time_pred (default False) should be set"

    log_dir = 'logs/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{args.train_dataset}_test_{args.test_dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = get_logger(log_file)

    # Train the model
    train_model(logger, args)

  