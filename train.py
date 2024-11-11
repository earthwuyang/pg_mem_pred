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
import collections
from src.training.train import train_model
from src.training.train_xgboost import train_XGBoost
from src.preprocessing.get_explain_json_plans import get_explain_json_plans
from src.preprocessing.extract_mem_time_info import extract_mem_info
from src.dataset.combine_stats import combine_stats

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--db_config', type=str, default='conn.json', help='database configuration file')
    parser.add_argument('--data_dir', type=str, default='/home/wuy/DB/pg_mem_data', help='dataset directory')
    parser.add_argument('--dataset', nargs='+', type=str, default=['tpch_sf1'], help='dataset name. train and validation will use the same dataset')
    parser.add_argument('--val_dataset', type=str, default=None, help='dataset name. validation will use the same dataset')
    parser.add_argument('--test_dataset', type=str, default=None, help='dataset name. test will use the same dataset')
    parser.add_argument('--model', type=str, default='HeteroGraphConv', help='model name') # XGBoost, GIN, HeteroGraphConv, HeteroGraphRGCN
    parser.add_argument('--encode_table_column', action='store_true', default=False, help='encode table and column nodes')
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
    parser.add_argument('--no_mem_pred', action='store_false', dest='mem_pred', help='do not predict memory')
    parser.add_argument('--time_pred', action='store_true', default=False, help='predict time')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--force', action='store_true', default=False, help='force overwrite existing files')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
    args = parser.parse_args()

    if args.test_dataset is None:
        assert len(args.dataset) == 1, "if test dataset not specified, --dataset must only have one item"
        args.test_dataset = args.dataset[0]
        args.val_dataset = args.dataset[0]
    args.train_dataset = args.dataset

    assert args.mem_pred or args.time_pred, "At least one of --mem_pred (default True) and --time_pred (default False) should be set"

    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False

    log_dir = 'logs/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{'_'.join(args.train_dataset)}_test_{'_'.join(args.test_dataset)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = get_logger(log_file)
    logger.info(f"Args: {args}")

    if args.skip_train:
        dataset_list = [args.test_dataset]
    else:
        dataset_list = args.train_dataset + [args.val_dataset, args.test_dataset]
    for dataset in dataset_list:
        if args.force or not os.path.exists(os.path.join(args.data_dir, dataset, 'total_plans.json')):
            get_explain_json_plans(args.data_dir, dataset)
        else:
            logger.info(f"explain json plans of {dataset} already exist, skipping getting explain json plans")


    # combined_stats = combine_stats(logger, args, dataset_list)
    with open(os.path.join(args.data_dir, 'combined_statistics_workload.json'), 'r') as f:
        combined_stats = json.load(f)

    # Train the model
    if args.model == 'XGBoost':
        train_XGBoost(logger, args, combined_stats)
    else:
        train_model(logger, args, combined_stats)

  