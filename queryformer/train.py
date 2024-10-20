import numpy as np
import os
import torch
import torch.nn as nn
import time
import pandas as pd
from scipy.stats import pearsonr
import logging
import json
from tqdm import tqdm
import re
import argparse
import pickle
from sklearn.preprocessing import RobustScaler

import ast  # Needed for literal_eval

from model.util import  seed_everything
from model.database_util import (
    generate_column_min_max,
    sample_all_tables,
    generate_query_bitmaps,
    generate_histograms_entire_db,
    filterDict2Hist,
    collator,
    save_histograms,
    load_entire_histograms,
    load_column_min_max,
    get_column_min_max_vals
)

from model.model import QueryFormer
from model.database_util import Encoding
from model.dataset import PlanTreeDataset
from model.trainer import eval_workload, train, evaluate

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='tpcds_sf1', help='Dataset name')
    argparser.add_argument('--data_dir', type=str, default='/home/wuy/DB/pg_mem_data', help='Data directory')
    argparser.add_argument('--seed', type=int, default=1, help='Random seed')
    argparser.add_argument('--skip_train', action='store_true', help='Skip training')
    argparser.add_argument('--bs', type=int, default=1024, help='Batch size')
    argparser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    argparser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    argparser.add_argument('--clip_size', type=int, default=50, help='Clip size')
    argparser.add_argument('--embed_size', type=int, default=64, help='Embedding size')
    argparser.add_argument('--pred_hid', type=int, default=128, help='Prediction head hidden size')
    argparser.add_argument('--ffn_dim', type=int, default=128, help='Feed-forward network hidden size')
    argparser.add_argument('--head_size', type=int, default=12, help='Multi-head attention head size')
    argparser.add_argument('--n_layers', type=int, default=8, help='Number of transformer layers')
    argparser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    argparser.add_argument('--sch_decay', type=float, default=0.6, help='Scheduler decay rate')
    argparser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device to use')
    argparser.add_argument('--newpath', type=str, default='./results/mem/', help='Directory to save results')
    argparser.add_argument('--to_predict', type=str, default='mem', help='Memory prediction task')
    argparser.add_argument('--max_workers', type=int, default=10, help='Number of multiprocessing workers')
    argparser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = argparser.parse_args()

    dataset = args.dataset
    data_dir = args.data_dir
    seed = args.seed

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s')


    column_type_file = os.path.join(os.path.dirname(__file__), f'../zsce/cross_db_benchmark/datasets/{dataset}/column_type.json')
    with open(column_type_file, 'r') as f:
        column_type = json.load(f)

    schema = {}
    for table, columns in column_type.items():
        for column, type_ in columns.items():
            if table not in schema:
                schema[table] = []
            schema[table].append(column)

    # Define table aliases as their original names for tpcds by iterating tpcds_schema, meanwhile get col2idx
    t2alias = {}
    col2idx = {}
    for table, columns in schema.items():
        for column in columns:
            t2alias[table] = table
            col2idx[table + '.' + column] = len(col2idx)

    alias2t = {v: k for k, v in t2alias.items()}

    # Ensure results directory exists
    os.makedirs(args.newpath, exist_ok=True)

    # Database connection parameters
    DB_PARAMS = {
        'database': dataset,
        'user': "wuy",
        'host': "127.0.0.1",
        'password': "wuy",
        'port': "5432"
    }


    column_min_max_vals = get_column_min_max_vals(dataset, DB_PARAMS, schema, t2alias, args.max_workers)
    

    # Initialize Scalers
    with open(os.path.join(data_dir, dataset, 'statistics_workload_combined.json')) as f:
        stats = json.load(f)
    max_label = stats['peakmem']['max']
    scale_label = stats['peakmem']['scale']
    center_label = stats['peakmem']['center']
    label_norm = RobustScaler()

    label_norm.scale_ = np.array([scale_label])
    label_norm.center_ = np.array([center_label])

    # Perform sampling per table
    sample_dir = f'./data/{dataset}/sampled_data/'
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    logging.info("Starting sampling of all tables.")
    sample_all_tables(
        db_params=DB_PARAMS,
        schema=schema,
        sample_dir=sample_dir,
        num_samples=1000,
        max_workers=args.max_workers
    )
    logging.info("Completed sampling of all tables.")

    
    encoding_file = f"./data/{dataset}/encoding.pkl"
    if not os.path.exists(encoding_file):
        logging.info("Encoding file not found. Generating encoding.")
        encoding = Encoding(column_min_max_vals=column_min_max_vals, col2idx=col2idx)
    else:
        logging.info("Loading encoding file.")
        with open(encoding_file, 'rb') as f:
            encoding = pickle.load(f)


    # Seed for reproducibility
    seed_everything(seed)

    if not args.skip_train:
        train_dataset = PlanTreeDataset(data_dir, dataset, 'val', alias2t, t2alias, schema, sample_dir, DB_PARAMS, encoding, args.max_workers, label_norm)
        logging.info(f"Training dataset length = {len(train_dataset)}")
        val_dataset = PlanTreeDataset(data_dir, dataset, 'val', alias2t, t2alias, schema, sample_dir, DB_PARAMS, encoding, args.max_workers, label_norm)
        logging.info(f"Validation dataset length = {len(val_dataset)}")
        with open(encoding_file, 'wb') as f:
            pickle.dump(encoding, f)

    test_dataset = PlanTreeDataset(data_dir, dataset, 'test', alias2t, t2alias, schema, sample_dir, DB_PARAMS, encoding, args.max_workers, label_norm)
    logging.info(f"Test dataset length = {len(test_dataset)}")
    # print("type2idx:", encoding.join2idx)
    # print("table2idx:", encoding.table2idx)
    # print(f"encoding.join2idx length = {len(encoding.join2idx)}")
   

    # Initialize the model
    model = QueryFormer(
        emb_size=args.embed_size,
        ffn_dim=args.ffn_dim,
        head_size=args.head_size,
        dropout=args.dropout,
        n_layers=args.n_layers,
        use_sample=True,
        use_hist=True,
        pred_hid=args.pred_hid,
        joins = len(encoding.join2idx), 
        tables = len(encoding.table2idx),
        types = len(encoding.type2idx),
    ).to(args.device)

    logging.info("Initialized QueryFormer model.")

    # Define loss function
    crit = nn.MSELoss()

    if not args.skip_train:
        # Train the model
        model, best_path = train(model, train_dataset, val_dataset, crit, label_norm, args)
        logging.info(f"Training completed. Best model saved at: {best_path}")


    # Load best model
    full_best_path = os.path.join(args.newpath, best_path)
    loaded_model = torch.load(full_best_path)
    model.load_state_dict(loaded_model['model'])
    logging.info(f"Loaded best model from {full_best_path}")
    # test on test_dataset
    result = evaluate(model, test_dataset, args.bs, label_norm, args.device, prints=True)
    print(f"test result: {result}")


if __name__ == '__main__':
    main()
