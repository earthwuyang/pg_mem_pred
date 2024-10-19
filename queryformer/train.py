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

from sklearn.preprocessing import RobustScaler

import ast  # Needed for literal_eval

from model.util import Normalizer, seed_everything
from model.database_util import (
    generate_column_min_max,
    sample_all_tables,
    generate_query_bitmaps,
    generate_histograms_entire_db,
    filterDict2Hist,
    collator,
    save_histograms,
    load_entire_histograms
)
from model.model import QueryFormer
from model.database_util import Encoding
from model.dataset import PlanTreeDataset
from model.trainer import eval_workload, train


# Load column_min_max_vals from CSV
def load_column_min_max(file_path):
    """
    Loads column min and max values from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        dict: Dictionary mapping column names to (min, max).
    """
    df = pd.read_csv(file_path)
    column_min_max_vals = {}
    for _, row in df.iterrows():
        column_min_max_vals[row['name']] = (row['min'], row['max'])
    return column_min_max_vals

def extract_query_info_from_plan(row, query_id, alias2t):
    json_plan = row['Plan']
    # Extract tables
    tables = set()
    joins = []
    predicates = []

    def parse_plan_node(node, parent_alias=None):
        alias = node.get('Alias', parent_alias)
        # Extract tables
        if 'Relation Name' in node:
            tables.add(node['Relation Name'])
            alias = node.get('Alias', node['Relation Name'])
            logging.debug(f"Query_id={query_id}: Detected table '{node['Relation Name']}' with alias '{alias}'.")

        # Process joins
        if 'Hash Cond' in node or 'Join Filter' in node:
            join_cond = node.get('Hash Cond', node.get('Join Filter'))
            if join_cond:
                joins.append(join_cond)
                logging.debug(f"Query_id={query_id}: Detected join condition: {join_cond}")

        # Process predicates
        conditions = []
        for cond_type in ['Filter', 'Index Cond', 'Recheck Cond']:
            if cond_type in node:
                conditions.append(node[cond_type])

        # Include full table name in predicates
        for cond in conditions:
            # Remove type casts using regex and clean the condition
            cond_clean = re.sub(r"::\w+", "", cond).replace('(', '').replace(')', '').strip()
            preds = cond_clean.split(' AND ')
            for pred in preds:
                # print(f"pred {pred}")
                parts = pred.strip().split(' ', 2)
                if len(parts) == 3:
                    col, op, val = parts
                    # Check if 'val' is a column name (contains '.')
                    if '.' in val:
                        # This is a join predicate, skip adding to predicates
                        continue
                    if '.' not in col:
                        if alias:
                            table = alias2t.get(alias)
                            if not table:
                                logging.warning(f"Alias '{alias}' not found in alias2t mapping. Skipping predicate '{pred}'.")
                                continue
                            col = f"{table}.{col}"
                            logging.debug(f"Query_id={query_id}: Prefixed column '{col}' with table '{table}'.")
                        else:
                            logging.warning(f"Cannot determine alias for column '{col}' in query_id={query_id}. Skipping predicate.")
                            continue
                    # Attempt to convert val to float; if it fails, skip the predicate
                    try:
                        val = float(val)
                        predicates.append(f"({col} {op} {val})")
                    except ValueError:
                        logging.warning(f"Non-numeric value '{val}' in predicate '{pred}' for query_id={query_id}. Skipping predicate.")
                        continue
                else:
                    logging.warning(f"Incomplete predicate: '{pred}' in query_id={query_id}. Skipping.")

        # Recursively handle subplans
        if 'Plans' in node:
            for subplan in node['Plans']:
                parse_plan_node(subplan, parent_alias=alias)

    parse_plan_node(json_plan)

    # Join tables, joins, and predicates into the desired format
    table_str = ",".join(sorted(list(tables)))
    join_str = ",".join(joins) if joins else ""
    predicate_str = ",".join(predicates) if predicates else ""
    mem = row['peakmem']
    mem_str = str(mem)

    logging.debug(f"Query_id={query_id}: Extracted query info: tables={table_str}, joins={join_str}, predicates={predicate_str}, cardinality={mem_str}")

    return f"{table_str}#{join_str}#{predicate_str}#{mem_str}"


def generate_for_samples(json_plans, output_path, alias2t):
    
    query_info_list = []
    print(f"Extracting query information from {len(json_plans)} plans.")
    for idx, row in tqdm(enumerate(json_plans), total=len(json_plans)):
        try:
            query_info = extract_query_info_from_plan(row, query_id=idx, alias2t=alias2t)
            query_info_list.append(query_info)
        except json.JSONDecodeError as e:
            logging.error(f"JSON decoding error for query_id={idx}: {e}. Skipping this query.")
            query_info_list.append("NA#NA#NA#0")  # Placeholder for failed parsing
        except KeyError as e:
            logging.error(f"Missing key {e} in query_id={idx}. Skipping this query.")
            query_info_list.append("NA#NA#NA#0")  # Placeholder for failed parsing

    # Save the query information to a CSV file
    with open(output_path, 'w') as f:
        for query_info in query_info_list:
            f.write(f"{query_info}\n")

    logging.info(f"extracted query information file saved to: {output_path}")



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s')

data_dir = '/home/wuy/DB/pg_mem_data'
dataset = 'tpcds_sf1'

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



# Define Args
class Args:
    bs = 36
    lr = 0.001
    epochs = 100
    clip_size = 50
    embed_size = 64
    pred_hid = 128
    ffn_dim = 128
    head_size = 12
    n_layers = 8
    dropout = 0.1
    sch_decay = 0.6
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    newpath = './results/mem/'
    to_predict = 'mem'
    max_workers = 10  # Limit the number of multiprocessing workers

args = Args()

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



column_min_max_file = f'./data/{dataset}/column_min_max_vals.csv'
if not os.path.exists(column_min_max_file):
    logging.info(f"Generating column min-max values and saving to '{column_min_max_file}'.")
    generate_column_min_max(
        db_params=DB_PARAMS,
        schema=schema,
        output_file=column_min_max_file,
        t2alias=t2alias,
        max_workers=args.max_workers,
        pool_minconn=1,
        pool_maxconn=args.max_workers  # Ensure pool_maxconn >= max_workers
    )

column_min_max_vals = load_column_min_max(column_min_max_file)
logging.info(f"Loaded column min-max values from '{column_min_max_file}'.")

# Initialize Normalizers
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

mode = 'tiny'
train_file = os.path.join(data_dir, dataset, f'{mode}_plans.json')
val_file = os.path.join(data_dir, dataset, f'{mode}_plans.json')
test_file = os.path.join(data_dir, dataset, f'{mode}_plans.json')

with open(train_file, 'r') as f:
    train_plans = json.load(f)

generated_train_path = f'./data/{dataset}/generated_queries.csv'

# Generate CSV for queries based on train_df and val_df
if os.path.exists(generated_train_path):
    logging.warning(f"Generated CSV file '{generated_train_path}' already exists. Skipping generation.")
else:
    generate_for_samples(train_plans, generated_train_path, alias2t)
    logging.info(f"Generated CSV file for training queries saved to: {generated_train_path}")


# Load all queries from the generated CSV
column_names = ['tables', 'joins', 'predicate', 'cardinality']
try:
    query_file = pd.read_csv(
        generated_train_path,
        sep='#',
        header=None,
        names=column_names,
        keep_default_na=False,   # Do not convert empty strings to NaN
        na_values=['']           # Treat empty strings as empty, not NaN
    )
    # only fetch the first 100 queries for testing
    # query_file = query_file.head(100)
except pd.errors.ParserError as e:
    logging.error(f"Error reading generated_queries.csv: {e}")
    exit(1)

# Generate bitmaps for each query based on pre-sampled table data
logging.info("Generating table sample bitmaps for each query.")

sampled_data = generate_query_bitmaps(
    query_file=query_file,
    alias2t=alias2t,
    sample_dir=sample_dir
)

# # After generating table_sample_bitmaps
# print(f"Type of table_sample: {type(sampled_data)}")  # Should be list
# print(f"Number of queries: {len(sampled_data)}")
# if len(sampled_data) > 0:
#     print(f"Type of first entry: {type(sampled_data[0])}")  # Should be dict
#     print(f"Keys in first entry: {list(sampled_data[0].keys())}")


logging.info("Completed generating table sample bitmaps for all queries.")

# Generate histograms based on entire tables
hist_dir = f'./data/{dataset}/histograms/'
histogram_file_path = f'./data/{dataset}/histogram_entire.csv'

if not os.path.exists(histogram_file_path):
    hist_file_df = generate_histograms_entire_db(
        db_params=DB_PARAMS,
        schema=schema,
        hist_dir=hist_dir,
        bin_number=50,
        t2alias=t2alias,
        max_workers=args.max_workers
    )
    # Save histograms with comma-separated bins
    save_histograms(hist_file_df, save_path=histogram_file_path)
else:
    hist_file_df = load_entire_histograms(load_path=histogram_file_path)

encoding = Encoding(column_min_max_vals=column_min_max_vals, col2idx=col2idx)
logging.info("Initialized Encoding object.")

# Seed for reproducibility
seed_everything()

# Initialize PlanTreeDataset
train_ds = PlanTreeDataset(
    json_df=train_plans,
    workload_df=None,  # Assuming workload_df is not needed for training
    encoding=encoding,
    hist_file=hist_file_df,
    table_sample=sampled_data,  # This should be a list indexed by query_id
    alias2t=alias2t,
)

val_ds = PlanTreeDataset(
    json_df=train_plans,
    workload_df=None,  # Assuming workload_df is not needed for validation
    encoding=encoding,
    hist_file=hist_file_df,
    table_sample=sampled_data,  # Ensure consistency
    alias2t=alias2t,
)

logging.info("Initialized training and validation datasets with sampled data.")

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

# Train the model
model, best_path = train(model, train_ds, val_ds, crit, label_norm, args)
logging.info(f"Training completed. Best model saved at: {best_path}")

# Define methods dictionary for evaluation
methods = {
    'get_sample': lambda workload_file: generate_query_bitmaps(
        query_file=pd.read_csv(workload_file, sep='#', header=None, names=['tables', 'joins', 'predicate', 'cardinality'], keep_default_na=False, na_values=['']),
        alias2t=alias2t,
        sample_dir=sample_dir
    ),
    'encoding': encoding,
    'label_norm': label_norm,
    'hist_file': hist_file_df,
    'model': model,
    'device': args.device,
    'bs': 512,
}

exit()
# Evaluate on 'job-light' workload
job_light_workload_file = './data/tpcds/workloads/job-light.csv'
if os.path.exists(job_light_workload_file):
    job_light_scores, job_light_corr = eval_workload('job-light', methods)
    logging.info(f"Job-Light Workload Evaluation: {job_light_scores}, Correlation: {job_light_corr}")
else:
    logging.warning(f"Job-Light workload file '{job_light_workload_file}' does not exist. Skipping evaluation.")

# Evaluate on 'synthetic' workload
synthetic_workload_file = './data/tpcds/workloads/synthetic.csv'
if os.path.exists(synthetic_workload_file):
    synthetic_scores, synthetic_corr = eval_workload('synthetic', methods)
    logging.info(f"Synthetic Workload Evaluation: {synthetic_scores}, Correlation: {synthetic_corr}")
else:
    logging.warning(f"Synthetic workload file '{synthetic_workload_file}' does not exist. Skipping evaluation.")
