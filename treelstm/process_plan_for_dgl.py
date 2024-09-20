import json
from plannode import PlanNode, parse_postgres_plan
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
import numpy as np
from tqdm import tqdm
import pickle
import os
import argparse


def _inv_log1p(x):
    return np.exp(x) - 1

def process(dataset, mode): # mode in ['train', 'val']
    filename = f'../{dataset}_data/{mode}_plans.json'
    print(f"processing {filename}")

    with open(filename, 'r') as f:
        plans = json.load(f)

    print(f"json file loaded")

    edge_src_nodes_list = []
    edge_tgt_nodes_list = []
    features_list_list = []
    peakmem_list = []
    num_nodes_list = []
    statistics_file = f'../{dataset}_data/statistics_workload_combined.json'
    with open(statistics_file) as f:
        statistics = json.load(f)

    for plan in tqdm(plans['parsed_plans']):
        number_nodes = 0
        node, _ = parse_postgres_plan(plan, statistics)
        edge_src_nodes = []
        edge_tgt_nodes = []
        features_list = []
        edge_src_nodes, edge_tgt_nodes, features_list = node.traverse_tree(edge_src_nodes, edge_tgt_nodes, features_list)
        edge_src_nodes_list.append(edge_src_nodes)
        edge_tgt_nodes_list.append(edge_tgt_nodes)
        features_list_list.append(features_list)
        peakmem_list.append(plan['peakmem'])

    # memory_scaler = FunctionTransformer(np.log1p, _inv_log1p, validate=True)
    # peakmem_list = memory_scaler.fit_transform(np.array(peakmem_list).reshape(-1,1)).reshape(-1)
    print(f"data preprocessing finished")

    dir = os.path.join(dataset, mode)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(os.path.join(dir, 'edge_src_nodes_list.pkl'), 'wb') as f:
        pickle.dump(edge_src_nodes_list, f)

    with open(os.path.join(dir, 'edge_tgt_nodes_list.pkl'), 'wb') as f:
        pickle.dump(edge_tgt_nodes_list, f)

    with open(os.path.join(dir, 'features_list.pkl'), 'wb') as f:
        pickle.dump(features_list_list, f)

    with open(os.path.join(dir, 'peakmem_list.pkl'), 'wb') as f:
        pickle.dump(peakmem_list, f)

    print(f"{mode} saved to pkl files")

def parse_args():
    parser = argparse.ArgumentParser(description='Process PostgreSQL plan for DGL')
    parser.add_argument('--dataset', type=str, default='tpch', help='dataset name')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    dataset = args.dataset
    for mode in ['train', 'val']:
        process(dataset, mode)
