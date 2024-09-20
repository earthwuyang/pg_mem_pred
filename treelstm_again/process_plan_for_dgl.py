import json
from plannode import PlanNode, parse_postgres_plan
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
import numpy as np
from tqdm import tqdm
import pickle

def _inv_log1p(x):
    return np.exp(x) - 1

filename = '../tpch_data/parsed_plan.json'

with open(filename, 'r') as f:
    plans = json.load(f)

print(f"json file loaded")

edge_src_nodes_list = []
edge_tgt_nodes_list = []
features_list_list = []
peakmem_list = []
num_nodes_list = []

for plan in tqdm(plans['parsed_plans']):
    number_nodes = 0
    node, _ = parse_postgres_plan(plan)
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

with open('edge_src_nodes_list.pkl','wb') as f:
    pickle.dump(edge_src_nodes_list, f)

with open('edge_tgt_nodes_list.pkl','wb') as f:
    pickle.dump(edge_tgt_nodes_list, f)

with open('features_list.pkl','wb') as f:
    pickle.dump(features_list_list, f)

with open('peakmem_list.pkl','wb') as f:    
    pickle.dump(peakmem_list, f)



print(f"saved to pkl files")