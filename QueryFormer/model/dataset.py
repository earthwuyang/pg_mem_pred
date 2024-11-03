import torch
from torch.utils.data import Dataset
import numpy as np
import json
import pandas as pd
import sys, os
from collections import deque
from tqdm import tqdm
from .database_util import formatFilter, formatJoin, TreeNode, filterDict2Hist
from .database_util import *

import sqlparse
from sqlparse.sql import Comparison, Where
from sqlparse.tokens import Keyword, DML
import re
import pickle

def is_number(value):
    """
    Check if the given value is a number.
    
    Args:
        value (str): The value to check.
        
    Returns:
        bool: True if value is a number, False otherwise.
    """
    try:
        float(value)
        return True
    except ValueError:
        return False

def extract_numeric_predicates(sql_query):
    """
    Extract predicates from the SQL WHERE clause where the right side of the operator is a number.
    
    Args:
        sql_query (str): The SQL query string.
        
    Returns:
        list: A list of numeric predicates as strings.
    """
    # Parse the SQL query
    parsed = sqlparse.parse(sql_query)
    if not parsed:
        return []
    
    stmt = parsed[0]
    numeric_predicates = []

    def extract_from_tokens(tokens):
        """
        Recursively traverse tokens to find numeric comparisons.
        
        Args:
            tokens (list): List of sqlparse tokens.
        """
        for token in tokens:
            if isinstance(token, Comparison):
                # Extract the comparison string
                comparison = str(token).strip()
                
                # Regex to split the comparison into left, operator, and right
                match = re.match(r'(.+?)(=|<>|<=|>=|<|>)(.+)', comparison)
                if match:
                    left, operator, right = match.groups()
                    left = left.strip()
                    operator = operator.strip()
                    right = right.strip()
                    
                    # Remove surrounding quotes from strings
                    if right.startswith("'") and right.endswith("'"):
                        continue  # It's a string predicate; skip
                    if right.startswith('"') and right.endswith('"'):
                        continue  # It's a string predicate; skip
                    
                    # Check if the right side is a number
                    if is_number(right):
                        numeric_predicates.append(comparison)
            elif token.is_group:
                # Recursively handle sub-tokens
                extract_from_tokens(token.tokens)

    # Iterate through the tokens to find the WHERE clause
    for token in stmt.tokens:
        if isinstance(token, Where):
            extract_from_tokens(token.tokens)
            break  # Assuming only one WHERE clause

    return numeric_predicates


class PlanTreeDataset(Dataset):
    def __init__(self, plans : pd.DataFrame, train : pd.DataFrame, encoding, hist_file, mem_scaler, to_predict, mode, column_type, db_params):
        table_sample = self.get_table_sample(mode, column_type, db_params)
        self.table_sample = table_sample
        self.encoding = encoding
        self.hist_file = hist_file
        
        self.length = len(plans)
        # train = train.loc[json_df['id']]
        
        nodes = [plan['Plan'] for plan in plans]
        self.labels = [plan['peakmem'] for plan in plans]
        # print(f"self.labels: {self.labels}")
        # print(f"mem_scaler.center_: {mem_scaler.center_}")
        # print(f"mem_scaler.scale_: {mem_scaler.scale_}")
        # self.labels = torch.from_numpy(mem_scaler.normalize_labels(np.array(self.labels)))
        self.labels = mem_scaler.transform(np.array(self.labels).reshape(-1,1)).flatten()
        # print(f"self.labels: {self.labels}")
        
        # self.card_labels = torch.from_numpy(card_norm.normalize_labels(self.cards))
        # self.cost_labels = torch.from_numpy(cost_norm.normalize_labels(self.costs))
        
        # self.to_predict = to_predict
        # if to_predict == 'cost':
        #     self.gts = self.costs
        #     self.labels = self.cost_labels
        # elif to_predict == 'card':
        #     self.gts = self.cards
        #     self.labels = self.card_labels
        # elif to_predict == 'both': ## try not to use, just in case
        #     self.gts = self.costs
        #     self.labels = self.cost_labels
        # else:
        #     raise Exception('Unknown to_predict type')
            
        idxs = np.arange(len(nodes)).tolist()
        
    
        self.treeNodes = [] ## for mem collection
        # load collated_dicts from file if exists
        collated_dicts_file = os.path.join('data/tpcds_sf1', f'{mode}_collated_dicts.pkl')
        if os.path.exists(collated_dicts_file):
            with open(collated_dicts_file, 'rb') as f:
                self.collated_dicts = pickle.load(f)
            print('Loaded collated_dicts from file.')
        else:
            self.collated_dicts = [self.js_node2dict(i,node) for i,node in tqdm(zip(idxs, nodes), total = len(nodes), desc=mode)]
            # Save collated_dicts to file
            with open(collated_dicts_file, 'wb') as f:
                pickle.dump(self.collated_dicts, f)
                print('Saved collated_dicts to file.')

    def get_table_sample(self, mode, column_type, db_params):
        db_params_copy = db_params.copy()
        db_params_copy['database'] = 'tpcds_sample'
        import psycopg2
        conn = psycopg2.connect(**db_params_copy)
        cur = conn.cursor()

        data_dir = '/home/wuy/DB/pg_mem_data'
        tmp_data_dir = 'data/tpcds_sf1'
        dataset = 'tpcds_sf1'
        # load table_sample from file if exists
        table_sample_file = os.path.join(tmp_data_dir, f'{mode}_table_samples.pkl')
        if os.path.exists(table_sample_file):
            with open(table_sample_file, 'rb') as f:
                table_samples = pickle.load(f)
            print('Loaded table_samples from file.')
        else:
            with open(os.path.join(data_dir, dataset, f'{mode}_plans.json')) as f:
                plans = json.load(f)

            table_pattern = r'\"([a-zA-Z_]+)\"\.'
            column_pattern = r'\.\"([a-zA-Z_]+)\"'

            table_samples = []
            for plan in tqdm(plans):
                table_sample = {}
                predicates = extract_numeric_predicates(plan['sql'])
                # print(plan['sql'])
                for predicate in predicates:
                    try:
                        table_name = re.search(table_pattern, predicate).group(1)
                        column_name = re.search(column_pattern, predicate).group(1)
                        if column_type[table_name][column_name] == 'char':
                            continue
                        q = 'select sid from {} where {}'.format(table_name, predicate)
                        cur.execute(q)
                        sps = np.zeros(1000).astype('uint8')
                        sids = cur.fetchall()
                        sids = np.array(sids).squeeze()
                        if sids.size>1:
                            sps[sids] = 1
                        if table_name in table_sample:
                            table_sample[table_name] = table_sample[table_name] & sps
                        else:
                            table_sample[table_name] = sps
                    except Exception as e:
                        print(f"Error: {e}")
                # if len(table_sample) > 0:
                table_samples.append(table_sample)

            # Save table_samples to file
            with open(table_sample_file, 'wb') as f:
                pickle.dump(table_samples, f)
                print('Saved table_samples to file.')
        cur.close()
        conn.close()
        print(f"table_samples length: {len(table_samples)}")
        return table_samples


    def js_node2dict(self, idx, node):
        treeNode = self.traversePlan(node, idx, self.encoding)
        _dict = self.node2dict(treeNode)
        collated_dict = self.pre_collate(_dict)
        
        self.treeNodes.clear()
        del self.treeNodes[:]

        return collated_dict

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        
        return self.collated_dicts[idx], self.labels[idx]

    def old_getitem(self, idx):
        return self.dicts[idx], (self.cost_labels[idx], self.card_labels[idx])
      
    ## pre-process first half of old collator
    def pre_collate(self, the_dict, max_node = 30, rel_pos_max = 20):

        x = pad_2d_unsqueeze(the_dict['features'], max_node)
        N = len(the_dict['features'])
        attn_bias = torch.zeros([N+1,N+1], dtype=torch.float)
        
        edge_index = the_dict['adjacency_list'].t()
        if len(edge_index) == 0:
            shortest_path_result = np.array([[0]])
            path = np.array([[0]])
            adj = torch.tensor([[0]]).bool()
        else:
            adj = torch.zeros([N,N], dtype=torch.bool)
            adj[edge_index[0,:], edge_index[1,:]] = True
            
            shortest_path_result = floyd_warshall_rewrite(adj.numpy())
        
        rel_pos = torch.from_numpy((shortest_path_result)).long()

        
        attn_bias[1:, 1:][rel_pos >= rel_pos_max] = float('-inf')
        
        attn_bias = pad_attn_bias_unsqueeze(attn_bias, max_node + 1)
        rel_pos = pad_rel_pos_unsqueeze(rel_pos, max_node)

        heights = pad_1d_unsqueeze(the_dict['heights'], max_node)
        
        return {
            'x' : x,
            'attn_bias': attn_bias,
            'rel_pos': rel_pos,
            'heights': heights
        }


    def node2dict(self, treeNode):

        adj_list, num_child, features = self.topo_sort(treeNode)
        heights = self.calculate_height(adj_list, len(features))

        return {
            'features' : torch.FloatTensor(np.array(features)),
            'heights' : torch.LongTensor(heights),
            'adjacency_list' : torch.LongTensor(np.array(adj_list)),
          
        }
    
    def topo_sort(self, root_node):
#        nodes = []
        adj_list = [] #from parent to children
        num_child = []
        features = []

        toVisit = deque()
        toVisit.append((0,root_node))
        next_id = 1
        while toVisit:
            idx, node = toVisit.popleft()
#            nodes.append(node)
            features.append(node.feature)
            num_child.append(len(node.children))
            for child in node.children:
                toVisit.append((next_id,child))
                adj_list.append((idx,next_id))
                next_id += 1
        
        return adj_list, num_child, features
    
    def traversePlan(self, plan, idx, encoding): # bfs accumulate plan

        nodeType = plan['Node Type']
        typeId = encoding.encode_type(nodeType)
        card = None #plan['Actual Rows']
        filters, alias = formatFilter(plan)
        join = formatJoin(plan)
        joinId = encoding.encode_join(join)
        filters_encoded = encoding.encode_filters(filters, alias)
        
        root = TreeNode(nodeType, typeId, filters, card, joinId, join, filters_encoded)
        
        self.treeNodes.append(root)

        if 'Relation Name' in plan:
            root.table = plan['Relation Name']
            root.table_id = encoding.encode_table(plan['Relation Name'])
        root.query_id = idx
        
        root.feature = node2feature(root, encoding, self.hist_file, self.table_sample)
        #    print(root)
        if 'Plans' in plan:
            for subplan in plan['Plans']:
                subplan['parent'] = plan
                node = self.traversePlan(subplan, idx, encoding)
                node.parent = root
                root.addChild(node)
        return root

    def calculate_height(self, adj_list,tree_size):
        if tree_size == 1:
            return np.array([0])

        adj_list = np.array(adj_list)
        node_ids = np.arange(tree_size, dtype=int)
        node_order = np.zeros(tree_size, dtype=int)
        uneval_nodes = np.ones(tree_size, dtype=bool)

        parent_nodes = adj_list[:,0]
        child_nodes = adj_list[:,1]

        n = 0
        while uneval_nodes.any():
            uneval_mask = uneval_nodes[child_nodes]
            unready_parents = parent_nodes[uneval_mask]

            node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
            node_order[node2eval] = n
            uneval_nodes[node2eval] = False
            n += 1
        return node_order 



def node2feature(node, encoding, hist_file, table_sample):
    # type, join, filter123, mask123
    # 1, 1, 3x3 (9), 3
    # TODO: add sample (or so-called table)
    num_filter = len(node.filterDict['colId'])
    pad = np.zeros((3,3-num_filter))
    filts = np.array(list(node.filterDict.values())) #cols, ops, vals
    ## 3x3 -> 9, get back with reshape 3,3
    filts = np.concatenate((filts, pad), axis=1).flatten() 
    mask = np.zeros(3)
    mask[:num_filter] = 1
    type_join = np.array([node.typeId, node.join])
    
    hists = filterDict2Hist(hist_file, node.filterDict, encoding)


    # table, bitmap, 1 + 1000 bits
    table = np.array([node.table_id])
    # print(f"length of table_sample: {len(table_sample)}")
    # print(f"node.query_id: {node.query_id}")
    # print(f"node: {node}")
    # print(f"node.table_id: {node.table_id}")
    # print(f"node.table: {node.table}")
    # print(f"table_sample: {table_sample}")
    # print(f"table_sample[node.query_id]: {table_sample[node.query_id]}")
    if node.table_id == 0 or node.table not in table_sample[node.query_id]:
        sample = np.zeros(1000)
    else:
        sample = table_sample[node.query_id][node.table]
    
    #return np.concatenate((type_join,filts,mask))
    return np.concatenate((type_join, filts, mask, hists, table, sample))
