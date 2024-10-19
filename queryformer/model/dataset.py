# model/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import pandas as pd
from collections import deque
from .database_util import formatFilter, formatJoin, TreeNode, filterDict2Hist
from .database_util import *

class PlanTreeDataset(Dataset):
    def __init__(self, json_df: pd.DataFrame, workload_df: pd.DataFrame, encoding, hist_file, table_sample, alias2t):
        self.table_sample = table_sample  # List of dicts indexed by query_id
        self.encoding = encoding
        self.hist_file = hist_file
        self.length = len(json_df)
        self.alias2t = alias2t
        
        nodes = [plan['Plan'] for plan in json_df]
        self.labels = [plan['peakmem'] for plan in json_df]
    
        
        idxs = np.arange(self.length).tolist()


        self.treeNodes = []
        self.collated_dicts = [self.js_node2dict(i, node) for i, node in zip(idxs, nodes)]
    
    def js_node2dict(self, idx, node):
        treeNode = self.traversePlan(node, idx, self.encoding)
        _dict = self.node2dict(treeNode)
        collated_dict = self.pre_collate(_dict)
        return collated_dict
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.collated_dicts[idx], (self.cost_labels[idx], self.card_labels[idx])
    
    def pre_collate(self, the_dict, max_node=30, rel_pos_max=20):
        x = pad_2d_unsqueeze(the_dict['features'], max_node)
        N = len(the_dict['features'])
        attn_bias = torch.zeros([N+1, N+1], dtype=torch.float)
        
        edge_index = the_dict['adjacency_list'].t()
        if len(edge_index) == 0:
            shortest_path_result = np.array([[0]])
            path = np.array([[0]])
            adj = torch.tensor([[0]]).bool()
        else:
            adj = torch.zeros([N, N], dtype=torch.bool)
            adj[edge_index[0, :], edge_index[1, :]] = True
            shortest_path_result = floyd_warshall_rewrite(adj.numpy())
        
        rel_pos = torch.from_numpy(shortest_path_result).long()
        attn_bias[1:, 1:][rel_pos >= rel_pos_max] = float('-inf')
        attn_bias = pad_attn_bias_unsqueeze(attn_bias, max_node + 1)
        rel_pos = pad_rel_pos_unsqueeze(rel_pos, max_node)
        heights = pad_1d_unsqueeze(the_dict['heights'], max_node)
        
        return {
            'x': x,
            'attn_bias': attn_bias,
            'rel_pos': rel_pos,
            'heights': heights
        }
    
    def node2dict(self, treeNode):
        '''
        Converts a tree node into a structured dictionary format
        '''

        adj_list, num_child, features = self.topo_sort(treeNode)
        heights = self.calculate_height(adj_list, len(features))

        return {
            'features' : torch.FloatTensor(np.array(features)),
            'heights' : torch.LongTensor(heights),
            'adjacency_list' : torch.LongTensor(np.array(adj_list)),
          
        }
    
    def topo_sort(self, root_node):
        '''
        Performs a topological sort on the tree to get the adjacency list and features
        '''
        adj_list = [] # from parent to children
        num_child = []
        features = []

        # initialize a deque for BFS traversal and add the root node to visit, starting with an index of 0 for the root
        toVisit = deque()
        toVisit.append((0, root_node))
        next_id = 1

        # Loops while there are nodes to visit, popping the next node from the deque
        while toVisit:
            idx, node = toVisit.popleft()
            features.append(node.feature)
            num_child.append(len(node.children))

            # iterate through the children of the current node, 
            # add them to the `toVisit` deque, and build the adjacency list, assigning a new ID for each child
            for child in node.children:
                toVisit.append((next_id, child))
                adj_list.append((idx, next_id))
                next_id += 1
        
        return adj_list, num_child, features
    
    def traversePlan(self, plan, idx, encoding): # bfs accumulate plan
        '''
        Recursively constructs a tree structure from a JSON query plan
        '''

        nodeType = plan['Node Type']
        typeId = encoding.encode_type(nodeType)
        card = None # plan['Actual Rows'] if needed
        filters, alias = formatFilter(plan)
        join = formatJoin(plan)
        joinId = encoding.encode_join(join)
        filters_encoded = encoding.encode_filters(filters, alias)
        
        root = TreeNode(nodeType, typeId, filters, card, joinId, join, filters_encoded)
        
        if 'Relation Name' in plan:
            root.table = plan['Relation Name']
            root.table_id = encoding.encode_table(plan['Relation Name'])
        root.query_id = idx
        
        root.feature = node2feature(root, encoding, self.hist_file, self.table_sample[root.query_id], self.alias2t)  # Pass the correct sample data
        if 'Plans' in plan:
            for subplan in plan['Plans']:
                subplan['parent'] = plan
                node = self.traversePlan(subplan, idx, encoding)
                node.parent = root
                root.addChild(node)
        return root

    def calculate_height(self, adj_list, tree_size):
        if tree_size == 1:
            return np.array([0])

        adj_list = np.array(adj_list)
        node_ids = np.arange(tree_size, dtype=int)
        node_order = np.zeros(tree_size, dtype=int)
        uneval_nodes = np.ones(tree_size, dtype=bool)

        # Separate the parent and child nodes from the adjacency list
        parent_nodes = adj_list[:,0]
        child_nodes = adj_list[:,1]

        # loops while there are nodes that haven't been evaluated,
        n = 0
        while uneval_nodes.any():
            # creating a mask for unevaluated child nodes and a mask for unready parents
            uneval_mask = uneval_nodes[child_nodes]
            unready_parents = parent_nodes[uneval_mask]

            # determine which nodes can be evaluated (i.e. nodes whose parents have been evaluated),
            # update the node order, and mark the nodes as evaluated, and increment the height counter
            node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
            node_order[node2eval] = n
            uneval_nodes[node2eval] = False
            n += 1

        # return the calculated heights for each node based on their order of evaluation
        return node_order 


def node2feature(node, encoding, hist_file, table_sample, alias2t):
    # Type and join IDs
    type_join = np.array([node.typeId, node.join])

    # Filters
    filter_cols = node.filterDict.get('colId', [])
    filter_ops = node.filterDict.get('opId', [])
    filter_vals = node.filterDict.get('val', [])
    num_filters = len(filter_cols)

    # Map colIds and opIds back to column names and operators for histogram
    mapped_filterDict = {}
    for colId, opId, val in zip(filter_cols, filter_ops, filter_vals):
        col = encoding.idx2col.get(colId, 'NA')
        op = encoding.idx2op.get(opId, 'NA')

        if '.' in col:
            alias, column = col.split('.', 1)
            table = alias2t.get(alias)
            if table:
                table_column = f"{table}.{column}"
                condition = {'op': op, 'value': val}
                mapped_filterDict[table_column] = condition
            else:
                logging.warning(f"Alias '{alias}' not found in alias2t mapping. Skipping column '{col}'.")
        else:
            # Handle columns without alias
            potential_tables = [table for table in alias2t.values() if f"{table}.{col}" in hist_file['table_column'].values]
            if len(potential_tables) == 1:
                table = potential_tables[0]
                table_column = f"{table}.{col}"
                condition = {'op': op, 'value': val}
                mapped_filterDict[table_column] = condition
            elif len(potential_tables) > 1:
                logging.warning(f"Ambiguous column '{col}' found in multiple tables {potential_tables}. Skipping.")
            else:
                # logging.warning(f"Column '{col}' not found in any table. Skipping.")
                pass

    # Histograms
    hists = filterDict2Hist(hist_file, mapped_filterDict, encoding)

    # Prepare filters for inclusion in the feature vector
    # Pad filters to fixed size
    max_filters = MAX_FILTERS
    filter_pad_length = max_filters - num_filters
    pad_value_col = encoding.col2idx.get('NA', 0)
    pad_value_op = encoding.op2idx.get('NA', 3)  # Updated to match new op2idx

    # Ensure filter arrays are numpy arrays
    filter_cols = np.array(filter_cols)
    filter_ops = np.array(filter_ops)
    filter_vals = np.array(filter_vals)

    filter_cols = np.pad(filter_cols, (0, filter_pad_length), 'constant', constant_values=pad_value_col)[:max_filters]
    filter_ops = np.pad(filter_ops, (0, filter_pad_length), 'constant', constant_values=pad_value_op)[:max_filters]
    filter_vals = np.pad(filter_vals, (0, filter_pad_length), 'constant', constant_values=0.0)[:max_filters]

    # Create filter mask
    filter_mask = np.zeros(max_filters)
    filter_mask[:min(num_filters, max_filters)] = 1

    # Concatenate filters into a single array
    filters = np.concatenate([filter_cols, filter_ops, filter_vals])

    # Table ID
    table_id = np.array([node.table_id])

    # Sample data
    sample = table_sample.get(node.table, np.zeros(SAMPLE_SIZE))

    # Ensure sample is of fixed size
    if len(sample) < SAMPLE_SIZE:
        sample = np.pad(sample, (0, SAMPLE_SIZE - len(sample)), 'constant')
    else:
        sample = sample[:SAMPLE_SIZE]

    # Concatenate all features
    feature_vector = np.concatenate([type_join, filters, filter_mask, hists, table_id, sample])

    # Debugging: Check the maximum opId
    max_op_id = filter_ops.max() if len(filter_ops) > 0 else 0
    # logging.debug(f"Max opId in node.feature: {max_op_id}")
    if max_op_id >= len(encoding.op2idx):
        logging.error(f"opId {max_op_id} exceeds embedding size {len(encoding.op2idx)}. Assigning 'NA'.")
        exit()
        # Optionally, set opId to 'NA' here if needed

    return feature_vector




def pad_1d_unsqueeze(x, padlen):
    x = x + 1 # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    # Don't know why add 1, kept as per original
    x = x + 1 # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype) + 1
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_rel_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)

def collator(small_set):
    """
    Collates a list of samples into a Batch object.
    """
    batch_dicts = [s[0] for s in small_set]
    y = [s[1] for s in small_set]
    xs = [s['x'] for s in batch_dicts]
    attn_bias = [s['attn_bias'] for s in batch_dicts]
    rel_pos = [s['rel_pos'] for s in batch_dicts]
    heights = [s['heights'] for s in batch_dicts]
    
    # Concatenate tensors
    x = torch.cat(xs, dim=0)
    attn_bias = torch.cat(attn_bias, dim=0)
    rel_pos = torch.cat(rel_pos, dim=0)
    heights = torch.cat(heights, dim=0)
    
    return Batch(attn_bias, rel_pos, heights, x), y

class Batch():
    def __init__(self, attn_bias, rel_pos, heights, x, y=None):
        super(Batch, self).__init__()

        self.heights = heights
        self.x, self.y = x, y
        self.attn_bias = attn_bias
        self.rel_pos = rel_pos
        
    def to(self, device):

        self.heights = self.heights.to(device)
        self.x = self.x.to(device)

        self.attn_bias, self.rel_pos = self.attn_bias.to(device), self.rel_pos.to(device)

        return self
