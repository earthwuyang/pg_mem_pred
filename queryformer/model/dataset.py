# model/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import pandas as pd
from collections import deque
from .database_util import formatFilter, formatJoin, TreeNode, filterDict2Hist
from .database_util import *

def process_node(args):
            idx, node, obj = args
            return obj.js_node2dict(idx, node)

class PlanTreeDataset(Dataset):
    def __init__(self, data_dir, dataset, mode, alias2t, t2alias, schema, sample_dir, DB_PARAMS, encoding, max_workers):
        self.data_dir = data_dir
        self.dataset = dataset
        self.mode = mode
        self.alias2t = alias2t
        self.t2alias = t2alias
        self.schema = schema
        self.sample_dir = sample_dir
        self.DB_PARAMS = DB_PARAMS
        self.encoding = encoding

        plan_file = os.path.join(data_dir, dataset, f'{mode}_plans.json')
        
        with open(plan_file, 'r') as f:
            plans = json.load(f)
        plans = plans[:100]
        generated_queries_path = f'./data/{dataset}/{mode}_generated_queries.csv'
        

        if os.path.exists(generated_queries_path):
            logging.info(f"Generated CSV file '{generated_queries_path}' already exists. Skipping generation.")
        else:
            generate_for_samples(plans, generated_queries_path, alias2t)
            logging.info(f"Generated CSV file for training queries saved to: {generated_queries_path}")


        # Load all queries from the generated CSV
        column_names = ['tables', 'joins', 'predicate', 'cardinality']
        try:
            query_file = pd.read_csv(
                generated_queries_path,
                sep='#',
                header=None,
                names=column_names,
                keep_default_na=False,   # Do not convert empty strings to NaN
                na_values=['']           # Treat empty strings as empty, not NaN
            )

        except pd.errors.ParserError as e:
            logging.error(f"Error reading generated_queries.csv: {e}")
            exit(1)

        # Generate bitmaps for each query based on pre-sampled table data
        logging.info("Generating table sample bitmaps for each query.")

        self.sampled_data = generate_query_bitmaps(
            query_file=query_file,
            alias2t=alias2t,
            sample_dir=sample_dir
        )


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
                max_workers=max_workers
            )
            # Save histograms with comma-separated bins
            save_histograms(hist_file_df, save_path=histogram_file_path)
        else:
            hist_file_df = load_entire_histograms(load_path=histogram_file_path)
        
        self.hist_file = hist_file_df
        self.length = len(plans)

        nodes = [plan['Plan'] for plan in plans]
        self.labels = [plan['peakmem'] for plan in plans]
    
        
        idxs = np.arange(self.length).tolist()
        # print(f"idxs length {len(idxs)}")
        # print(f"self.sampled_data length {len(self.sampled_data)}")

        self.treeNodes = []

        

        logging.info(f"traversing tree and getting collated dicts for {len(nodes)} plans")
        # load if collated_dicts in file else process
        collated_dicts_file = f'./data/{dataset}/{mode}_collated_dicts.npy'
        if os.path.exists(collated_dicts_file):
            logging.info(f"Loading collated_dicts from {collated_dicts_file}")
            self.collated_dicts = np.load(collated_dicts_file, allow_pickle=True)
            logging.info(f"Loaded collated_dicts from {collated_dicts_file}")
        else:
            self.collate(nodes, idxs)
            np.save(collated_dicts_file, self.collated_dicts)
            logging.info(f"Saved collated_dicts to {collated_dicts_file}")

    def collate(self, nodes, idxs):
        # Wrap it with tqdm and multiprocessing
        # with mp.Pool(processes=mp.cpu_count()) as pool:
        # with mp.Pool(processes=1) as pool:
        #     self.collated_dicts = list(tqdm(pool.imap(process_node, [(i, node, self) for i, node in zip(idxs, nodes)]), total=len(nodes)))
        # self.collated_dicts = list(tqdm(pool.imap(process_node, [(i, node, self) for i, node in zip(idxs, nodes)]), total=len(nodes)))
        self.collated_dicts = [process_node((i, node, self)) for i, node in tqdm(zip(idxs, nodes), total=len(nodes))]
        
        
    
    def js_node2dict(self, idx, node):
        treeNode = self.traversePlan(node, idx, self.encoding)
        _dict = self.node2dict(treeNode)
        collated_dict = self.pre_collate(_dict)
        return collated_dict
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.collated_dicts[idx], self.labels[idx]
    
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
            shortest_path_result = compute_shortest_paths_bfs_numba(adj.numpy())
        
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
        # logging.info(f"encoding.type2idx: {len(encoding.type2idx)}, typeId: {typeId}")
        card = None # plan['Actual Rows'] if needed
        filters, alias = formatFilter(plan)
        join = formatJoin(plan)
        joinId = encoding.encode_join(join)
        # logging.info(f"encoding.join2idx: {len(encoding.join2idx)}, joinId: {joinId}")
        if joinId >= len(encoding.join2idx):
            print(f"joinId: {joinId}, encoding.join2idx: {len(encoding.join2idx)}")
        filters_encoded = encoding.encode_filters(filters, alias)
        
        root = TreeNode(nodeType, typeId, filters, card, joinId, join, filters_encoded)
        
        if 'Relation Name' in plan:
            root.table = plan['Relation Name']
            root.table_id = encoding.encode_table(plan['Relation Name'])
        root.query_id = idx

        root.feature = node2feature(root, encoding, self.hist_file, self.sampled_data[root.query_id], self.alias2t)  # Pass the correct sample data
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
    filter_cols = np.atleast_1d(node.filterDict.get('colId', []))
    filter_ops = np.atleast_1d(node.filterDict.get('opId', []))
    filter_vals = np.atleast_1d(node.filterDict.get('val', []))

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

    # Pad filters to fixed size
    max_filters = MAX_FILTERS
    filter_pad_length = max_filters - num_filters
    pad_value_col = encoding.col2idx.get('NA', 0)
    pad_value_op = encoding.op2idx.get('NA', 3)

    if len(filter_cols) == 0:
        print(f"filter_cols is empty")
        while 1:pass
    if len(filter_ops) == 0:    
        print(f"filter_ops is empty")
        while 1:pass
    if len(filter_vals) == 0:
        print(f"filter_vals is empty")
        while 1:pass
    
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

    # Concatenate all features into one feature vector
    feature_vector = np.concatenate([type_join, filters, filter_mask, hists, table_id, sample])

    # Ensure feature vector length is correct
    expected_length = 2 + (3 * MAX_FILTERS) + MAX_FILTERS + (HIST_BINS * MAX_FILTERS) + 1 + SAMPLE_SIZE
    actual_length = len(feature_vector)
    assert actual_length == expected_length, f"Feature vector length {actual_length} does not match expected {expected_length}."

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
