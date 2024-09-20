### pkl files
edge_src_nodes_list.pkl  edge_tgt_nodes_list.pkl  features_list.pkl  memory_scaler.pkl  num_nodes_list.pkl  peakmem_list.pkl
These pkl files are used for the dataset loading.

### Because the process of traversing trees is slow, we use process_plan_for_dgl.py to generate those pkl files. Loading the pkl files is faster than loading the json plan and traversing the tree and generating edge_src_nodes, edge_tgt_nodes, features, and peakmem.