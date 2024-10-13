from sklearn.preprocessing import RobustScaler
import dgl
import collections
import numpy as np
import torch
from functools import reduce
from operator import mul
from enum import Enum

def add_numerical_scalers(feature_statistics):
    for k, v in feature_statistics.items():
        if v.get('type') == 'numeric':
            scaler = RobustScaler()
            scaler.center_ = v['center']
            scaler.scale_ = v['scale']
            feature_statistics[k]['scaler'] = scaler

def encode(column, plan_params, feature_statistics):

    if column == 'Workers Planned':
        value = plan_params.get(column, 0) # if column is not in plan_params, return 0
    else:
        print()
        print(f"plan_params {plan_params}")
        value = plan_params[column]
  
    if feature_statistics[column].get('type') == 'numeric':  # FeatureType(enum) has numeric and categorical
        enc_value = feature_statistics[column]['scaler'].transform(np.array([[value]])).item()
    elif feature_statistics[column].get('type') == 'categorical':
        value_dict = feature_statistics[column]['value_dict']
        enc_value = value_dict.get(str(value), 0)  # wuy: I think value is always str if featuretype is categorical.
    else:
        raise NotImplementedError
    return enc_value



def parse_predicates(db_column_features, feature_statistics, filter_column, filter_to_plan_edges, plan_featurization,
                     predicate_col_features, predicate_depths, intra_predicate_edges, logical_preds, plan_node_id=None,
                     parent_filter_node_id=None, depth=0):
    """
    Recursive parsing of predicate columns

    :param db_column_features:
    :param feature_statistics:
    :param filter_column:
    :param filter_to_plan_edges:
    :param plan_featurization:
    :param plan_node_id:
    :param predicate_col_features:
    :return:
    """
    filter_node_id = len(predicate_depths)
    predicate_depths.append(depth)

    # print(f"########### filter_column type {type(filter_column)}")   # types.SimpleNamespace
    # gather features
    # print(f"filter_column {filter_column}")
    
    # arithmetic operator
    if filter_column.operator in {str(op) for op in list(Operator)}:  # NEQ = '!=', EQ = '=', LEQ = '<=', GEQ = '>=', LIKE = 'LIKE', NOT_LIKE = 'NOT LIKE', IS_NOT_NULL = 'IS NOT NULL', IS_NULL = 'IS NULL', IN = 'IN', BETWEEN = 'BETWEEN'
        curr_filter_features = [encode(feature_name, vars(filter_column), feature_statistics)
                                for feature_name in plan_featurization.FILTER_FEATURES]  # FILTER_FEATURES = ['operator', 'literal_feature']

        if filter_column.column is not None:
            curr_filter_col_feats = [
                encode(column, vars(db_column_features[filter_column.column]), feature_statistics)
                for column in plan_featurization.COLUMN_FEATURES]  # COLUMN_FEATURES = ['avg_width', 'correlation', 'data_type', 'n_distinct', 'null_frac']
        # hack for cases in which we have no base filter column (e.g., in a having clause where the column is some
        # result column of a subquery/groupby). In the future, this should be replaced by some graph model that also
        # encodes the structure of this output column
        else:
            curr_filter_col_feats = [0 for _ in plan_featurization.COLUMN_FEATURES]
        curr_filter_features += curr_filter_col_feats
        logical_preds.append(False)
    # logical operator
    else:  # AND, OR
        curr_filter_features = [encode(feature_name, vars(filter_column), feature_statistics)
                                for feature_name in plan_featurization.FILTER_FEATURES]  # FILTER_FEATURES: ['operator', 'literal_feature']+
        logical_preds.append(True)
    # print(f"curr_filter_features {curr_filter_features}")
    predicate_col_features.append(curr_filter_features)

    # add edge either to plan or inside predicates
    if depth == 0:
        assert plan_node_id is not None
        # in any case add the corresponding edge
        filter_to_plan_edges.append((filter_node_id, plan_node_id))

    else:
        assert parent_filter_node_id is not None
        intra_predicate_edges.append((filter_node_id, parent_filter_node_id))

    # recurse
    for c in filter_column.children:
        parse_predicates(db_column_features, feature_statistics, c, filter_to_plan_edges,
                         plan_featurization, predicate_col_features, predicate_depths, intra_predicate_edges,
                         logical_preds, parent_filter_node_id=filter_node_id, depth=depth + 1)

def handle_children_product_card(node):
    """
    Computes the product of the cardinalities of the children of a node.

    :param node:
    :return:
    """
    children = node.get('Plans', [])
    plan_rows = [child.get('Plan Rows', 1) for child in children]
    product = reduce(mul, plan_rows, 1)
    node['Children Rows Product'] = product


def plan_to_graph(node, plan_depths, plan_features, plan_to_plan_edges, db_statistics, feature_statistics,
                  filter_to_plan_edges, predicate_col_features, output_column_to_plan_edges, output_column_features,
                  column_to_output_column_edges, column_features, table_features, table_to_plan_edges,
                  output_column_idx, column_idx, table_idx, predicate_depths, intra_predicate_edges,
                  logical_preds, parent_node_id=None, depth=0):
    plan_node_id = len(plan_depths) # plan_node_id is auto incremental
    plan_depths.append(depth)

    node = node.get('Plan', node)

    # encode() normalizes a numerical value and transforms a categorical value to its index, as defined in the feature_statistics file.
    curr_plan_features = [encode(column, node, feature_statistics) for column in
                          ['Plan Rows', 'Plan Width', 'Workers Planned', 'Node Type', 'Children Rows Product']]  # PostgresEstSystemCardDetail.PLAN_FEATURES = ['est_card', 'est_width', 'workers_planned', 'op_name', 'est_children_card']
    plan_features.append(curr_plan_features)
    print(f"##### curr_plan_features:\n{curr_plan_features}")  # [-0.012496119994793284, 4.666666666666667, -2.0, 6, 0.0]
    while 1:pass
    # encode output columns which can in turn have several columns as a product in the aggregation
    output_columns = plan_params.get('output_columns')  # example output:  plan['plan_parameters']['output_columns']: [{'aggregation': 'AVG', 'columns': [23]}, {'aggregation': 'AVG', 'columns': [21]}]
    if output_columns is not None:
        for output_column in output_columns:
            # print(f"##### output_column: {output_column}")
            output_column_node_id = output_column_idx.get(
                (output_column.aggregation, tuple(output_column.columns), database_id))

            # if not, create
            if output_column_node_id is None:
                curr_output_column_features = [encode(column, vars(output_column), feature_statistics)
                                               for column in plan_featurization.OUTPUT_COLUMN_FEATURES]  # OUTPUT_COLUMN_FEATURES = ['aggregation']
                # print(f"##### curr_output_column_features: {curr_output_column_features}")  # sum=2, None=3 as defined in aggregation in feature_statistics
                output_column_node_id = len(output_column_features)
                output_column_features.append(curr_output_column_features)
                output_column_idx[(output_column.aggregation, tuple(output_column.columns), database_id)] \
                    = output_column_node_id   # output_column_idx is a dict

                # featurize product of columns if there are any
                db_column_features = db_statistics[database_id].column_stats  # each database_id acturally refers to a workload_run json file as defined in read_workload_run function
                for column in output_column.columns:
                    column_node_id = column_idx.get((column, database_id))
                    if column_node_id is None:
                        curr_column_features = [
                            encode(feature_name, vars(db_column_features[column]), feature_statistics)
                            for feature_name in plan_featurization.COLUMN_FEATURES]  # COLUMN_FEATURES = ['avg_width', 'correlation', 'data_type', 'n_distinct', 'null_frac']
                        column_node_id = len(column_features)
                        column_features.append(curr_column_features)
                        column_idx[(column, database_id)] = column_node_id
                    column_to_output_column_edges.append((column_node_id, output_column_node_id))

            # in any case add the corresponding edge
            output_column_to_plan_edges.append((output_column_node_id, plan_node_id))

    # filter_columns (we do not reference the filter columns to columns since we anyway have to create a node per filter node)
    filter_column = plan_params.get('filter_columns') 
    if filter_column is not None:
        db_column_features = db_statistics[database_id].column_stats

        # check if node already exists in the graph
        # filter_node_id = fitler_node_idx.get((filter_column.operator, filter_column.column, database_id))

        parse_predicates(db_column_features, feature_statistics, filter_column, filter_to_plan_edges,
                         plan_featurization, predicate_col_features, predicate_depths, intra_predicate_edges,
                         logical_preds, plan_node_id=plan_node_id)  # parent_filter_node_id=None, depth=0

    # tables
    table = plan_params.get('table')
    if table is not None:
        table_node_id = table_idx.get((table, database_id))
        db_table_statistics = db_statistics[database_id].table_stats

        if table_node_id is None:
            curr_table_features = [encode(feature_name, vars(db_table_statistics[table]), feature_statistics)
                                   for feature_name in plan_featurization.TABLE_FEATURES]  # ['reltuples', 'relpages']
            table_node_id = len(table_features)
            table_features.append(curr_table_features)
            table_idx[(table, database_id)] = table_node_id

        table_to_plan_edges.append((table_node_id, plan_node_id))

    # add edge to parent
    if parent_node_id is not None:
        plan_to_plan_edges.append((plan_node_id, parent_node_id))

    # continue recursively
    for c in node.children:
        plan_to_graph(c, plan_depths, plan_features, plan_to_plan_edges, db_statistics, feature_statistics,
                      filter_to_plan_edges, predicate_col_features, output_column_to_plan_edges, output_column_features,
                      column_to_output_column_edges, column_features, table_features, table_to_plan_edges,
                      output_column_idx, column_idx, table_idx, predicate_depths,
                      intra_predicate_edges, logical_preds, parent_node_id=plan_node_id, depth=depth + 1)
        

def plan_collator(plans, feature_statistics, db_statistics):
    plan_depths = []
    plan_features = []
    plan_to_plan_edges = []
    filter_to_plan_edges = []
    filter_features = []
    output_column_to_plan_edges = []
    output_column_features = []
    column_to_output_column_edges = []
    column_features = []
    table_features = []
    table_to_plan_edges = []
    labels = []
    predicate_depths = []
    intra_predicate_edges = []
    logical_preds = []

    output_column_idx = dict()
    column_idx = dict()
    table_idx = dict()

    # prepare robust encoder for the numerical fields
    add_numerical_scalers(feature_statistics)

    # iterate over plans and create lists of edges and features per node
    sample_idxs = []
    for p in plans:

        # labels.append(p.plan_runtime)
        labels.append(p['peakmem'])
        
        plan_to_graph(p, plan_depths, plan_features, plan_to_plan_edges, db_statistics,
                      feature_statistics, filter_to_plan_edges, filter_features, output_column_to_plan_edges,
                      output_column_features, column_to_output_column_edges, column_features, table_features,
                      table_to_plan_edges, output_column_idx, column_idx, table_idx,
                      predicate_depths, intra_predicate_edges, logical_preds)
    assert len(labels) == len(plans)
    assert len(plan_depths) == len(plan_features)

    data_dict, nodes_per_depth, plan_dict = create_node_types_per_depth(plan_depths, plan_to_plan_edges)  
    # data_dict[(f'plan{d_u}', f'intra_plan', f'plan{d_v}')].append((u_node_id, v_node_id))

    # similarly create node types:
    #   pred_node_{depth}, filter column
    pred_dict = dict()
    nodes_per_pred_depth = collections.defaultdict(int)
    no_filter_columns = 0
    for pred_node, d in enumerate(predicate_depths):
        # predicate node
        if logical_preds[pred_node]:
            pred_dict[pred_node] = (nodes_per_pred_depth[d], d)
            nodes_per_pred_depth[d] += 1
        # filter column
        else:
            pred_dict[pred_node] = no_filter_columns
            no_filter_columns += 1

    adapt_predicate_edges(data_dict, filter_to_plan_edges, intra_predicate_edges, logical_preds, plan_dict, pred_dict, pred_node_type_id)  # pred_node_type_id is a function

    # we additionally have filters, tables, columns, output_columns and plan nodes as node types
    data_dict[('column', 'col_output_col', 'output_column')] = column_to_output_column_edges
    for u, v in output_column_to_plan_edges:
        v_node_id, d_v = plan_dict[v]
        data_dict[('output_column', 'to_plan', f'plan{d_v}')].append((u, v_node_id))
    for u, v in table_to_plan_edges:
        v_node_id, d_v = plan_dict[v]
        data_dict[('table', 'to_plan', f'plan{d_v}')].append((u, v_node_id))

    # also pass number of nodes per type
    max_depth, max_pred_depth = get_depths(plan_depths, predicate_depths)
    num_nodes_dict = {
        'column': len(column_features),
        'table': len(table_features),
        'output_column': len(output_column_features),
        'filter_column': len(logical_preds) - sum(logical_preds),
    }
    # print(f"column_feature {len(column_features[0])}, table_feature {len(table_features[0])}, output_column_feature {len(output_column_features[0])}")
    # previous line outputs: column_feature 5, table_feature 2, output_column_feature 1

    # update 'plan{d}' and 'logical_pred_{d}' and filter out any node type with zero nodes
    num_nodes_dict = update_node_counts(max_depth, max_pred_depth, nodes_per_depth, nodes_per_pred_depth,
                                        num_nodes_dict) 

    # create graph
    graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict) # idtype, device
    graph.max_depth = max_depth
    graph.max_pred_depth = max_pred_depth

    features = collections.defaultdict(list)
    features.update(dict(column=column_features, table=table_features, output_column=output_column_features,
                         filter_column=[f for f, log_pred in zip(filter_features, logical_preds) if not log_pred]))
    # sort the plan features based on the depth
    for u, plan_feat in enumerate(plan_features):
        u_node_id, d_u = plan_dict[u]
        features[f'plan{d_u}'].append(plan_feat)

    # sort the predicate features based on the depth
    for pred_node_id, pred_feat in enumerate(filter_features):
        if not logical_preds[pred_node_id]:
            continue
        node_type, _ = pred_node_type_id(logical_preds, pred_dict, pred_node_id)  # node_type = f'logical_pred_{depth}'
        features[node_type].append(pred_feat)

    features = postprocess_feats(features, num_nodes_dict)

    # rather deal with runtimes in secs
    labels = postprocess_labels(labels)  # original plan['plan_runtime'] is in miliseconds

    # print(f"graph {graph}\nfeatures {features}\nlabels {labels}\nsample_idxs {sample_idxs}")
    # features: dict_keys(['column', 'table', 'output_column', 'filter_column', 'plan0', 'plan1', 'plan2', 'plan3', 'plan4', 'plan5', 'plan6', 'plan7', 'plan8', 'plan9', 'logical_pred_0'])
    # print(f"column {features['column'].shape}, table {features['table'].shape}, output_column {features['output_column'].shape}, filter_column {features['filter_column'].shape}, plan0 {features['plan0'].shape}, plan1 {features['plan1'].shape}, plan2 {features['plan2'].shape}, plan3 {features['plan3'].shape}, plan4 {features['plan4'].shape}, plan5 {features['plan5'].shape}, plan6 {features['plan6'].shape}, plan7 {features['plan7'].shape}, plan8 {features['plan8'].shape}, plan9 {features['plan9'].shape}, logical_pred_0 {features['logical_pred_0'].shape}")
    # print(f"column {features['column'].shape}, table {features['table'].shape}, output_column {features['output_column'].shape}, filter_column {features['filter_column'].shape}, plan0 {features['plan0'].shape}, logical_pred_0 {features['logical_pred_0'].shape}")
    # previous line outputs: column torch.Size([23, 5]), table torch.Size([4, 2]), output_column torch.Size([24, 1]), filter_column torch.Size([7, 7]), plan0 torch.Size([1, 5]), logical_pred_0 torch.Size([2, 2])
    return graph, features, labels, sample_idxs