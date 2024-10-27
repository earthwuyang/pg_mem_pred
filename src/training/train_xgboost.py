import json
import os
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from .metrics import compute_metrics

def load_plans(file_path):
    """
    Load parsed query plans from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        list: A list of parsed query plans.
    """
    with open(file_path, 'r') as f:
        plans = json.load(f)
    return plans

def one_hot_encode(n, num_classes):
    """
    One-hot encode a number into a binary vector of length num_classes.
    
    Args:
        n (int): The number to encode.
        num_classes (int): The number of classes.
        
    Returns:
        one_hot (list): A binary vector of length num_classes.
    """
    one_hot = [0] * num_classes
    one_hot[n] = 1
    return one_hot

def extract_features(plan, statistics):
    """
    Extract features from a parsed PostgreSQL query plan.

    Args:
        plan (dict): The parsed query plan.

    Returns:
        dict: A dictionary of aggregated features.
    """
    features = defaultdict(float)

    def traverse(plan_node):
        if 'Plan' in plan_node:
            plan_node = plan_node.get('Plan', {})
        else:
            plan_node = plan_node

        for key in ['Startup Cost', 'Total Cost', 'Plan Rows', 'Plan Width', 'Node Type']:
            if statistics[key]['type'] == 'numeric':
                value = ( plan_node[key] - statistics[key]['center']) / statistics[key]['scale']
                features[key] = value
            elif statistics[key]['type'] == 'categorical':
                for value in statistics[key]['value_dict'].values():
                    features[value] = 0
                if key in plan_node:
                    features[plan_node[key]] = 1
    
        # Traverse children
        if 'Plans' in plan_node:
            for child in plan_node['Plans']:
                traverse(child)

    # Traverse the plan and collect features
    traverse(plan)
    return features




def prepare_dataset(logger, data_dir, datasets, mode, statistics, debug, mem_pred, time_pred, not_cross_datasets):
    """
    Prepare a dataset by extracting features and collecting labels.

    Args:
        plans (list): A list of parsed query plans.

    Returns:
        pd.DataFrame: DataFrame containing features.
        pd.Series: Series containing labels (peak memory).
    """
    plans = []
    if not isinstance(datasets, list):
        datasets = [datasets]
    for ds in tqdm(datasets, desc=f"Loading {mode} plans"):
        if mode != 'test':
            plan_file = os.path.join(data_dir, ds, f'total_plans.json')
        else:
            if not_cross_datasets:
                plan_file = os.path.join(data_dir, ds, f'{mode}_plans.json')
            else:
                plan_file = os.path.join(data_dir, ds, f'total_plans.json')
        if debug:
            plan_file = os.path.join(data_dir, 'tpch_sf1', 'tiny_plans.json')
        plan = load_plans(plan_file)
        plans.extend(plan)
    logger.info(f"number of {mode} plans: {len(plans)}")

    # Extract features and labels
    feature_dicts = []
    labels = []

    for plan in tqdm(plans):
        features = extract_features(plan, statistics)
        feature_dicts.append(features)
        if mem_pred:
            label = plan.get('peakmem', 0.0)
        elif time_pred:
            label = plan.get('time', 0.0)
  
        labels.append(label)  

    # Convert to DataFrame
    df_features = pd.DataFrame(feature_dicts)
    df_labels = pd.Series(labels, name='peakmem')

    # Handle missing values if any
    df_features.fillna(0, inplace=True)

    return df_features, df_labels


def collate(df, columns):
    """
    Collate a DataFrame by selecting only the specified columns.

    Args:
        df (pd.DataFrame): The DataFrame to collate.
        columns (list): A list of columns to select.

    Returns:
        pd.DataFrame: The collated DataFrame.
    """
    missing_cols = set(columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    df = df[columns]
    return df

def train_XGBoost(logger, args, combined_stats):

    not_cross_datasets = isinstance(args.train_dataset, str)

    column_names_file = os.path.join('data', 'xgb_column_names.pickle')

    if not args.skip_train:
        # Prepare training and validation datasets
        X_train, y_train = prepare_dataset(logger, args.data_dir, args.train_dataset, 'train', combined_stats, args.debug, args.mem_pred, args.time_pred, not_cross_datasets)
        column_names = X_train.columns
        with open(column_names_file, 'wb') as f:
            pickle.dump(X_train.columns, f)
        logger.info(f"Saving column names to {column_names_file}...")
        logger.info("Training features shape:", X_train.shape)
        X_val, y_val = prepare_dataset(logger, args.data_dir, args.val_dataset, 'val', combined_stats, args.debug, args.mem_pred, args.time_pred, not_cross_datasets)
        X_val = collate(X_val, column_names)
        
        logger.info("Validation features shape:", X_val.shape)
        
    else:
        with open(column_names_file, 'rb') as f:
            column_names = pickle.load(f)
        logger.info(f"Loading column names from {column_names_file}...")

    X_test, y_test = prepare_dataset(logger, args.data_dir, args.test_dataset, 'test', combined_stats, args.debug, args.mem_pred, args.time_pred, not_cross_datasets)
    
    X_test = collate(X_test, column_names)

    logger.info("Testing features shape:", X_test.shape)

    

    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )

    best_model_dir = 'checkpoints'

    logger.info(f"Training XGBoost model...")

    model_path = os.path.join(best_model_dir, f'xgb_reg_{"".join(args.dataset)}.pkl')
    if not args.skip_train:
        # Train the model
        xgb_reg.fit(X_train, y_train)
        
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
        
        logger.info(f"Saving XGBoost model to {model_path}...")
        with open(model_path, 'wb') as f:
            pickle.dump(xgb_reg, f)
    else:
        logger.info(f"Skip training and Load XGBoost model from {model_path}...")
        with open(model_path, 'rb') as f:
            xgb_reg = pickle.load(f)

    logger.info(f"Testing XGBoost model...")
    # Predict on test set
    y_pred = xgb_reg.predict(X_test)

    if args.mem_pred:
        y_pred = np.array(y_pred) * combined_stats['peakmem']['scale'] + combined_stats['peakmem']['center']
        y_test = np.array(y_test) * combined_stats['peakmem']['scale'] + combined_stats['peakmem']['center']
    elif args.time_pred:
        y_pred = np.array(y_pred) * combined_stats['time']['scale'] + combined_stats['time']['center']
        y_test = np.array(y_test) * combined_stats['time']['scale'] + combined_stats['time']['center']

    # # Evaluate
    # mse = mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)

    # print(f"Test MSE: {mse:.2f}")
    # print(f"Test R² Score: {r2:.2f}")

    metrics = compute_metrics(y_test, y_pred)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")