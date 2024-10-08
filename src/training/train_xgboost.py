import json
import os
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
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


def extract_features(parsed_plan):
    """
    Extract features from a parsed PostgreSQL query plan.

    Args:
        parsed_plan (dict): The parsed query plan.

    Returns:
        dict: A dictionary of aggregated features.
    """
    features = defaultdict(float)
    op_counts = defaultdict(int)
    numerical_features = defaultdict(list)
    max_depth = 0

    def traverse(node, depth=1):
        nonlocal max_depth
        max_depth = max(max_depth, depth)

        params = node.get('plan_parameters', {})
        op_name = params.get('op_name', 'Unknown')
        op_counts[op_name] += 1

        # Collect numerical features
        numerical_features['est_startup_cost'].append(params.get('est_startup_cost', 0.0))
        numerical_features['est_cost'].append(params.get('est_cost', 0.0))
        numerical_features['est_card'].append(params.get('est_card', 0.0))
        numerical_features['est_width'].append(params.get('est_width', 0.0))
        numerical_features['workers_planned'].append(params.get('workers_planned', 0.0))
        numerical_features['est_children_card'].append(params.get('est_children_card', 0.0))

        # Recursively traverse children
        for child in node.get('children', []):
            traverse(child, depth + 1)

    traverse(parsed_plan)

    # Aggregate operation counts
    for op, count in op_counts.items():
        features[f'op_count_{op}'] = count

    # Aggregate numerical features
    for feature, values in numerical_features.items():
        features[f'{feature}_sum'] = sum(values)
        features[f'{feature}_mean'] = np.mean(values) if values else 0.0
        features[f'{feature}_max'] = max(values) if values else 0.0
        features[f'{feature}_min'] = min(values) if values else 0.0

    # Add structural features
    features['tree_depth'] = max_depth
    features['num_nodes'] = len(numerical_features['est_cost'])

    return features




def prepare_dataset(plans):
    """
    Prepare a dataset by extracting features and collecting labels.

    Args:
        plans (list): A list of parsed query plans.

    Returns:
        pd.DataFrame: DataFrame containing features.
        pd.Series: Series containing labels (peak memory).
    """
    feature_dicts = []
    labels = []

    for plan in tqdm(plans['parsed_plans']):
        features = extract_features(plan)
        feature_dicts.append(features)
        labels.append(plan.get('peakmem', 0.0))  # Assuming 'peakmem' is the target

    # Convert to DataFrame
    df_features = pd.DataFrame(feature_dicts)
    df_labels = pd.Series(labels, name='peakmem')

    # Handle missing values if any
    df_features.fillna(0, inplace=True)

    # Encode categorical features (if any)
    # Assuming 'op_count_*' are categorical; adjust based on actual data
    op_count_features = [col for col in df_features.columns if col.startswith('op_count_')]
    # If they are counts, you might treat them as numerical
    # Otherwise, use label encoding or one-hot encoding as needed

    # Feature scaling (optional)
    scaler = StandardScaler()
    numerical_cols = [col for col in df_features.columns if any(sub in col for sub in ['est_startup_cost', 'est_cost', 'est_card', 'est_width', 'workers_planned', 'est_children_card'])]
    df_features[numerical_cols] = scaler.fit_transform(df_features[numerical_cols])

    return df_features, df_labels





def train_XGBoost(args):
    train_plan_file = os.path.join(args.dataset_dir, args.train_dataset, 'zsce', 'train_plans.json')
    train_plans = load_plans(train_plan_file)
    print(f"number of training plans: {len(train_plans['parsed_plans'])}")

    val_plan_file = os.path.join(args.dataset_dir, args.train_dataset, 'zsce', 'val_plans.json')
    val_plans = load_plans(val_plan_file)
    print(f"number of validation plans: {len(val_plans['parsed_plans'])}")

    test_plan_file = os.path.join(args.dataset_dir, args.test_dataset, 'zsce', 'test_plans.json')
    test_plans = load_plans(test_plan_file)
    print(f"number of testing plans: {len(test_plans['parsed_plans'])}")

    # Prepare training and validation datasets
    X_train, y_train = prepare_dataset(train_plans)
    X_val, y_val = prepare_dataset(val_plans)
    X_val = X_val[X_train.columns]
    X_test, y_test = prepare_dataset(test_plans)
    X_test = X_test[X_train.columns]

    print("Training features shape:", X_train.shape)
    print("Validation features shape:", X_val.shape)
    print("Testing features shape:", X_test.shape)

    

    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )

    print(f"Training XGBoost model...")
    # Train the model
    xgb_reg.fit(X_train, y_train)

    print(f"Testing XGBoost model...")
    # Predict on test set
    y_pred = xgb_reg.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test MSE: {mse:.2f}")
    print(f"Test R² Score: {r2:.2f}")

    metrics = compute_metrics(y_test, y_pred)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")