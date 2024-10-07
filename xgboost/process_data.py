import json
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm

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