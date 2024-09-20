import json
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import metrics
from sklearn.model_selection import GridSearchCV
import argparse
from time import time

import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from process_data import load_plans, prepare_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tpch')
    # parser.add_argument('--train_plan_file', type=str, default='../tpch_data/train_plans.json')
    # parser.add_argument('--val_plan_file', type=str, default='../tpch_data/val_plans.json')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    train_plan_file = f'../{args.dataset}_data/train_plans.json'
    val_plan_file = f'../{args.dataset}_data/val_plans.json'


    # Load training and validation plans
    train_plans = load_plans(train_plan_file)
    val_plans = load_plans(val_plan_file)

    print(f"Number of training plans: {len(train_plans['parsed_plans'])}")
    print(f"Number of validation plans: {len(val_plans['parsed_plans'])}")

    # Prepare training and validation datasets
    X_train, y_train = prepare_dataset(train_plans)
    X_val, y_val = prepare_dataset(val_plans)
    X_val = X_val[X_train.columns]

    print("Training features shape:", X_train.shape)
    print("Validation features shape:", X_val.shape)

    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )


    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Initialize the XGBoost regressor
    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_reg,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    begin=time()
    # Perform grid search
    grid_search.fit(X_train, y_train)
    end=time()
    print(f"Grid search time: {end-begin:.2f}s")

    # Best parameters and score
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV Score (Negative MSE): {grid_search.best_score_}")

    # Use the best estimator for predictions
    best_xgb = grid_search.best_estimator_

    saved_model_path = f'xgb_model_{args.dataset}_best.json'
    best_xgb.save_model(saved_model_path)
    print(f"best xgb model saved to {saved_model_path}")


    y_pred_best = best_xgb.predict(X_val)

    # Compute metrics for the best model
    metrics_best = metrics.compute_metrics(y_val, y_pred_best)

    print(f"\nValidation Metrics on {val_plan_file} for Best Model:")
    for metric, value in metrics_best.items():
        print(f"{metric}: {value:.4f}")

    