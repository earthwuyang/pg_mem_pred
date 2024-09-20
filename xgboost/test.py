import xgboost as xgb
import argparse
import metrics
from process_data import prepare_dataset, load_plans

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--test_dataset', type=str, help='Name of the test dataset', default='tpch')
parser.add_argument('--xgb_model_path', type=str,  help='Path to the saved XGBoost model', default='xgb_model_tpch_best.json')
args = parser.parse_args()

xgb_model_path = args.xgb_model_path

# Create a new XGBRegressor instance
xgb_regressor = xgb.XGBRegressor()

# Load the saved model
xgb_regressor.load_model(xgb_model_path)

test_plan_file = f"../{args.test_dataset}_data/val_plans.json"
test_plans=load_plans(test_plan_file)
X_test, y_test = prepare_dataset(test_plans)

feature_names = xgb_regressor.get_booster().feature_names
missing_cols = set(feature_names) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0
X_test = X_test[feature_names]

# Make predictions on the test set
y_pred = xgb_regressor.predict(X_test)

# Compute metrics
print(f"metrics on {test_plan_file} by {args.xgb_model_path}")
metrics_dict = metrics.compute_metrics(y_test, y_pred)
for metric, value in metrics_dict.items():
    print(f"{metric}: {value:.4f}")



