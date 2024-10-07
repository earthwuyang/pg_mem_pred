### train_and_test.py is the main script to train and test the XGBoost model. Grid search is used to find the best hyperparameters for the model.

### metrics.py contains the evaluation metrics used to evaluate the model's performance, includiing qerror, mse etc.

### process_data.py load plans and process datasets

### test.py test the trained model on the test dataset, example usage: python test.py --test_dataset 'tpch' --xgb_model_path 'xgb_model_tpcds_best.json'

# train.ipynb, train_small_dataset.ipynb, train_tpcds.ipynb are the training scripts for different datasets in notebook format.