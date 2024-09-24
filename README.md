## Usage

run `python src/datasets/prepare_data/execute_*_workload.py` execute worklods and get mem info.
You need to `SET log_statement_stats = on` to enable logging memory usage.

copy those logs to pg_mem_data/pg_log, e.g. from /usr/local/pgsql/data/log.
chmod +rw of these logs.

I currently will use three database: tpch_sf1, tpcds_sf1, and tpch_sf10.


First run src/preprocessing/extract_mem_info.py to extract the memory usage information from the logs. 
You need to modify the data_dir to where you put the pg_log. Also modify the name of the dataset, e.g 'tpch_sf1'
`python src/preprocessing/extract_mem_time_info.py`
This will output mem_info.csv containing <queryid, peakmem, time>. 
peakmem in KB, time in seconds.
The script also write explain verbose format json of each plan to plan_dir (with queryid as filename),
and query sql text to query_dir (with queryid as filename).

<!-- run `python zsce/get_raw_plans.py` which will create new_mem_info.csv containing <queryid, peakmem> but there are fewer rows, because only the queries that succeed executing are included (timeout queries are excluded). This is to ensure all methods have the same train set, val set, and test set. -->

run `python zsce/get_raw_plans.py` that runs 'explain analyze queries' to get raw_plans.json for zsce method. Need to modify the name of the dataset, e.g 'tpch_sf1'

run `python zsce/parse_plans.py` to parse 'raw_plans.json' into 'parsed_plans.json'. Need to modify the name of the dataset, e.g 'tpch_sf1'

run `python zsce/split_parsed_plans.py` to split 'parsed_plans.json' into train, val, test splits. Need to modify the name of the dataset, e.g 'tpch_sf1'

run `python zsce/gather_feature_statistics.py` for zsce method to collect dictionary mapping of categorical values, and get robust scaler statistics for each numerical values. Output is statistics_workload_combined.json. Need to modify the name of the dataset, e.g 'tpch_sf1'

run `python zsce/train.py` to train the zsce method. Need to modify the name of the dataset, e.g 'tpch_sf1'

run `python src/preprocessing/get_database_stats.py` to get database statistics (column_stats, and table_stats). Output is 'database_stats.json'. Need to modify the name of the dataset, e.g 'tpch_sf1'

run `python src/preprocessing/get_explain_json_plans.py` to aggregate the explain json plans and peakmem and time into a giant json object. Need to modify the name of the dataset, e.g 'tpch_sf1'

run `python src/preprocessing/split_json_plans.py` to split the giant json object 'total_json_plans.json' into train, val, test splits. Need to modify the name of the dataset, e.g 'tpch_sf1'

run `python src/preprocessing/gather_feature_statistics.py` to collect dictionary mapping of categorical values, and get robust scaler statistics for each numerical values. Output is statistics_workload_combined.json. Need to modify the name of the dataset, e.g 'tpch_sf1'

Example usage to train:
```
python train.py --train_dataset 'tpch_sf1' --test_dataset 'tpch_sf1' --model_name 'GAT'
```

Example usage to test across database (test transferability):
```
python train.py --skip_train --train_dataset 'tpch_sf1' --test_dataset 'tpcds_sf1' --model_name 'GAT'
```

### heterogeneous_graph
When you update plan_to_graph code, do not forget to `rm -rf data`, because the graph datset is cached in `data` directory.
`python train.py --train_dataset 'tpch_sf1' --test_dataset 'tpcds_sf1'`


## Code Structure

`src` contains the code for GIN, GAT, GraphTransformer, TreeTransformer, `zsce` contains the code for zero-shot cost estimation method, `xgboost` contains the code for XGBoost method, `treelstm` contains the code for TreeLSTM method, `heterogeneous_graph` contains the code for heterogeneous graph method.