## conda environment
`conda create -n zsce python=3.8.13`
`pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113`
`pip install -r requirements.txt`

## Usage
run `python zsce/cross_db_benchmark/datasets/tpc_h/scripts/script_to_get_column_type.py` to get column type for each table in TPC-H dataset, which outputs `column_type.json`.

run `zsce/generate_column_stats.py` and `zsce/generate_string_stats.py`

run `zsce/generate_zsce_queries.py` to generate zsce queries (random sampling of joins and predicates).


run `python src/datasets/prepare_data/execute_workload.py` execute worklods and get mem info and time info as well as writing the output of explain analyze to analyzed_plan_dir (with queryid as filename). pass arguments `--dataset_dir` and `--dataset` to specify the dataset_dir and dataset correspondingly.
`SET log_statement_stats = on` is needed to enable logging memory usage.

copy those logs to pg_mem_data/pg_log, e.g. from /usr/local/pgsql/data/log.
chmod +r of these logs.

I currently will use three database: tpch_sf1, tpcds_sf1, and tpch_sf10.


First run `python src/preprocessing/extract_mem_time_info.py` to extract the memory usage information from the logs. 
You need to modify the data_dir to where you put the pg_log. Also modify the name of the dataset, e.g 'tpch_sf1'
This will output mem_info.csv containing <queryid, peakmem, time>. queryid corresponds to the query id in the workload file and file name in query_dir, plan_dir and analyzed_plan_dir.
peakmem in KB, time in seconds.
The script also write explain verbose format json of each plan to plan_dir (with queryid as filename)


run `psql`, connect to the database, run `analyze` to collect statistics for each table.

run `python zsce/get_raw_plans.py` that gets raw_plans.json for zsce method. Need to modify the name of the dataset, e.g 'tpch_sf1'.   Note that this will take a long time.

run `python zsce/parse_plans.py` to parse 'raw_plans.json' into 'parsed_plans.json'. Need to modify the name of the dataset, e.g 'tpch_sf1'

run `python zsce/split_parsed_plans.py` to split 'parsed_plans.json' into train, val, test splits. Need to modify the name of the dataset, e.g 'tpch_sf1' 


run `python zsce/gather_feature_statistics.py` for zsce method to collect dictionary mapping of categorical values, and get robust scaler statistics for each numerical values. Output is statistics_workload_combined.json. Need to modify the name of the dataset, e.g 'tpch_sf1'


run `python zsce/train.py` to train the zsce method. Need to modify the name of the dataset, e.g 'tpch_sf1'

run `python src/preprocessing/get_database_stats.py` to get database statistics (column_stats, and table_stats). Output is 'database_stats.json'. Need to modify the name of the dataset, e.g 'tpch_sf1'

run `python src/preprocessing/get_explain_json_plans.py` generate train_plans, val_plans, test_plans in json format. Need to modify the name of the dataset, e.g 'tpch_sf1'

run `python src/preprocessing/transform_to_zsce_format.py` to transform train, val, test plans into zsce format. output is in 'zsce' subdirectory.


run `python src/preprocessing/gather_feature_statistics.py` to collect dictionary mapping of categorical values, and get robust scaler statistics for each numerical values. Output is statistics_workload_combined.json. Need to modify the name of the dataset, e.g 'tpch_sf1'. This is for most methods, not for zsce .

Example usage to train:
```
python train.py --train_dataset 'tpch_sf1' --test_dataset 'tpch_sf1' --model_name 'GAT'
```

Example usage to test across database (test transferability):
```
python train.py --skip_train --train_dataset 'tpch_sf1' --test_dataset 'tpcds_sf1' --model_name 'GAT'
```

### heterogeneous_graph
<!-- When you update plan_to_graph code, do not forget to `rm -rf data`, because the graph datset is cached in `data` directory. -->
`python train.py --train_dataset 'tpch_sf1' --test_dataset 'tpcds_sf1'`


## Code Structure

`src` contains the code for GIN, GAT, GraphTransformer, TreeTransformer, `zsce` contains the code for zero-shot cost estimation method, `xgboost` contains the code for XGBoost method, `treelstm` contains the code for TreeLSTM method, `heterogeneous_graph` contains the code for heterogeneous graph method.