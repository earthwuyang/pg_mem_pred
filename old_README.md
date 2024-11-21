## conda environment
`conda create -n zsce python=3.8.13`
`pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113`
`pip install -r requirements.txt`

## Usage: for each dataset, follow the steps below:

write a conn.json file at the root directory of the project, containing the database connection information. 
Example content: {"user":"wuy", "password":"wuy","host":"localhost", "port":5432}


### create databases from mysql relational database server
create database, create tables, load data into tables(scripts/import_data.py).
<!-- run `python src/preprocessing/export_data.py` to download data from relational database mariadb server.
run `python src/preprocessing/import_data.py --dataset tpch_sf1 --port 5432` to import data into local postgresql database. -->
run `python src/preprocessing/export_import_data.py` to download data from relational database mariadb server and import data into local postgresql database.

### create tpch and tpcds
put tpch data csv files in /data/datasets/tpch_sf1
put tpdcs data csv files in /data/datasets/tpcds_sf1
`/data/datasets/tpch-kit/import_data.sh` to import data into local postgresql database.
`/data/datasets/tpcds-kit/import_data.sh` to import data into local postgresql database.

run `python src/preprocessing/analyze_datasets.py --port 5432` to analyze datasets

run `python src/preprocessing/get_column_type_for_databases.py` to get column type for dataset, which outputs `column_type.json` in each dataset directory.

run `python src/preprocessing/generate_column_string_stats.py` to generate column string statistics for each dataset, which outputs `column_string_stats.json` in each dataset directory.

run `zsce/generate_zsce_queries.py` to generate zsce queries (random sampling of joins and predicates). 


`python src/preprocessing/execute_all_workloads.py`execute worklosds and get mem info and time info as well as writing the output of explain analyze to analyzed_plan_dir (with queryid as filename). Modify postgres_dir and data_dir as needed

<!-- run `python src/preprocessing/execute_workload.py` execute worklosds and get mem info and time info as well as writing the output of explain analyze to analyzed_plan_dir (with queryid as filename). pass arguments `--dataset_dir` and `--dataset` to specify the dataset_dir and dataset correspondingly.
`SET log_statement_stats = on` is needed to enable logging memory usage. -->

<!-- copy those logs to pg_mem_data/pg_log, e.g. from /usr/local/pgsql/data/log.
chmod +r of these logs. -->


First for every dataset, run `python src/preprocessing/extract_mem_time_info.py --dataset tpch_sf1 tpcds_sf1` to extract the memory usage information from the logs. 
You need to modify the data_dir to where you put the pg_log. Also modify the name of the dataset, e.g 'tpch_sf1' or the dataset you want to extract.
This will output mem_info.csv containing <queryid, peakmem, time>. queryid corresponds to the query id in the workload file and file name in query_dir, plan_dir and analyzed_plan_dir.
peakmem in KB, time in seconds.
The script also write explain verbose format json of each plan to plan_dir (with queryid as filename)

Then run `python src/dataset/combine_stats.py`

### for zsce method:
<!-- run `python zsce/get_raw_plans.py --dataset tpch_sf1 tpcds_sf1` that gets raw_plans.json for zsce method. Need to modify the name of the dataset, e.g 'tpch_sf1'.   Note that this will take a long time.

run `python zsce/parse_plans.py --dataset tpch_sf1 tpcds_sf1` to parse 'raw_plans.json' into 'parsed_plans.json'. Need to modify the name of the dataset, e.g 'tpch_sf1'

run `python zsce/split_parsed_plans.py --dataset tpch_sf1 tpcds_sf1` to split 'parsed_plans.json' into train, val, test splits. Need to modify the name of the dataset, e.g 'tpch_sf1' 


run `python zsce/gather_feature_statistics.py --dataset tpch_sf1 tpcds_sf1` for zsce method to collect dictionary mapping of categorical values, and get robust scaler statistics for each numerical values. Output is statistics_workload_combined.json. Need to modify the name of the dataset, e.g 'tpch_sf1' -->
run `python zsce/combine_stats.py`

run `python zsce/train.py` to train the zsce method. Need to modify the name of the dataset, e.g 'tpch_sf1'

### for most methods:
`sh train_cross_datasets.sh` to train across datasets.

Its content: `python train.py --dataset airline carcinogenesis credit employee financial geneea --val_dataset hepatitis --test_dataset tpcds_sf1`
<!-- run `python src/preprocessing/get_database_stats.py --dataset tpch_sf1 tpcds_sf1` to get database statistics (column_stats, and table_stats). Output is 'database_stats.json'. Need to modify the name of the dataset, e.g 'tpch_sf1' -->

<!-- run `python src/preprocessing/get_explain_json_plans.py --dataset tpch_sf1 tpcds_sf1` generate train_plans, val_plans, test_plans in json format. Need to modify the name of the dataset, e.g 'tpch_sf1'

<!-- run `python src/preprocessing/transform_to_zsce_format.py` to transform train, val, test plans into zsce format. output is in 'zsce' subdirectory. -->

<!-- run `python src/preprocessing/gather_feature_statistics.py --dataset tpch_sf1 tpcds_sf1` to collect dictionary mapping of categorical values, and get robust scaler statistics for each numerical values. Output is statistics_workload_combined.json. Need to modify the name of the dataset, e.g 'tpch_sf1'. This is for most methods, not for zsce . -->

<!-- Example usage to train:
```
python train.py --train_dataset 'tpch_sf1' --test_dataset 'tpch_sf1' --model_name 'GAT'
```

Example usage to test across database (test transferability):
```
python train.py --skip_train --train_dataset 'tpch_sf1' --test_dataset 'tpcds_sf1' --model_name 'GAT'
``` -->

### cross machine test example:
`python zsce/train.py --data_dir /home/wuy/DB/pg_mem_data_qh3 --dataset airline credit carcinogenesis employee hepatitis --val_dataset tpcds_sf1 --test_dataset geneea --skip_train`

### heterogeneous_graph
<!-- When you update plan_to_graph code, do not forget to `rm -rf data`, because the graph datset is cached in `data` directory. -->
`python train.py --train_dataset 'tpch_sf1' --test_dataset 'tpcds_sf1'`


## cross docker postgres mem data collection
cd cross_machines/1, modify corresponding parameters, sh start_docker.sh

run python src/preprocessing/import_data.py --port 5422

run python src/preprocessing/execute_all_workloads.py --port 5422 --docker_name my_postgres_2


## use of queryformer code
first run Example Tool to collect sample data.ipynb, some code need not be executed
and then run `python train.py`

## sequential execution
remember to unset https_proxy and http_proxy.


## Code Structure

`src` contains the code for GIN, GAT, GraphTransformer, TreeTransformer, `zsce` contains the code for zero-shot cost estimation method, `xgboost` contains the code for XGBoost method, `treelstm` contains the code for TreeLSTM method, `heterogeneous_graph` contains the code for heterogeneous graph method.