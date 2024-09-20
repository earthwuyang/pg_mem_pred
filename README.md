### Steps for pg_mem_pred
execute_tpch_workload
cp pg_logs from pgsql/data to current directory
chmod +r+w pg_logs/*
extract_mem_info.py
get_raw_plans.py
parse_plans.py
gather_feature_statistics.py
split_train_val.py
train.py
