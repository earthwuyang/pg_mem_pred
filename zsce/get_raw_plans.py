import os
from collect_db_stats import collect_db_statistics
import sys
import psycopg2
from tqdm import tqdm
import json
import pandas as pd
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

def get_result_from_file(queryid, plan_dir):
    file_name = os.path.join(plan_dir, str(queryid) + '.txt')
    with open(file_name, 'r') as f:
        plans = f.readlines()
    plans = [[plan] for plan in plans]
    return plans


def process_query(i, df, dir, analyzed_plan_dir, verbose_plan_dir, conn_params):
    queryid = int(df.loc[i, 'queryid'])
    peakmem = int(df.loc[i, 'peakmem'])
    time = float(df.loc[i, 'time'])
    file_name = os.path.join(dir, str(queryid) + '.sql')
    with open(file_name, 'r') as f:
        sql = f.read().strip()
    try:
        plan_tuple = {}
        verbose_query = "explain verbose " + sql
        plan_tuple['analyze_plans'] = [get_result_from_file(queryid, analyzed_plan_dir)]
        plan_tuple['verbose_plan'] = get_result_from_file(queryid, verbose_plan_dir)
        plan_tuple['sql'] = sql
        plan_tuple['peakmem'] = peakmem
        plan_tuple['time'] = time
        plan_tuple['queryid'] = queryid
        return plan_tuple
    except Exception as e:
        print(f"Error in query {queryid}: {e}")
        return None

def get_raw_plans(data_dir, dataset):
    with open(os.path.join(os.path.dirname(__file__), '../conn.json')) as f:
        conn_params = json.load(f)
    conn_params = {
        "dbname": dataset,
        "user": conn_params['user'],
        "password": conn_params['password'],
        "host": conn_params['host'],
        "port": conn_params['port']
    }
    plan = {}
    plan['query_list'] = []
    plan['database_stats'] = collect_db_statistics(conn_params)
    plan['run_kwargs'] = {'hardware': 'qh1'}
    plan['total_time_secs'] = 0

    df = pd.read_csv(os.path.join(data_dir, dataset, 'raw_data', 'mem_info.csv'))
    number = df.shape[0]

    dir = os.path.join(data_dir, dataset, 'raw_data', 'query_dir')
    analyzed_plan_dir = os.path.join(data_dir, dataset, 'raw_data', 'analyzed_plan_dir')
    verbose_plan_dir = os.path.join(data_dir, dataset, 'raw_data', 'verbose_plan_dir')

    # Create a partial function to pass arguments to process_query
    process_query_partial = partial(process_query, df=df, dir=dir, analyzed_plan_dir=analyzed_plan_dir, 
                                    verbose_plan_dir=verbose_plan_dir, conn_params=conn_params)

    # Use multiprocessing to process queries in parallel
    with Pool(processes=10) as pool:
        query_plans = list(tqdm(pool.imap(process_query_partial, range(number)), total=number))

    # Filter out None values in case of errors
    plan['query_list'] = [qp for qp in query_plans if qp is not None]

    raw_plan_path = os.path.join(data_dir, dataset, 'zsce', 'raw_plan.json')
    if not os.path.exists(os.path.dirname(raw_plan_path)):
        os.makedirs(os.path.dirname(raw_plan_path), exist_ok=True)
    with open(raw_plan_path, 'w') as f:
        json.dump(plan, f)

if __name__ == '__main__':
    data_dir = '/home/wuy/DB/pg_mem_data'
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default=['tpch_sf1'], nargs='+')
    args = argparser.parse_args()
    for dataset in args.dataset:
        get_raw_plans(data_dir, dataset)
