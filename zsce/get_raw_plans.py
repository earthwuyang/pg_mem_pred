import os
from collect_db_stats import collect_db_statistics
import sys
import psycopg2
from tqdm import tqdm
import json
import pandas as pd

def get_raw_plans(data_dir, dataset):
    database = dataset
    conn_params = {
        "dbname": database,
        "user": "wuy",
        "password": "",
        "host": "localhost"
    }
    plan={}
    plan['query_list']=[]
    plan['database_stats'] = collect_db_statistics()
    plan['run_kwargs']={'hardware': 'qh1'}
    plan['total_time_secs']=0

    def get_result_analyze(query):
        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()
        cur.execute('set statement_timeout = 0;')
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        result=[]
        for row in rows:
            result.append([row[0]])
        return [result]

    def get_result_verbose(query):
        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        result=[]
        for row in rows:
            result.append([row[0]])
        return result

    # new_mem_csv = os.path.join(data_dir, dataset, 'raw_data', 'new_mem_info.csv')
    # csv_header = 'queryid,peakmem,time'
    # with open(new_mem_csv, 'w') as f:
    #     f.write(csv_header+'\n')

    df=pd.read_csv(os.path.join(data_dir, dataset, 'raw_data', 'mem_info.csv'))
    number=df.shape[0]
    # if dataset == 'tpcds':
    #     number = round(number *0.711) # to make the number of queries roughly equal to TPC-H

    dir = os.path.join(data_dir, dataset, 'raw_data', 'query_dir')
    for i in tqdm(range(1, number+1)):
        file_name = os.path.join(dir, str(i) + '.sql')
        with open(file_name, 'r') as f:
            sql = f.read().strip()
        try:
            plan_tuple = {}
            analyze_query = "explain analyze " + sql
            verbose_query = "explain verbose " + sql
            plan_tuple['analyze_plans']=get_result_analyze(analyze_query)
            plan_tuple['verbose_plan']=get_result_verbose(verbose_query)
            plan_tuple['sql'] = sql
            plan_tuple['peakmem'] = int(df.loc[i-1, 'peakmem'])
            plan_tuple['time'] = float(df.loc[i-1, 'time'])
            plan_tuple['queryid'] = int(df.loc[i-1, 'queryid'])
            plan['query_list'].append(plan_tuple)
            assert i == df.loc[i-1, 'queryid'], f"queryid {i} does not match with the index in mem_info.csv"
            # with open(new_mem_csv, 'a') as f:
            #     f.write(f"{i},{df.loc[i-1, 'peakmem']}\n")
        except Exception as e:
            print(f"Error in query {i}: {e}")


    raw_plan_path = os.path.join(data_dir, dataset, 'zsce', 'raw_plan.json')
    if not os.path.exists(os.path.dirname(raw_plan_path)):
        os.makedirs(os.path.dirname(raw_plan_path), exist_ok=True)
    with open(raw_plan_path, 'w') as f:
        json.dump(plan, f)

if __name__ == '__main__':
    data_dir = '/home/wuy/DB/pg_mem_data'
    for dataset in ['tpcds_sf1']:
        get_raw_plans(data_dir, dataset)