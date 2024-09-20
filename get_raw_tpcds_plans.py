import os
from collect_db_stats import collect_db_statistics
import sys
import psycopg2
from tqdm import tqdm
import json
import pandas as pd

conn_params = {
    "dbname": "tpc_ds",
    "user": "wuy",
    "password": "",
    "host": "localhost"
}
plan={}
plan['query_list']=[]
plan['database_stats'] = collect_db_statistics(conn_params=conn_params)
plan['run_kwargs']={'hardware': 'qh1'}
plan['total_time_secs']=0

def get_result_analyze(query):
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
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


df=pd.read_csv('tpcds_data/mem_info.csv')
number=df.shape[0]

dir = '/home/wuy/DB/pg_mem_pred/tpcds_data/query_dir'
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
        plan['query_list'].append(plan_tuple)
    except Exception as e:
        print(f"Error in query {i}: {e}")



with open('tpcds_data/raw_plan.json', 'w') as f:
    json.dump(plan, f)