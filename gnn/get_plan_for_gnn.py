import os
import sys
sys.path.append('../')
from collect_db_stats import collect_db_statistics
import sys
import psycopg2
from tqdm import tqdm
import json
import pandas as pd


conn_params = {
    "dbname": "tpc_h",
    "user": "wuy",
    "password": "",
    "host": "localhost"
}

def get_result(query):
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    cur.execute(query)
    row = cur.fetchone()[0][0]
    cur.close()
    conn.close()
    return row

plans = []

df = pd.read_csv('../tpch_data/mem_info.csv')
number = df.shape[0]

dir = '../tpch_data/query_dir'
for i in tqdm(range(1, number+1)):
    file_name = os.path.join(dir, str(i) + '.sql')
    with open(file_name, 'r') as f:
        sql = f.read().strip()
    try:
        query = "explain (format json) " + sql
        plan = get_result(query)
        plan['peakmem'] = int(df.loc[i-1, 'peakmem'])
        plans.append(plan)
    except Exception as e:
        print(f"Error in query {i}, sql: {sql}, error: {e}")



with open('../tpch_data/explain_json_plans.json', 'w') as f:
    json.dump(plans, f)
