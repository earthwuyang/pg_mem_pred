import os
import sys
import psycopg2
from tqdm import tqdm

conn_params = {
    "dbname": "tpc_h",
    "user": "wuy",
    "password": "",
    "host": "localhost"
}
 


with open('workloads/tpc_h/workload_100k_s1.sql') as f:
    queries = f.read().split('\n')
    for query in tqdm(queries[200:]):
        if query.strip():
            try:
                conn = psycopg2.connect(**conn_params)
                cur = conn.cursor()
                cur.execute("SET log_statement_stats = on")
                cur.execute(query)
                cur.close()     
                conn.close()
            except Exception as e:
                print(e)
                continue
conn.close()