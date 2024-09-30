import os
import sys
import psycopg2
from tqdm import tqdm

conn_params = {
    "dbname": "tpcds_sf1",
    "user": "wuy",
    "password": "",
    "host": "localhost"
}
 

counter = 0
with open('/home/wuy/DB/pg_mem_data/workloads/tpc_ds/workload_100k_s1.sql') as f:
    queries = f.read().split('\n')
    for query in tqdm(queries):
        if counter >= 70000:
            break
        if query.strip():
            try:
                conn = psycopg2.connect(**conn_params)
                cur = conn.cursor()
                cur.execute("SET log_statement_stats = on")
                cur.execute("SET statement_timeout = 300000")
                if 'dbgen_version' in query:
                    continue
                cur.execute(query)
                cur.close()     
                conn.close()
                counter += 1
            except Exception as e:
                print(e)
                continue
conn.close()