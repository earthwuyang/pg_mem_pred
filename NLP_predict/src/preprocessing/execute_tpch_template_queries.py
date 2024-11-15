import os
import sys
import psycopg2
from tqdm import tqdm

conn_params = {
    "dbname": "tpch_sf1",
    "user": "wuy",
    "password": "",
    "host": "localhost"
}
 
base_dir = '/home/wuy/DB/tpch-kit/dbgen/random_queries'
for file in tqdm(os.listdir(base_dir)):
    with open(os.path.join(base_dir, file), 'r') as f:
        query = f.read()
        query=query.replace('(3)','').split('limit')[0]
        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()
        cur.execute("SET log_statement_stats = on")
        cur.execute("SET statement_timeout = 30000")
        try:
            cur.execute(query)
        except psycopg2.Error as e:
            print(e)
            print(query)
        conn.commit()
        conn.close()