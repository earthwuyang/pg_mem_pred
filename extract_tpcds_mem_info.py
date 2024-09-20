# first copy logs from /usr/local/pgsql/data/log 
# then run this script to extract memory usage information from logs and save it to mem_info.csv and query_dir/ and plan_dir/ 

import os
import sys
import psycopg2
from tqdm import tqdm

conn_params = {
    "dbname": "tpc_ds",
    "user": "wuy",
    "password": "",
    "host": "localhost"
}
 
# 连接到数据库
conn = psycopg2.connect(**conn_params)


# os.makedirs('pg_log', exist_ok=True)
# os.system('sudo cp -r /usr/local/pgsql/data/log/* pg_log')
# cp -r /usr/local/pgsql/data/log/* /home/wuy/DB/pg_mem_pred/pg_tpcds_log
# chmod +r+w pg_tpcds_log/*

log_dir='pg_tpcds_log'
data_dir='tpcds_data'
mem_csv = os.path.join(data_dir,'mem_info.csv')
query_dir = os.path.join(data_dir,'query_dir')
plan_dir = os.path.join(data_dir,'plan_dir')
os.makedirs(query_dir, exist_ok=True)
os.makedirs(plan_dir, exist_ok=True)

csv_header = 'queryid,peakmem'
with open(mem_csv, 'w') as f:
    f.write(csv_header+'\n')


count = 0
for file in os.listdir(log_dir):
    print(f"file: {file}")
    with open(os.path.join(log_dir, file), 'r') as f:
        file_data = f.readlines()
    start = False
    for line in tqdm(file_data):
        if 'DETAIL:  ! system usage stats' in line:
            start = True
        if start:
            try:
                if 'max resident size' in line:
                    mem = line.split('kB')[0].split('!')[1].strip()
                if 'STATEMENT:' in line:
                    statement = line.split('STATEMENT:')[1].strip()
                    start = False
                    if statement.startswith('commit') or statement.startswith('set') or statement.startswith('explain') or statement.startswith('analyze'):
                        continue
                    
                    cur = conn.cursor()
                    cur.execute('EXPLAIN (format json)' + statement)
                    plan = cur.fetchall()

                    cur.close()

                    count += 1

                    plan_file = os.path.join(plan_dir, str(count) + '.txt')
                    with open(plan_file, 'w') as f:
                        f.write(str(plan[0][0][0]))
                        # for p in plan:
                        #     f.write(p[0] + '\n')

                    with open(mem_csv, 'a') as f:
                        f.write(str(count) + ',' + mem + '\n')
                    query_file = os.path.join(query_dir, str(count) + '.sql')
                    with open(query_file, 'w') as f:
                        f.write(statement)
            except Exception as e:
                print('Error:', e)
                continue

            