import os
import sys
import psycopg2
from tqdm import tqdm
import argparse
import json
# os.makedirs('pg_log', exist_ok=True)
# os.system('sudo cp -r /usr/local/pgsql/data/log/* pg_log')
# cp -r /usr/local/pgsql/data/log/* /home/wuy/DB/pg_mem_pred/pg_log
# chmod +r+w pg_log/*


def extract_mem_info(data_dir, dataset):
    log_dir = os.path.join(data_dir, 'pg_log', dataset)
    mem_csv = os.path.join(data_dir, dataset, 'raw_data', 'mem_info.csv')
    query_dir = os.path.join(data_dir, dataset, 'raw_data','query_dir')
    plan_dir = os.path.join(data_dir, dataset, 'raw_data','plan_dir')
    os.makedirs(query_dir, exist_ok=True)
    os.makedirs(plan_dir, exist_ok=True)

    if dataset == 'tpcds':
        database_name = 'tpc_ds'
    elif dataset == 'tpch':
        database_name = 'tpc_h'
    else:
        print('unsupported dataset name')

    conn_params = {
        "dbname": database_name,
        "user": "wuy",
        "password": "",
        "host": "localhost"
    }
    
    # 连接到数据库
    conn = psycopg2.connect(**conn_params)

    csv_header = 'queryid,peakmem,time'
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
                    if 'elapsed' in line:
                        time = line.split(',')[-1].split('s')[0].strip()
                    if 'STATEMENT:' in line:
                        statement = line.split('STATEMENT:')[1].strip()
                        start = False
                        if statement.startswith('commit') or statement.startswith('set') or statement.startswith('explain') or statement.startswith('analyze'):
                            continue
                        
                        cur = conn.cursor()
                        cur.execute('EXPLAIN (verbose, format json)' + statement)
                        plan = cur.fetchone()

                        cur.close()

                        count += 1

                        plan_file = os.path.join(plan_dir, str(count) + '.json')
                        with open(plan_file, 'w') as f:
                            json.dump(plan[0][0], f)

                        with open(mem_csv, 'a') as f:
                            f.write(str(count) + ',' + mem + ',' + time  + '\n')
                        query_file = os.path.join(query_dir, str(count) + '.sql')
                        with open(query_file, 'w') as f:
                            f.write(statement)
                except Exception as e:
                    print('Error:', e)
                    continue

    conn.close()


if __name__ == '__main__':
    data_dir='/home/wuy/DB/pg_mem_data'
    for dataset in ['tpch', 'tpcds']:
        extract_mem_info(data_dir, dataset)