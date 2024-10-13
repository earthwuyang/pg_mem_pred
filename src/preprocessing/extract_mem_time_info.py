import os
import sys
import psycopg2
from tqdm import tqdm
import argparse
import json
import argparse

# os.makedirs('pg_log', exist_ok=True)
# os.system('sudo cp -r /usr/local/pgsql/data/log/* pg_log')
# cp -r /usr/local/pgsql/data/log/* /home/wuy/DB/pg_mem_pred/pg_log
# chmod +r+w pg_log/*


def extract_mem_info(data_dir, dataset):
    log_dir = os.path.join(data_dir, 'pg_log', dataset)
    mem_csv = os.path.join(data_dir, dataset, 'raw_data', 'mem_info.csv')
    plan_dir = os.path.join(data_dir, dataset, 'raw_data','plan_dir')
    os.makedirs(plan_dir, exist_ok=True)


    with open(os.path.join(os.path.dirname(__file__), '../../conn.json')) as f:
        conn_params = json.load(f)
    conn_params = {
        "dbname": dataset,
        "user": conn_params['user'],
        "password": conn_params['password'],
        "host": conn_params['host'],
        "port": conn_params['port']
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
                        # if statement.startswith('commit') or statement.startswith('set') or statement.startswith('explain') or statement.startswith('analyze') \
                        #     or statement.startswith('COMMIT') or statement.startswith('SET') or statement.startswith('EXPLAIN') or statement.startswith('ANALYZE'):
                        #     continue
                        if not statement.startswith('/*'):
                            continue
                        statement_dataset = statement.split(' ')[0].split('/*')[1].strip()
                        if statement_dataset != dataset:
                            continue
                        queryid = statement.split('*/')[0].split('No.')[1].strip()
                        sql = statement.split('*/')[1].split('explain analyze')[1].strip()

                        cur = conn.cursor()
                        cur.execute('EXPLAIN (verbose, format json) ' + sql)
                        plan = cur.fetchone()

                        cur.close()

                        count += 1
                        # print(f"queryid: {queryid}, mem: {mem}, time: {time}, statement_dataset: {statement_dataset}, sql: {sql}")
                        # while 1:pass
                        plan_file = os.path.join(plan_dir, queryid + '.json')
                        with open(plan_file, 'w') as f:
                            json.dump(plan[0][0], f)

                        with open(mem_csv, 'a') as f:
                            f.write(queryid + ',' + mem + ',' + time  + '\n')
                       
                except Exception as e:
                    print('Error:', e)
                    continue

    conn.close()
    print(f"Number of queries with memory info: {count}")

if __name__ == '__main__':
    data_dir='/home/wuy/DB/pg_mem_data'
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', nargs='+', type=str, default=['tpch_sf1'])
    args = argparser.parse_args()

    for dataset in args.dataset:
        extract_mem_info(data_dir, dataset)