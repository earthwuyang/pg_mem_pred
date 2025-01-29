import os
import sys
import psycopg2
from tqdm import tqdm
import argparse
import random
import numpy as np
import json

def execute_workload(data_dir, save_dir, dataset, cap_queries, workload_file_name, port):
    with open(os.path.join(os.path.dirname(__file__), '../../conn.json')) as f:
        conn_params = json.load(f)
    conn_params = {
        "dbname": dataset,
        "user": conn_params['user'],
        "password": conn_params['password'],
        "host": conn_params['host'],
        "port": port
    }
    
    workload_file = os.path.join(data_dir, 'workloads', dataset, 'workload_100k_s1_group_order_by_more_complex.sql')
    TP_workload_file = os.path.join(data_dir, 'workloads', dataset, 'TP_queries.sql')
    query_dir = os.path.join(save_dir, dataset, 'raw_data','query_dir')
    verbose_plan_dir = os.path.join(save_dir, dataset, 'raw_data','verbose_plan_dir')
    analyzed_plan_dir = os.path.join(save_dir, dataset, 'raw_data','analyzed_plan_dir')
    os.makedirs(query_dir, exist_ok=True)
    os.makedirs(analyzed_plan_dir, exist_ok=True)
    os.makedirs(verbose_plan_dir, exist_ok=True)

    with open(workload_file, 'r') as f:
        AP_queries = f.read().split('\n')
    with open(TP_workload_file, 'r') as f:
        TP_queries = f.read().split('\n')
    queries = AP_queries + TP_queries
    queries = random.sample(queries, cap_queries)

    count = 0
    for queryid, query in tqdm(enumerate(queries), total=len(queries)):
        query = query.strip()
        if query:
            query_file = os.path.join(query_dir, f'{queryid}.sql')
            with open(query_file, 'w') as f:
                f.write(query)
            try:
                conn = psycopg2.connect(**conn_params)
                cur = conn.cursor()
                cur.execute("SET log_statement_stats = on")
                cur.execute("SET statement_timeout = 300000")
                analyze_query = f"/*{dataset} No.{queryid}*/ explain analyze " + query
                cur.execute(analyze_query)
                rows = cur.fetchall()
                analyzed_plan_file = os.path.join(analyzed_plan_dir, f'{queryid}.txt')
                with open(analyzed_plan_file, 'w') as f:
                    for row in rows:
                        f.write(str(row[0]) + '\n')
                cur.close()     
                conn.close()

                conn = psycopg2.connect(**conn_params)
                cur = conn.cursor()
                cur.execute("explain verbose " + query)
                rows = cur.fetchall()
                verbose_plan_file = os.path.join(verbose_plan_dir, f'{queryid}.txt')
                with open(verbose_plan_file, 'w') as f:
                    for row in rows:
                        f.write(str(row[0]) + '\n')
                cur.close()     
                conn.close()

                count += 1
                if count >= cap_queries:
                    print(f"Cap of {cap_queries} queries reached. Exiting.")
                    break
            except Exception as e:
                print(e)
                continue

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_dir', type=str, default='/home/wuy/DB/pg_mem_data')    

    argparser.add_argument('--dataset', type=str, required=True)
    argparser.add_argument('--workload_file_name', type=str, default='workload_100k_s1_group_order_by_more_complex.sql')
    argparser.add_argument('--cap_queries', type=int, default=50000)
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--port', type=int, default=5432)

    args = argparser.parse_args()
    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    save_dir = args.data_dir
    execute_workload(args.data_dir, save_dir, args.dataset, args.cap_queries, args.workload_file_name, args.port)

if __name__ == '__main__':
    main()