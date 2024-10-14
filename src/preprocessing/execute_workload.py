import os
import sys
import psycopg2
from tqdm import tqdm
import argparse
import json

def execute_workload(data_dir, dataset, cap_queries):
    with open(os.path.join(os.path.dirname(__file__), '../../conn.json')) as f:
        conn_params = json.load(f)
    conn_params = {
        "dbname": dataset,
        "user": conn_params['user'],
        "password": conn_params['password'],
        "host": conn_params['host'],
        "port": conn_params['port']
    }
    
    workload_file = os.path.join(data_dir, 'workloads', dataset, 'workload_100k_s1_group_order_by.sql')
    query_dir = os.path.join(data_dir, dataset, 'raw_data','query_dir')
    analyzed_plan_dir = os.path.join(data_dir, dataset, 'raw_data','analyzed_plan_dir')
    os.makedirs(query_dir, exist_ok=True)
    os.makedirs(analyzed_plan_dir, exist_ok=True)

    count = 0
    with open(workload_file, 'r') as f:
        queries = f.read().split('\n')
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
                    cur.execute("SET statement_timeout = 30000")
                    analyze_query = f"/*{dataset} No.{queryid}*/ explain analyze " + query
                    cur.execute(analyze_query)
                    rows = cur.fetchall()
                    analyzed_plan_file = os.path.join(analyzed_plan_dir, f'{queryid}.txt')
                    with open(analyzed_plan_file, 'w') as f:
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
    argparser.add_argument('--cap_queries', type=int, default=50000)

    args = argparser.parse_args()
    execute_workload(args.data_dir, args.dataset, args.cap_queries)

if __name__ == '__main__':
    main()