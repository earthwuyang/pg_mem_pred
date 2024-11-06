import os
import sys
import psycopg2
from tqdm import tqdm
import argparse
import json
import argparse
import subprocess
import getpass

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from execute_workload import execute_workload
from database_list import database_list, mysql_database_list, full_database_list

def execute_all_workloads(args):
    previous_dataset = None
    for dataset in args.dataset:

        # move previous log files to data_dir/pg_log/previous_dataset or remove logs if it's the first dataset
        rm_command = f"echo {args.password} | sudo -S -u postgres bash -c 'rm -rf {os.path.join(args.postgres_dir, 'data/log')}'"
        if previous_dataset is None:
            subprocess.run(rm_command, shell=True)
            print(f"Removed old postgres logs files")
        else:
            pg_log_dataset_dir = os.path.join(args.data_dir, 'pg_log', previous_dataset)
            os.makedirs(pg_log_dataset_dir, exist_ok=True)
            mv_command = f"echo {args.password} | sudo -S -u root bash -c 'mv {os.path.join(args.postgres_dir, 'data/log/post*')} {pg_log_dataset_dir}'"
            subprocess.run(mv_command, shell=True)
            chmod_command = f"echo {args.password} | sudo -S -u root bash -c 'chmod -R +r {pg_log_dataset_dir}'"
            subprocess.run(chmod_command, shell=True)
            print(f"move previous postgres logs files of dataset {previous_dataset} to {pg_log_dataset_dir} and grant read permission")

        # restart postgres
        pg_restart_command = f"echo {args.password} | sudo -S -u postgres bash -c 'cd {args.postgres_dir}/bin && ./pg_ctl restart -D ../data'"
        subprocess.run(pg_restart_command, shell=True)
        print(f"Restarted postgres for {dataset}")

        # execute workload
        execute_workload(args.data_dir, dataset, args.cap_queries)
        previous_dataset = dataset

    # move log files to data_dir/pg_log/dataset
    pg_log_dataset_dir = os.path.join(args.data_dir, 'pg_log', previous_dataset)
    os.makedirs(pg_log_dataset_dir, exist_ok=True)
    mv_command = f"echo {args.password} | sudo -S -u root bash -c 'mv {os.path.join(args.postgres_dir, 'data/log/post*')} {pg_log_dataset_dir}'"
    subprocess.run(mv_command, shell=True)
    chmod_command = f"echo {args.password} | sudo -S -u root bash -c 'chmod -R +r {pg_log_dataset_dir}'"
    subprocess.run(chmod_command, shell=True)
    print(f"move postgres logs files of dataset {previous_dataset} to {pg_log_dataset_dir} and grant read permission")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_dir', type=str, default='/home/wuy/DB/pg_mem_data')    
    argparser.add_argument('--dataset', type=str, nargs='+', default=full_database_list)
    argparser.add_argument('--postgres_dir', type=str, default='/usr/local/pgsql/')
    argparser.add_argument('--password', type=str, default=None)
    argparser.add_argument('--cap_queries', type=int, default=50000)
    args = argparser.parse_args()

    if args.password is None:
        password = getpass.getpass(prompt='Enter password for user to get sudo privileges: ')
        args.password = password

    execute_all_workloads(args)