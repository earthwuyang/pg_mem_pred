import os
import time
import multiprocessing as mp
import argparse
import sys
sys.path.append('../../zsce')

from cross_db_benchmark.benchmark_tools.generate_workload import generate_workload

workload_defs = {
    'workload_100k_s1_group_order_by': dict(num_queries=100000,
                             max_no_predicates=5,
                             max_no_aggregates=3,
                             max_no_group_by=3,
                             max_cols_per_agg=2,
                             groupby_limit_prob=0.2,
                             groupby_having_prob=0.2,
                             seed=1)
}

def workload_gen(input):
    source_dataset, workload_path, max_no_joins, workload_args = input
    generate_workload(source_dataset, workload_path, max_no_joins=max_no_joins, **workload_args)
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload_dir', type=str, default=os.path.join(os.path.dirname(__file__), '../../pg_mem_data/workloads'), help='Directory to store generated workloads')
    args = parser.parse_args()

    if not os.path.exists(args.workload_dir):
        os.makedirs(args.workload_dir)

    workload_gen_setups = []
    for dataset in ['tpcds_sf1']:
        for workload_name, workload_args in workload_defs.items():
            workload_path = os.path.join(args.workload_dir, dataset, f'{workload_name}.sql')
            workload_gen_setups.append((dataset, workload_path, 5, workload_args))
            print(f"Generating workload {workload_name} for {dataset} to {workload_path} with {workload_args}, max_no_joins=5")

    start_t = time.perf_counter()
    proc = mp.cpu_count() - 2
    p = mp.Pool(initargs=('arg',), processes=proc)
    p.map(workload_gen, workload_gen_setups)
    print(f"Generated workloads in {time.perf_counter() - start_t:.2f} secs")