import sys
import os


from cross_db_benchmark.benchmark_tools.generate_column_stats import generate_stats

if __name__ == '__main__':
    # dataset = 'tpch_sf1'
    # generate_stats('/home/wuy/DB/pg_mem_data/datasets/tpch-kit/data-1', dataset)
    dataset = 'tpcds_sf1'
    dir = '/home/wuy/DB/pg_mem_data/datasets/tpcds-kit/data-1'
    generate_stats(dir, dataset)