import sys
import os


from cross_db_benchmark.benchmark_tools.generate_string_statistics import generate_string_stats

if __name__ == '__main__':
    dataset = 'tpch_sf1'
    generate_string_stats('/home/wuy/DB/pg_mem_data/datasets/tpch-kit/data-1', dataset)