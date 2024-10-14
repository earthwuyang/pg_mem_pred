import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../zsce')))

from cross_db_benchmark.benchmark_tools.generate_column_stats import generate_stats
from cross_db_benchmark.benchmark_tools.generate_string_statistics import generate_string_stats

def main(dataset):
    print(f"Generating column string statistics for {dataset}...")
    dir = f'/home/wuy/DB/pg_mem_data/datasets/{dataset}'
    generate_stats(dir, dataset)   
    generate_string_stats(dir, dataset)


if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from database_list import database_list

    for dataset in database_list:
        main(dataset)