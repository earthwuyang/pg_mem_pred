import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../zsce')))

from cross_db_benchmark.benchmark_tools.generate_column_stats import generate_stats
from cross_db_benchmark.benchmark_tools.generate_string_statistics import generate_string_stats

def main(dataset, overwrite):
    print(f"Generating column string statistics for {dataset}...")
    dir = f'/home/wuy/DB/pg_mem_data/datasets/{dataset}'
    generate_stats(dir, dataset, force = overwrite)   
    generate_string_stats(dir, dataset, force = overwrite) 


if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from database_list import full_database_list

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default=None, nargs='+', help='dataset to generate column string statistics')
    argparser.add_argument('--overwrite', action='store_true', help='overwrite existing statistics')
    args = argparser.parse_args()

    if args.dataset is not None:
        full_database_list = args.dataset

    for dataset in full_database_list:
        main(dataset, args.overwrite)