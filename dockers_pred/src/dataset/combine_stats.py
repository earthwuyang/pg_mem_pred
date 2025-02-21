import json
import os
import collections
import sys
sys.path.append('/home/wuy/DB/memory_prediction/')
import logging
from src.preprocessing.gather_feature_statistics import gather_feature_statistics, combine_statistics, compute_numeric_statistics
from database_list import full_database_list as dataset_list

def combine_stats(logger, args, dataset_list):
    
    combined_stats = {}
    combined_raw_numeric = collections.defaultdict(list)
    # combined_statistics_file = os.path.join(args.data_dir, f'combined_statistics_workload_{"_".join(dataset_list)}.json')
    combined_statistics_file = os.path.join(args.data_dir, f'combined_statistics_workload.json')
    if args.force or not os.path.exists(combined_statistics_file):
        with open(combined_statistics_file, 'w') as f:
            for dataset in dataset_list:
                logger.info(f"gathering feature statistics for {dataset}...")
                stats, raw_numeric = gather_feature_statistics(args.data_dir, dataset)
                combine_statistics(combined_stats, stats, raw_numeric, combined_raw_numeric)
                logger.info(f"Completed dataset {dataset}")
            logger.info(f"Computing scale and center for numeric features using combined raw data...")
            combined_stats = compute_numeric_statistics(combined_stats, combined_raw_numeric)
            with open(combined_statistics_file, 'w') as f:
                json.dump(combined_stats, f, indent=4)
    else:
        logger.info(f"{combined_statistics_file} already exists, skipping gathering feature statistics")
        with open(combined_statistics_file, 'r') as f:
            combined_stats = json.load(f)
    return combined_stats

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../pg_mem_data', help='path to data directory')
    parser.add_argument('--force', action='store_true', help='force re-gathering feature statistics')
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    combined_stats = combine_stats(logger, args, dataset_list)
