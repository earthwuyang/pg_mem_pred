import json
import os
import collections
from src.preprocessing.gather_feature_statistics import gather_feature_statistics, combine_statistics, compute_numeric_statistics

def combine_stats(logger, args, dataset_list):
    
    combined_stats = {}
    combined_raw_numeric = collections.defaultdict(list)
    combined_statistics_file = os.path.join(args.data_dir, f'combined_statistics_workload_{"_".join(dataset_list)}.json')
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