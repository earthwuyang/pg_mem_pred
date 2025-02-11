# sudo setcap cap_sys_ptrace+ep /home/wuy/software/anaconda3/envs/zsce/bin/python3.8


import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import concurrent.futures
import threading
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
from collections import deque
from datetime import datetime
import subprocess
import torch




@dataclass
class Query:
    id: int
    sql: str
    explain_json_plan: Dict[str, Any]
    pred_peakmem: int
    pred_duration: float
    type: str # large or small

def load_queries(plan_file: str, total_query_memory_limit_kb: int) -> List[Query]:
    """
    Loads queries from a JSON file.
    
    :param plan_file: Path to the JSON file containing query plans.
    :param total_query_memory_limit_kb: Total memory limit for query operations in KB.
    :return: List of Query objects.
    """
    try:
        with open(plan_file, 'r') as f:
            plans = json.load(f)
    except FileNotFoundError:
        logging.error(f"Plan file not found: {plan_file}")
        return []
    except json.JSONDecodeError as jde:
        logging.error(f"JSON decode error in plan file: {jde}")
        return []
    
    queries = []
    for idx, plan in enumerate(plans):
        # Skip queries that exceed the memory limit
        if plan.get('peakmem', 0) >= total_query_memory_limit_kb:
            logging.warning(f"Query {idx+1} requires more memory ({plan.get('peakmem')} KB) than available ({total_query_memory_limit_kb} KB). Skipping.")
            continue

        Q = Query(
            id=idx + 1,
            sql=plan['sql'],
            explain_json_plan=plan,  # Directly assign the plan dict
            pred_peakmem=plan.get('peakmem', 0),
            pred_duration=plan.get('time', 0),
            type = 'large'
        )
        queries.append(Q)
    print(f"Total queries loaded: {len(queries)}")
    return queries

# ----------------------------
# Main Function to Compare Strategies
# ----------------------------
def main():
    # Removed duplicate logging configuration to avoid conflicts
    
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--no_naive', action='store_true', help='Do not execute Naive Strategy.')
    argparser.add_argument('--num_queries', type=int, default=100, help='Number of queries to execute.')
    argparser.add_argument('--dataset', type=str, default='tpcds_sf1', help='Dataset to use.')
    argparser.add_argument('--exp_num', type=int, default=1, help='Number of experimental runs for each strategy.')
    argparser.add_argument('--shared_buffers_mb_in_peakmem', type=int, default=128, help='Shared_buffers in peakmem in MB.')
    argparser.add_argument('--maintenance_work_mem_mb_in_peakmem', type=int, default=64, help='Maintenance_work_mem in peakmem in MB.')
    argparser.add_argument('--device', type=str, default='cpu', help='Device to use for model training and inference.')
    argparser.add_argument('--debug', action='store_true', help='Enable debug logging.')
    argparser.add_argument('--no_ffd', action='store_true', help='Disable FFD.')
    args = argparser.parse_args()

    # if args.debug:
    #     args.exp_num = 1
        # args.no_naive = True

    log_level = logging.DEBUG if args.debug else logging.INFO
    # ----------------------------
    # Configure Logging
    # ----------------------------
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d]: %(message)s",
        handlers=[
            logging.FileHandler("scheduling.log"),
            logging.StreamHandler()
        ]
    )
    

    total_query_memory_limit_kb = 3 * 1024**2  # 3GB
    # Load queries from JSON file
    plan_file = f'/home/wuy/DB/pg_mem_data/{args.dataset}/total_plans.json'
    queries = load_queries(plan_file, total_query_memory_limit_kb)
    queries = sorted(queries, key=lambda x: x.pred_peakmem, reverse=True)

    final_queries = []
    large_num = args.num_queries // 2
    small_num = args.num_queries - large_num
    final_queries.extend(queries[:large_num])
    final_queries.extend(queries[-small_num:])
    queries = final_queries


# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    main()