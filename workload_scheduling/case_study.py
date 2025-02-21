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
import itertools
import heapq
from tqdm import tqdm
from copy import deepcopy

@dataclass
class Query:
    id: int
    pred_peakmem: int
    pred_duration: float
    type: str  = None # large or small
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


def factorial(x):
    if x == 0 or x == 1:
        return 1
    else:
        return x*(x-1)

def schedule_queries(queries: List[Query], memory_limit_kb: int) -> Tuple[List[Query], float]:
    """
    Brute-force scheduling of queries to minimize makespan under memory constraints.
    
    :param queries: List of queries to schedule.
    :param memory_limit_kb: Available memory limit in KB.
    :return: A tuple containing the optimal scheduling order and the minimum makespan.
    """
    best_makespan = float('inf')
    best_schedule = []

    # Generate all permutations of the queries
    perms = itertools.permutations(queries)
    number = factorial(len(queries))
    for original_perm in tqdm(perms, total=number):
        # copy original_perm to perm using deepcopy to avoid modifying the original list, you can't use q.copy() here because Query object has no attribute copy
        
        perm = [deepcopy(q) for q in original_perm]
        current_time = 0
        memory_in_use = 0
        
        running_queries = deque()
        for query in perm:
            # Wait until the current memory usage allows the query to run
            while memory_in_use + query.pred_peakmem > memory_limit_kb:
                # Simulate waiting (this could be enhanced with more detailed time management)
                earliest_finish_query = min(running_queries, key=lambda q: q.end_time)
                current_time = earliest_finish_query.end_time
                # delete earliest_finish_query from running_queries
                running_queries.remove(earliest_finish_query)
                # Update memory_in_use as the sum of remaining running queries, excluding the one finishing first
                memory_in_use = sum(q.pred_peakmem for q in running_queries)
            
            # Add query's memory usage and duration to the current schedule
            memory_in_use += query.pred_peakmem
            query.start_time = current_time
            query.end_time = current_time + query.pred_duration
            running_queries.append(query)
        
        makespan = max(q.end_time for q in perm)
        
        # Update best schedule if we find a better makespan
        if makespan <= best_makespan:
            best_makespan = makespan
            best_schedule = perm
    
    return best_schedule, best_makespan


def schedule_queries_ffd(queries: List[Query], memory_limit_kb: int) -> Tuple[List[Query], float]:
    """
    FFD scheduling of queries to minimize makespan under memory constraints.
    
    :param queries: List of queries to schedule.
    :param memory_limit_kb: Available memory limit in KB.
    :return: A tuple containing the optimal scheduling order and the minimum makespan.
    """

    # Generate all permutations of the queries
    sorted_queries = sorted(queries, key=lambda x: x.pred_peakmem, reverse=True)
    current_time = 0
    memory_in_use = 0
    finish_times = []
        
    running_queries = deque()
    for query in sorted_queries:
        # Wait until the current memory usage allows the query to run
        while memory_in_use + query.pred_peakmem > memory_limit_kb:
            # print(f"memory exceeding")
            # Simulate waiting (this could be enhanced with more detailed time management)
            earliest_finish_query = min(running_queries, key=lambda q: q.end_time)
            current_time = earliest_finish_query.end_time
            # delete earliest_finish_query from running_queries
            running_queries.remove(earliest_finish_query)
            # Update memory_in_use as the sum of remaining running queries, excluding the one finishing first
            memory_in_use = sum(q.pred_peakmem for q in running_queries)
        
        # Add query's memory usage and duration to the current schedule
        memory_in_use += query.pred_peakmem
        # logging.info(f"now query: {query.id}, memory_in_use: {memory_in_use} KB, memory_limit_kb: {memory_limit_kb} KB, peakmem: {query.pred_peakmem} KB")

        query.start_time = current_time
        query.end_time = current_time + query.pred_duration
        running_queries.append(query)
        
    ffd_schedule = sorted_queries
    makespan = max(q.end_time for q in sorted_queries)

    return ffd_schedule, makespan

def schedule_queries_bf(queries: List[Query], memory_limit_kb: int) -> Tuple[List[Query], float]:
    """
    FFD scheduling of queries to minimize makespan under memory constraints.
    
    :param queries: List of queries to schedule.
    :param memory_limit_kb: Available memory limit in KB.
    :return: A tuple containing the optimal scheduling order and the minimum makespan.
    """

    # Generate all permutations of the queries
    sorted_queries = sorted(queries, key=lambda x: x.pred_peakmem, reverse=True)
    bf_schedule = []
    current_time = 0
    memory_in_use = 0
    finish_times = []
    queue = deque()
    for query in sorted_queries:
        queue.append(query)
    mode = "large"
        
    running_queries = deque()
    while len(queue) > 0:
        if mode == "large":
            query = queue.popleft()
            # print(f"popleft {query.pred_peakmem}")
            if memory_in_use + query.pred_peakmem > memory_limit_kb:
                # print(f"switching to small")
                mode = 'small'
                # insert query at front of queue
                queue.appendleft(query)
            else:
                # Add query's memory usage and duration to the current schedule
                memory_in_use += query.pred_peakmem
                # logging.info(f"now query: {query.id}, memory_in_use: {memory_in_use} KB, memory_limit_kb: {memory_limit_kb} KB, peakmem: {query.pred_peakmem} KB")
                query.start_time = current_time
                query.end_time = current_time + query.pred_duration
                query.type = 'large'
                running_queries.append(query)
                bf_schedule.append(query)
        if mode == "small":
            query = queue.pop()  # default popright()
            earliest_finish_large_query = None
            earliest_time = float('inf')
            for q in running_queries:
                if q.type == 'large':
                    if q.end_time < earliest_time:
                        earliest_finish_large_query = q
                        earliest_time = q.end_time
            print(f"current_time: {current_time}, duration: {query.pred_duration}, earliest_time: {earliest_time}")
            # if memory_in_use + query.pred_peakmem < memory_limit_kb and earliest_time < current_time + query.pred_duration:
                # print(f"####################### memory allows but time does not allow #############################")
            if memory_in_use + query.pred_peakmem > memory_limit_kb or earliest_time < current_time + query.pred_duration:
                # print(f"memory exceeding")
                # Simulate waiting (this could be enhanced with more detailed time management)
                earliest_finish_query = min(running_queries, key=lambda q: q.end_time)
                current_time = earliest_finish_query.end_time
                # delete earliest_finish_query from running_queries
                running_queries.remove(earliest_finish_query)
                # Update memory_in_use as the sum of remaining running queries, excluding the one finishing first
                memory_in_use = sum(q.pred_peakmem for q in running_queries)
                mode = 'large'
                queue.append(query)
            else:
                # print(f"!!!!!!!! memory allows and time allows !!!!!!!!!!")
                # Add query's memory usage and duration to the current schedule
                memory_in_use += query.pred_peakmem
                # logging.info(f"now query: {query.id}, memory_in_use: {memory_in_use} KB, memory_limit_kb: {memory_limit_kb} KB, peakmem: {query.pred_peakmem} KB")

                query.start_time = current_time
                query.end_time = current_time + query.pred_duration
                query.type = 'small'
                running_queries.append(query)
                bf_schedule.append(query)

        
    makespan = max(q.end_time for q in sorted_queries)

    return bf_schedule, makespan



# ----------------------------
# Main Function to Compare Strategies
# ----------------------------
def main():
    # Removed duplicate logging configuration to avoid conflicts
    
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--no_naive', action='store_true', help='Do not execute Naive Strategy.')
    argparser.add_argument('--num_queries', type=int, default=7, help='Number of queries to execute.')
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
    

    total_query_memory_limit_kb = 1.125 * 1024**2 # 1180000
    print(f"total_query_memory_limit_kb: {total_query_memory_limit_kb} KB")
    
    queries = []

    queries.append(Query(
        id=1,
        pred_peakmem=545000,
        pred_duration=39.2775,
    ))
    queries.append(Query(   
        id=2,
        pred_peakmem=490000,
        pred_duration=39.2775,
    ))
    queries.append(Query(
        id=3,
        pred_peakmem=445000,
        pred_duration=26.5, 
    ))
    queries.append(Query(
        id=4,
        pred_peakmem=380000,
        pred_duration=38,
    ))
    queries.append(Query(
        id=5,
        pred_peakmem=320000,
        pred_duration=40.2,
    ))
    queries.append(Query(
        id=6,
        pred_peakmem=255000,
        pred_duration=27,
    ))
    queries.append(Query(
        id=7,
        pred_peakmem=137400,
        pred_duration=39.2711,
    ))

    # import random
    # queries = random.sample(queries, args.num_queries)


    optimal_schedule, optimal_makespan = schedule_queries(queries, total_query_memory_limit_kb)
    # Print out the optimal schedule and makespan
    print("Optimal Query Schedule (ID order):")
    for query in optimal_schedule:
        print(f"Query ID: {query.id}, Peak Memory: {query.pred_peakmem} KB, Duration: {query.pred_duration} sec, start_time: {query.start_time}, end_time: {query.end_time}")
    print(f"Optimal Makespan: {optimal_makespan} sec")

    # optimal_schedule, optimal_makespan = schedule_queries(queries, total_query_memory_limit_kb)
    ffd_schedule, ffd_makespan = schedule_queries_ffd(queries, total_query_memory_limit_kb)
    print("FFD Query Schedule (ID order):")
    for query in ffd_schedule:
        print(f"Query ID: {query.id}, Peak Memory: {query.pred_peakmem} KB, Duration: {query.pred_duration} sec, start_time: {query.start_time}, end_time: {query.end_time}")
    print(f"FFD Makespan: {ffd_makespan} sec")


    bf_schedule, bf_makespan = schedule_queries_bf(queries, total_query_memory_limit_kb)
    print("BF Query Schedule (ID order):")
    for query in bf_schedule:
        print(f"Query ID: {query.id}, Peak Memory: {query.pred_peakmem} KB, Duration: {query.pred_duration} sec, start_time: {query.start_time}, end_time: {query.end_time}")
    print(f"BF Makespan: {bf_makespan} sec")
    


# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    main()