# sudo setcap cap_sys_ptrace+ep /home/wuy/software/anaconda3/envs/zsce/bin/python3.8


import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import concurrent.futures
import threading
import time
import psutil
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
import heapq  # For priority queue implementation
from collections import deque
from datetime import datetime
import subprocess
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from GIN import GIN
from parse_plan import parse_plan

def get_postgres_memory_limit_kb():
    try:
        # Run systemctl command to get MemoryMax for PostgreSQL
        result = subprocess.run(
            ["systemctl", "show", "postgresql", "--property=MemoryMax"],
            capture_output=True, text=True, check=True
        )

        # Parse the output
        output = result.stdout.strip()
        if "=" in output:
            _, value = output.split("=")
            value = value.strip()

            # Convert bytes to KB
            if value.isdigit():  # If the value is a number (in bytes)
                memory_kb = int(value) // 1024  # Convert to KB
                return memory_kb
            elif value == "infinity":  # No limit set
                return -1  # Use -1 to indicate "infinity"

        return None  # MemoryMax not found
    
    except subprocess.CalledProcessError as e:
        return None
    
    
def get_process_swap_memory(pid):
    """
    Get swap memory usage for a given process using /proc/<pid>/smaps.
    """
    swap_memory = 0
    try:
        with open(f'/proc/{pid}/smaps', 'r') as smaps:
            for line in smaps:
                if line.startswith("Swap:"):
                    swap_memory += int(line.split()[1])  # Swap memory is in KB
    except Exception as e:
        print(f"Error reading swap memory for PID {pid}: {e}")
    return swap_memory


def monitor_postgres_memory(interval, metrics, key_prefix, stop_event):
    """
    Monitor memory usage of all PostgreSQL processes, including swap memory.
    """
    while not stop_event.is_set():
        total_swap_memory = 0
        total_memory = 0
        try:
            for proc in psutil.process_iter(['name']):
                if 'postgres' in proc.info['name']:
                    try:
                        mem_info = proc.memory_info()
                        rss = mem_info.rss // 1024  # Resident memory in KB
                        swap = get_process_swap_memory(proc.pid)  # Swap memory in KB
                        total_swap_memory += swap
                        total_memory += rss
                    except psutil.NoSuchProcess:
                        continue
                    except Exception as e:
                        print(f"Error monitoring process {proc.pid}: {e}")
            # Record the metrics
            metrics[key_prefix]['time'].append(time.time())
            metrics[key_prefix]['swap_mem'].append(total_swap_memory)  # In KB
            metrics[key_prefix]['total_mem'].append(total_memory)  # In KB
        except Exception as e:
            print(f"Error monitoring PostgreSQL processes: {e}")
            break
        time.sleep(interval)


@dataclass(order=True)
class PrioritizedQuery:
    priority: float
    query: 'Query' = field(compare=False)
    enqueue_time: float = field(compare=False, default_factory=time.time)
    start_time: Optional[float] = field(compare=False, default=None)
    retry_count: int = field(compare=False, default=0)
    next_available_time: float = field(compare=False, default_factory=lambda: time.time())

@dataclass
class Query:
    id: int
    sql: str
    explain_json_plan: Dict[str, Any]
    pred_peakmem: int
    pred_duration: float
    type: str # large or small


class DequeQueue:
    def __init__(self):
        self.deque = deque()
        self.lock = threading.Lock()
    
    def push_back(self, prioritized_query: PrioritizedQuery):
        with self.lock:
            self.deque.append(prioritized_query)

    def push_front(self, prioritized_query: PrioritizedQuery):
        with self.lock:
            self.deque.appendleft(prioritized_query)
    
    def pop_front(self) -> Optional[PrioritizedQuery]:
        with self.lock:
            return self.deque.popleft() if self.deque else None
    
    def pop_back(self) -> Optional[PrioritizedQuery]:
        with self.lock:
            return self.deque.pop() if self.deque else None
    
    def is_empty(self) -> bool:
        with self.lock:
            return len(self.deque) == 0
    
    def peek_front(self) -> Optional[PrioritizedQuery]:
        with self.lock:
            return self.deque[0] if self.deque else None
    
    def peek_back(self) -> Optional[PrioritizedQuery]:
        with self.lock:
            return self.deque[-1] if self.deque else None
        
    def peek_next_available_time(self, mode):
        with self.lock:
            if self.deque:
                if mode == 'large':
                    return self.deque[0].next_available_time
                elif mode =='small':
                    return self.deque[-1].next_available_time
            return None
        
    def size(self) -> int:
        with self.lock:
            return len(self.deque)

# ----------------------------
# Priority Queue Implementation
# ----------------------------
class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.lock = threading.Lock()
    
    def push(self, prioritized_query: PrioritizedQuery):
        with self.lock:
            heapq.heappush(self.heap, prioritized_query)
    

    def pop_ready_queries(self, current_time: float) -> List[PrioritizedQuery]:
        ready = []
        with self.lock:
            while self.heap and self.heap[0].next_available_time <= current_time:
                ready.append(heapq.heappop(self.heap))
        return ready    

    def peek_next_available_time(self) -> Optional[float]:
        with self.lock:
            if self.heap:
                return self.heap[0].next_available_time
            return None
    
    def is_empty(self) -> bool:
        with self.lock:
            return len(self.heap) == 0

# ----------------------------
# Function to Retrieve PostgreSQL Memory Settings using SQLAlchemy
# ----------------------------
def get_postgres_memory_settings(engine: Engine) -> Optional[Dict[str, Any]]:
    """
    Retrieves PostgreSQL memory settings using SHOW commands.

    :param engine: SQLAlchemy Engine instance.
    :return: Dictionary containing memory settings or None if failed.
    """
    settings = {}
    try:
        with engine.connect() as conn:
            # Execute SHOW commands
            result = conn.execute(text("SHOW shared_buffers;"))
            shared_buffers = result.fetchone()[0]
            settings['shared_buffers'] = parse_memory_setting(shared_buffers)
            
            result = conn.execute(text("SHOW work_mem;"))
            work_mem = result.fetchone()[0]
            settings['work_mem'] = parse_memory_setting(work_mem)
            
            result = conn.execute(text("SHOW maintenance_work_mem;"))
            maintenance_work_mem = result.fetchone()[0]
            settings['maintenance_work_mem'] = parse_memory_setting(maintenance_work_mem)
            
            result = conn.execute(text("SHOW max_connections;"))
            max_connections = result.fetchone()[0]
            settings['max_connections'] = int(max_connections)
    except Exception as e:
        logging.error(f"Error retrieving PostgreSQL memory settings: {e}")
        return None
    return settings

def parse_memory_setting(setting: str) -> int:
    """
    Parses PostgreSQL memory settings and converts them to KB.

    :param setting: Memory setting as a string (e.g., '4GB', '512MB', '64kB').
    :return: Memory in KB as an integer.
    """
    units = {
        'kB': 1,
        'KB': 1,
        'MB': 1024,
        'GB': 1024 * 1024,
        'k': 1,
        'm': 1024,
        'g': 1024 * 1024,
    }
    number = ''
    unit = ''
    for char in setting:
        if char.isdigit() or char == '.':
            number += char
        else:
            unit += char
    try:
        number = float(number)
    except ValueError:
        number = 0
    unit = unit.strip()
    multiplier = units.get(unit, 1)  # Default to kB if unit is unrecognized
    return int(number * multiplier)

def get_postgres_background_memory_usage():
    process_specific_memory_kb = 0
    try:
        # Use psutil to get the process-specific memory usage
        for proc in psutil.process_iter(['name', 'memory_info']):
            if 'postgres' in proc.info['name']:
                # Subtract shared_buffers_kb from each process to avoid double counting
                rss_kb = proc.info['memory_info'].rss // 1024
                process_specific_memory_kb += rss_kb

        # Total memory is the shared_buffers plus the unique memory usage of each process
        total_memory_kb = process_specific_memory_kb
        return total_memory_kb

    except Exception as e:
        print(f"Error: {e}")
        return None

def get_postgres_memory_usage(shared_buffers_kb):
    process_specific_memory_kb = 0
    process_memory_list = []
    matched_processes = 0
    try:
        # Use psutil to get the process-specific memory usage
        for proc in psutil.process_iter(['name', 'memory_info']):
            if 'postgres' in proc.info['name']:
                # Subtract shared_buffers_kb from each process to avoid double counting
                rss_kb = proc.info['memory_info'].rss // 1024
                process_specific_memory_kb += rss_kb
                process_memory_list.append(rss_kb)
                # process_specific_memory_kb += max(rss_kb - shared_buffers_kb, 0)
                # if rss_kb > shared_buffers_kb:
                #     process_specific_memory_kb += max(rss_kb - shared_buffers_kb, 0)
                #     process_memory_list.append(max(rss_kb - shared_buffers_kb, 0))
                # else:
                #     process_specific_memory_kb += max(rss_kb, 0)
                #     process_memory_list.append(max(rss_kb, 0))
                # process_specific_memory_kb += rss_kb
                matched_processes += 1

        # logging.debug(f"process_memory_list: {[i/1024 for i in process_memory_list]} MB")
        # Total memory is the shared_buffers plus the unique memory usage of each process
        total_memory_kb = shared_buffers_kb + process_specific_memory_kb
        # if matched_processes == 0:
        #     total_memory_kb = process_specific_memory_kb 
        # else:
        #     total_memory_kb = process_specific_memory_kb - shared_buffers_kb
        return total_memory_kb

    except Exception as e:
        print(f"Error: {e}")
        return None

# def get_postgres_memory_usage(shared_buffers_kb):
#     return 0
#     process_specific_memory_kb = 0
#     try:
#         for proc in psutil.process_iter(['name', 'pid']):
#             if 'postgres' in proc.info['name']:
#                 rss_process_kb = 0
#                 pid = proc.info['pid']
#                 smaps_path = f'/proc/{pid}/smaps'
                
#                 try:
#                     with open(smaps_path, 'r') as smaps_file:
#                         for line in smaps_file:
#                             if line.startswith('Rss:'):  # Read only RSS memory
#                                 rss_kb = int(line.split()[1])
#                                 rss_process_kb += rss_kb
#                 except FileNotFoundError:
#                     continue
#                 process_specific_memory_kb += max(rss_process_kb - shared_buffers_kb, 0)
            
#         return process_specific_memory_kb
#     except Exception as e:
#         print(f"Error reading smaps: {e}")
#         return 0

    


# ----------------------------
# Naive Strategy Implementation
# ----------------------------
class NaiveStrategy:
    def __init__(
        self,
        engine: Engine,
        queries: List[Query],
        executor: concurrent.futures.ThreadPoolExecutor,
        max_retries: int = 5,
        base_wait_time: float = 2.0,
        exp: int = 0,
        exp_num: int = 0,
        total_query_memory_limit_kb: int = 0
    ):
        """
        :param engine: The SQLAlchemy Engine instance.
        :param queries: List of Query objects.
        :param executor: ThreadPoolExecutor to manage concurrency.
        :param max_retries: Maximum number of retries for each query.
        :param base_wait_time: Base wait time for retries in seconds.
        """
        self.engine = engine
        self.queries = queries
        self.results = {}
        self.lock = threading.Lock()
        self.executor = executor
        self.max_retries = max_retries
        self.base_wait_time = base_wait_time
        self.exp = exp
        self.exp_num = exp_num
        self.success_count = 0
        
        self.total_query_memory_limit_kb = total_query_memory_limit_kb

        self.running_queries = {}


    def naive_execute_query(
        self,
        executor: concurrent.futures.ThreadPoolExecutor,
        engine: Engine,
        prioritized_query: PrioritizedQuery,
        lock: threading.Lock,
        strategy: str,
        max_retries: int = 10,
        base_wait_time: float = 2.0,
        exp: int = 0,
        exp_num: int = 0
    ):
        """
        Executes a single query in naive strategy and records execution and waiting times.
        Adjusts priority based on retry attempts.

        :return: Tuple containing (query_id, success, error_message)
        """
        query = prioritized_query.query
        query_id = query.id

        while prioritized_query.retry_count < max_retries:
            # Record the start time of execution
            start_exec_time = time.time()
            prioritized_query.start_time = start_exec_time  # Set start_time to avoid None
            self.running_queries[prioritized_query.query.id] = prioritized_query.query.pred_peakmem

            try:
                with engine.connect() as conn:
                    max_connection = self.total_query_memory_limit_kb // max([v for v in self.running_queries.values()])
                    conn.execute(text(f"SET max_connections = {max_connection}"))
                    logging.info(f"Setting max_connections to {max_connection}")
                    logging.debug(f"{strategy}: Executing Query {query_id} whose retry is {prioritized_query.retry_count}...")
                    # Execute the query
                    result = conn.execute(text(query.sql))
                    result.fetchall()
                    del self.running_queries[prioritized_query.query.id]
                
                end_exec_time = time.time()
                exec_time = end_exec_time - start_exec_time
                total_time = end_exec_time - prioritized_query.enqueue_time

                # Update result_dict with execution time and waiting time
                with lock:
                    self.results[query_id] = {
                        'execution_time': exec_time,
                        'total_time': total_time,
                        'success': True,
                        'retry_count': prioritized_query.retry_count
                    }
                    self.success_count += 1
                
                logging.info(f"{strategy}({exp+1}/{exp_num}): Query {query_id} executed in {exec_time:.2f} seconds. Total time: {total_time:.2f} seconds. its retry is {prioritized_query.retry_count}. success_count: {self.success_count}")
                
                return (query_id, True, None)  # Success

            except Exception as e:
                error_message = str(e)
                prioritized_query.retry_count += 1
                prioritized_query.priority = prioritized_query.priority - 1
                wait_time = min(base_wait_time ** prioritized_query.retry_count, 2)

                logging.debug(f"{strategy}: Query {query_id} failed with error {error_message} and will sleep for {wait_time:.2f} seconds before retrying...")
                time.sleep(wait_time)  # Wait before retrying

        # If all retries failed
        with lock:
            total_time = time.time() - prioritized_query.enqueue_time
            self.result_dict[query_id] = {
                'execution_time': float('inf'),
                'total_time': total_time,
                'success': False,
                'error_message': "Max retries exceeded."
            }
        logging.error(f"{strategy}: Query {query_id} failed after {max_retries} retries.")
        return (query_id, False, "Max retries exceeded.")  # Failure after max retries


    def execute(self) -> float:
        """
        Executes all queries concurrently without considering memory constraints.
        Relies on execute_query to handle retries.

        :return: Total execution time in seconds.
        """
        start_time = time.time()

        futures = {}
        for query in self.queries:
            priority_value = 0
            self.prioritized_query = PrioritizedQuery(
                priority=priority_value,
                query=query,
                enqueue_time=time.time()
            )
            future = self.executor.submit(
                self.naive_execute_query,
                self.executor,
                self.engine,
                self.prioritized_query,
                self.lock,
                'naive',
                self.max_retries,
                self.base_wait_time,
                self.exp,
                self.exp_num
            )
            futures[future] = query

        # Collect results
        for future in concurrent.futures.as_completed(futures):
            query = futures[future]
            try:
                query_id, success, error_message = future.result()
                if not success:
                    logging.warning(f"Naive Strategy: Query {query.id} failed after {self.max_retries} retries.")
            except Exception as e:
                logging.error(f"Naive Strategy: Unexpected error with Query {query.id}: {e}")

        end_time = time.time()
        total_exec_time = end_time - start_time
        logging.info(f"Naive Strategy Total Execution Time: {total_exec_time:.2f} seconds.")
        return total_exec_time


# ----------------------------
# Function to Load Queries from JSON File
# ----------------------------
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

def monitor_memory_spill(engine: Engine) -> Dict[str, Any]:
    """
    Monitor how much memory is spilled to disk by checking the temp_files and temp_bytes statistics
    in the PostgreSQL database.

    :param engine: SQLAlchemy Engine instance.
    :return: A dictionary containing database name and memory spilled to disk in a readable format.
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT datname, pg_size_pretty(temp_bytes / temp_files) AS overflow
                FROM pg_stat_database
                WHERE temp_files > 0;
            """))
            memory_spills = {}
            for row in result:
                # Access the columns by index instead of key
                memory_spills[row[0]] = row[1]
            return memory_spills
    except Exception as e:
        logging.error(f"Failed to retrieve memory spill data: {e}")
        return {}



# Call this function periodically or after each strategy execution
def log_memory_spill(engine: Engine, strategy_name: str):
    """
    Log the memory spill information after executing a strategy.

    :param engine: SQLAlchemy Engine instance.
    :param strategy_name: The name of the strategy (e.g., 'naive' or 'memory-based').
    """
    memory_spills = monitor_memory_spill(engine)
    if memory_spills:
        logging.info(f"{strategy_name} Strategy: Memory spilled to disk (temp_bytes/temp_files):")
        for db_name, spill_size in memory_spills.items():
            if db_name == 'airline':
                logging.info(f"  Database: {db_name}, Memory Spilled: {spill_size}")
    else:
        logging.info(f"{strategy_name} Strategy: No memory spilled to disk.")

import os
def save_strategy_results(save_dir, bf_strategy_results, ffd_strategy_results, naive_strategy_results):
    try:
        # Save bf strategy results
        with open(os.path.join(save_dir,'bf_strategy_results.json'), 'w') as memory_file:
            json.dump(bf_strategy_results, memory_file, indent=4)
        print("Memory-based strategy results saved to 'bf_strategy_results.json'.")

        # Save ffd strategy results
        with open(os.path.join(save_dir, 'ffd_strategy_results.json'), 'w') as ffd_file:
            json.dump(ffd_strategy_results, ffd_file, indent=4)
        print("FFD strategy results saved to 'ffd_strategy_results.json'.")

        # Save naive strategy results
        with open(os.path.join(save_dir, 'naive_strategy_results.json'), 'w') as naive_file:
            json.dump(naive_strategy_results, naive_file, indent=4)
        print("Naive strategy results saved to 'naive_strategy_results.json'.")
    except Exception as e:
        print(f"Error saving results: {e}")

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

    # PostgreSQL connection parameters
    connection_params = {
        'dbname': args.dataset,
        'user': 'wuy',
        'password': 'wuy',
        'host': 'localhost',
        'port': 5432
    }

    # Create SQLAlchemy Engine with connection pool
    try:
        engine = create_engine(
            "postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}".format(**connection_params),
            pool_size=1000,          # Adjust based on max_connections
            max_overflow=0,        # No additional connections beyond pool_size
            pool_timeout=30,       # Timeout for getting connection
            pool_recycle=5,      # Recycle connections after 30 minutes
            pool_pre_ping=True,    # Ping connections before using them
        )
        logging.info("SQLAlchemy Engine created successfully.")
    except Exception as e:
        logging.error(f"Error creating SQLAlchemy Engine: {e}")
        return

    # Retrieve PostgreSQL memory settings
    memory_settings = get_postgres_memory_settings(engine)
    if not memory_settings:
        logging.error("Failed to retrieve memory settings. Exiting.")
        return

    shared_buffers_kb = memory_settings.get('shared_buffers', 0)
    work_mem_kb = memory_settings.get('work_mem', 0)
    maintenance_work_mem_kb = memory_settings.get('maintenance_work_mem', 0)
    max_connections = memory_settings.get('max_connections', 90)

    logging.info(f"PostgreSQL Memory Settings:")
    logging.info(f"shared_buffers = {shared_buffers_kb} KB")
    logging.info(f"work_mem = {work_mem_kb} KB")
    logging.info(f"maintenance_work_mem = {maintenance_work_mem_kb} KB")
    logging.info(f"max_connections = {max_connections}")

    # Define a buffer to account for administrative connections
    admin_connection_buffer = 3
    adjusted_max_connections = max_connections - admin_connection_buffer
    if adjusted_max_connections <= 0:
        logging.error("Adjusted max_connections is non-positive. Increase PostgreSQL's max_connections.")
        return

    logging.info(f"Adjusted max_connections for connection pool: {adjusted_max_connections}")

    # ----------------------------
    # Dynamic Calculation of Available Memory
    # ----------------------------
    # Total system memory in KB
    available_memory_kb = psutil.virtual_memory().available // 1024
    # available_memory_kb = int(get_postgres_memory_limit_kb())

    postgres_background_memory_kb = get_postgres_background_memory_usage()
    logging.info(f"PostgreSQL background memory usage: {postgres_background_memory_kb} KB, total available memory: {available_memory_kb} KB")
    available_memory_kb += postgres_background_memory_kb
    # available_memory_kb = 56 * 1024**2

    # # ----------------------------
    # # Improved PostgreSQL Memory Estimation
    # # ----------------------------
    # # Per-connection overhead (adjust based on your environment)
    # per_connection_overhead_kb = 10 * 1024  # Assuming 10 MB per connection

    # # Number of active connections or expected peak load (e.g., concurrent queries)
    # active_connections = adjusted_max_connections

    # # Estimate memory based on active queries and complexity
    # average_sort_hash_operations_per_query = 2  # Estimate based on typical queries

    # # Static memory usage
    # static_memory_usage_kb = shared_buffers_kb + maintenance_work_mem_kb

    # # Dynamic memory usage for concurrent queries
    # dynamic_query_memory_usage_kb = (
    #     active_connections * per_connection_overhead_kb +
    #     active_connections * average_sort_hash_operations_per_query * work_mem_kb
    # )

    # # Total estimated PostgreSQL memory usage
    # estimated_pg_memory_kb = static_memory_usage_kb + dynamic_query_memory_usage_kb
    # logging.info(f"Estimated PostgreSQL memory usage (static + dynamic): {estimated_pg_memory_kb} KB")

    # # Ensure the memory limit doesn't exceed container/VM capacity
    # total_query_memory_limit_kb = min(estimated_pg_memory_kb, available_memory_kb)
    total_query_memory_limit_kb = available_memory_kb
    
    logging.info(f"Adjusted total memory limit for query operations: {total_query_memory_limit_kb} KB")

    

    # Load queries from JSON file
    plan_file = f'/home/wuy/DB/pg_mem_data/{args.dataset}/total_plans.json'
    queries = load_queries(plan_file, total_query_memory_limit_kb)
    # queries = queries[:args.num_queries]  # Limit to 100 queries for testing
    # if not queries:
    #     logging.error("No queries to execute. Exiting.")
    #     engine.dispose()
    #     return
    queries = sorted(queries, key=lambda x: x.pred_peakmem, reverse=True)

    final_queries = []
    large_num = args.num_queries // 2
    small_num = args.num_queries - large_num
    final_queries.extend(queries[:large_num])
    final_queries.extend(queries[-small_num:])
    queries = final_queries
    # for q in queries:
    #     logging.debug(f"Query {q.id}: {q.pred_peakmem}")
    # queries = queries[:args.num_queries]  # Limit to 100 queries for testing
    if not queries:
        logging.error("No queries to execute. Exiting.")
        engine.dispose()
        return

    sum_duration = sum(q.pred_duration for q in queries)
    logging.info(f"Total duration of queries: {sum_duration} seconds.")
    # exit()

    # ----------------------------
    # Initialize ThreadPoolExecutor with max_workers equal to adjusted_max_connections
    # ----------------------------
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=adjusted_max_connections)
    logging.info(f"Initialized ThreadPoolExecutor with {adjusted_max_connections} workers.")

    max_retries = 10000

        
    with open('/home/wuy/DB/pg_mem_data/combined_statistics_workload.json') as f:
        statistics = json.load(f)

    model = GIN(hidden_channels=32, out_channels=1, num_layers=6, num_node_features=23, dropout=0.5)
    logging.info(f"Loading checkpoint")
    model.load_state_dict(torch.load('GIN_airline_carcinogenesis_hepatitis_financial_geneea_tpch_sf1_tpcds_sf1_mem__best.pth', map_location=args.device))
    model = model.to(args.device)
    model.eval()
    logging.info(f"Model loaded")

    interval = 0.2
    metrics = {
        'naive': {'time': [], 'swap_mem': [], 'total_mem': []},
        'bf': {'time': [], 'swap_mem': [], 'total_mem': []},
        'ffd': {'time': [], 'swap_mem': [], 'total_mem': []}
    }

    # Stop events for monitoring threads
    naive_stop_event = threading.Event()
    bf_stop_event = threading.Event()
    ffd_stop_event = threading.Event()

    # Threads for monitoring
    naive_thread = threading.Thread(target=monitor_postgres_memory, args=(interval, metrics, 'naive', naive_stop_event))
    bf_based_thread = threading.Thread(target=monitor_postgres_memory, args=(interval, metrics, 'bf', bf_stop_event))
    ffd_based_thread = threading.Thread(target=monitor_postgres_memory, args=(interval, metrics, 'ffd', ffd_stop_event))

    

    # ----------------------------
    # Execute Naive Strategy Multiple Times
    # ----------------------------
    if not args.no_naive:
        
        naive_total_time_list = []
        naive_waiting_sum_list = []
        import random
        # random permutate queries
        random_queries = random.sample(queries, args.num_queries)
        for i in range(args.exp_num):
            logging.info(f"\nExecuting Naive Strategy - Run {i+1}/{args.exp_num}:")
            # Initialize a new NaiveStrategy instance for each run
            naive_strategy = NaiveStrategy(
                engine=engine,
                queries=random_queries,
                executor=executor,
                max_retries=max_retries,  # Set as needed
                base_wait_time=1.1,
                exp = i,
                exp_num = args.exp_num,
                total_query_memory_limit_kb = total_query_memory_limit_kb
            )
            naive_thread.start()
            naive_total_time = naive_strategy.execute()
            naive_stop_event.set()
            naive_thread.join()

            naive_total_time_list.append(naive_total_time)

            # Log memory spill after strategy execution
            log_memory_spill(engine, 'naive')

            # Calculate sum of waiting times
            naive_waiting_sum = sum(
                info['total_time'] for info in naive_strategy.results.values() if 'total_time' in info
            )
            naive_waiting_sum_list.append(naive_waiting_sum)

    

    # ----------------------------
    # Compare Performance
    # ----------------------------
    logging.info("\nComparison of Strategies:")
    mean_naive_time = mean(naive_total_time_list) if not args.no_naive else 0

    mean_naive_waiting = mean(naive_waiting_sum_list) if not args.no_naive else 0

    

    naive_total_retry_count = sum(
        info['retry_count'] for info in naive_strategy.results.values() if'retry_count' in info
    ) if not args.no_naive else 0


    naive_max_retry_count = max(
        info['retry_count'] for info in naive_strategy.results.values() if'retry_count' in info
    ) if not args.no_naive else 0

    
    if not args.no_naive:
        logging.info(f"Naive Strategy Average Total Execution Time: {mean_naive_time:.2f} seconds.")
        logging.info(f"Naive Strategy Average Sum of Waiting Times: {mean_naive_waiting:.2f} seconds.")
        logging.info(f"Total retries for Naive Strategy: {naive_total_retry_count}")
        logging.info(f"Naive Strategy Max Retries: {naive_max_retry_count}\n")

    

    

    # Plot the results
    # plot_memory_metrics(metrics, result_dir=result_path)
    import pickle
    with open(f'{args.num_queries}_metrics.pkl','wb') as f:
        pickle.dump(metrics, f)


    # ----------------------------
    # Close the Engine and Executor
    # ----------------------------
    engine.dispose()
    executor.shutdown(wait=True)
    logging.info("Connection pool and executor shut down successfully.")

# ----------------------------
# Helper Function to Calculate Mean
# ----------------------------
def mean(numbers: List[float]) -> float:
    """
    Calculates the mean of a list of numbers.

    :param numbers: List of float numbers.
    :return: Mean value.
    """
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    main()