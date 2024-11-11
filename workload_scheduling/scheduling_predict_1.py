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
from datetime import datetime
import subprocess
import torch
from torch_geometric.data import Data

from GIN import GIN
from parse_plan import parse_plan


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



# ### ----------------------------
# ### Function to Get PostgreSQL Process Memory Usage
# ### ----------------------------
# def get_postgres_memory_usage(shared_buffers_kb) -> int:
#     """
#     Returns the total memory usage of all PostgreSQL processes in KB by leveraging
#     a system call with 'pgrep' and 'ps'.
#     """
#     try:
#         # Use subprocess to execute the shell command
#         result = subprocess.run(
#             "pgrep postgres | xargs ps -o rss= -p | awk '{s+=$1} END {print s}'",
#             shell=True,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True,
#         )
        
#         # Check for any errors during the command execution
#         if result.returncode != 0:  
#             raise Exception(f"Error in fetching memory usage: {result.stderr.strip()}")
        
#         # Convert the result to an integer (result is in KB already)
#         total_memory_kb = int(result.stdout.strip())  # Strip to remove any extra spaces or newlines
#         return total_memory_kb
    
#     except Exception as e:
#         # Handle any exception and return 0 in case of failure
#         logging.error(f"Failed to get PostgreSQL memory usage: {e}")
#         return 0

# def get_postgres_memory_usage(shared_buffers_kb):
#     process_specific_memory_kb = 0

#     try:
#         # Execute the shell command to get total RSS of PostgreSQL processes
#         result = subprocess.run(
#             "pgrep postgres | xargs ps -o rss= -p | awk '{s+=$1} END {print s}'",
#             shell=True,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True,
#         )

#         if result.returncode != 0:
#             raise Exception(f"Error in fetching memory usage: {result.stderr.strip()}")

#         # Convert the result to an integer (result is in KB already)
#         total_rss_kb = int(result.stdout.strip())

#         # Subtract shared_buffers_kb to avoid double counting
#         process_specific_memory_kb = max(total_rss_kb - shared_buffers_kb, 0)

#         # Total memory is the shared_buffers plus the unique memory usage of each process
#         total_memory_kb = shared_buffers_kb + process_specific_memory_kb
#         return total_memory_kb

#     except Exception as e:
#         print(f"Error: {e}")
#         return None

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
    try:
        # Use psutil to get the process-specific memory usage
        for proc in psutil.process_iter(['name', 'memory_info']):
            if 'postgres' in proc.info['name']:
                # Subtract shared_buffers_kb from each process to avoid double counting
                rss_kb = proc.info['memory_info'].rss // 1024
                process_specific_memory_kb += max(rss_kb - shared_buffers_kb, 0)

        # Total memory is the shared_buffers plus the unique memory usage of each process
        total_memory_kb = shared_buffers_kb + process_specific_memory_kb
        return total_memory_kb

    except Exception as e:
        print(f"Error: {e}")
        return None
    

    
# ----------------------------
# Function to Execute a Single Query using SQLAlchemy
# ----------------------------
def execute_query(
    executor: concurrent.futures.ThreadPoolExecutor,
    engine: Engine,
    memory_based_priority_queue: PriorityQueue,
    prioritized_query: PrioritizedQuery,
    result_dict: Dict[int, Dict[str, Any]],
    lock: threading.Lock,
    strategy: str,
    max_retries: int = 10,
    base_wait_time: float = 2.0,
    total_memory_kb: float = 0,
    exp: int = 0,
    exp_num: int = 0
):
    """
    Executes a single query and records execution and waiting times.
    Adjusts priority based on retry attempts.

    :return: Tuple containing (query_id, success, error_message)
    """
    query = prioritized_query.query
    query_id = query.id

    # Record the start time of execution
    start_exec_time = time.time()
    prioritized_query.start_time = start_exec_time  # Set start_time to avoid None

    try:
        with engine.connect() as conn:
            # wait_for_available_memory(prioritized_query, total_memory_kb)
            logging.debug(f"{strategy}: Executing Query {query_id} whose retry is {prioritized_query.retry_count}...")
            # Execute the query
            result = conn.execute(text(query.sql))
            result.fetchall()
        
        end_exec_time = time.time()
        exec_time = end_exec_time - start_exec_time
        total_time = end_exec_time - prioritized_query.enqueue_time

        # Update result_dict with execution time and waiting time
        with lock:
            result_dict[query_id] = {
                'execution_time': exec_time,
                'total_time': total_time,
                'success': True
            }
        
        logging.info(f"{strategy}({exp+1}/{exp_num}): Query {query_id} executed in {exec_time:.2f} seconds. Total time: {total_time:.2f} seconds. its retry is {prioritized_query.retry_count}.")
        
        return (prioritized_query, True, None)  # Success

    except Exception as e:
        error_message = str(e)
        
        prioritized_query.retry_count += 1
        prioritized_query.priority = prioritized_query.priority - 1

        return (prioritized_query, False, error_message)

def naive_execute_query(
    executor: concurrent.futures.ThreadPoolExecutor,
    engine: Engine,
    prioritized_query: PrioritizedQuery,
    result_dict: Dict[int, Dict[str, Any]],
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

        try:
            with engine.connect() as conn:
                logging.debug(f"{strategy}: Executing Query {query_id} whose retry is {prioritized_query.retry_count}...")
                # Execute the query
                result = conn.execute(text(query.sql))
                result.fetchall()
            
            end_exec_time = time.time()
            exec_time = end_exec_time - start_exec_time
            total_time = end_exec_time - prioritized_query.enqueue_time

            # Update result_dict with execution time and waiting time
            with lock:
                result_dict[query_id] = {
                    'execution_time': exec_time,
                    'total_time': total_time,
                    'success': True
                }
            
            logging.info(f"{strategy}({exp+1}/{exp_num}): Query {query_id} executed in {exec_time:.2f} seconds. Total time: {total_time:.2f} seconds. its retry is {prioritized_query.retry_count}.")
            
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
        result_dict[query_id] = {
            'execution_time': float('inf'),
            'total_time': total_time,
            'success': False,
            'error_message': "Max retries exceeded."
        }
    logging.error(f"{strategy}: Query {query_id} failed after {max_retries} retries.")
    return (query_id, False, "Max retries exceeded.")  # Failure after max retries

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
        exp_num: int = 0
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
            prioritized_query = PrioritizedQuery(
                priority=priority_value,
                query=query,
                enqueue_time=time.time()
            )
            future = self.executor.submit(
                naive_execute_query,
                self.executor,
                self.engine,
                prioritized_query,
                self.results,
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
# Memory-Based Strategy Implementation
# ----------------------------
class MemoryBasedStrategy:
    def __init__(
        self,
        model,
        statistics,
        engine: Engine,
        queries: List[Query],
        total_memory_kb: int,
        work_mem_kb: int,
        shared_buffers_kb: int,
        executor: concurrent.futures.ThreadPoolExecutor,
        max_retries: int = 5,
        base_wait_time: float = 2.0,
        exp: int = 0,
        exp_num: int = 0,
        device: str = 'cpu'
    ):
        """
        :param engine: The SQLAlchemy Engine instance.
        :param queries: List of Query objects.
        :param total_memory_kb: Total memory allocated to PostgreSQL for query operations in KB.
        :param work_mem_kb: work_mem setting in KB.
        :param executor: ThreadPoolExecutor to manage concurrency.
        :param max_retries: Maximum number of retries for each query.
        :param base_wait_time: Base wait time for retries in seconds.
        """
        # Assign initial priority based on execution time and peak memory
        # Higher execution time and higher peak memory get higher priority
        # For heapq, lower priority value means higher priority, so invert the priority
        self.model = model
        self.statistics = statistics
        self.mem_scale = self.statistics['peakmem']['scale']
        self.mem_center = self.statistics['peakmem']['center']
        self.engine = engine
        self.total_memory_kb = total_memory_kb
        self.work_mem_kb = work_mem_kb
        self.shared_buffers_kb = shared_buffers_kb
        self.results = {}
        self.lock = threading.Lock()
        self.executor = executor
        self.max_retries = max_retries
        self.base_wait_time = base_wait_time
        self.available_memory = total_memory_kb
        self.queries = queries
        self.exp = exp
        self.exp_num = exp_num
        self.device = device
        
        # Initialize the priority queue
        self.ready_queue = PriorityQueue()
        self.success_count = 0
        
        for query in queries:
            peak_memory = self.predict_peak_memory(query.explain_json_plan)
            if peak_memory > self.total_memory_kb:
                logging.warning(
                    f"Memory-Based Strategy: Query {query.id} requires more memory "
                    f"({peak_memory} KB) than available ({self.total_memory_kb} KB). Skipping."
                )
                self.results[query.id] = {
                    'execution_time': float('inf'),
                    'total_time': float('inf'),
                    'success': False,
                    'error_message': 'Exceeds memory limit.'
                }
                continue
            
            # Calculate initial priority (example: higher time + peakmem)
            execution_time = query.explain_json_plan.get('time', 0)  # Default to 0 if not present
            peakmem = peak_memory
            # Define weights for time and memory; adjust as needed
            alpha = 1.0  # Weight for execution time
            beta = 0.5   # Weight for peak memory
            # priority_value = -alpha * execution_time * 1e2   # + beta * peakmem 
            priority_value =  - beta * peakmem
            # For heapq, lower priority value has higher priority 
            prioritized_query = PrioritizedQuery(
                priority=priority_value,
                query=query,
                enqueue_time=time.time()
            )
            self.ready_queue.push(prioritized_query)

        # Condition variable to synchronize scheduler and executor
        self.condition = threading.Condition()
        self.active_queries = 0
        self.finished = False

    def execute(self) -> float:
        """
        Executes queries based on available memory and prioritizes longer, more memory-intensive queries.
        Relies on execute_query to handle retries.

        :return: Total execution time in seconds.
        """
        start_time = time.time()
        
        # Start the scheduler thread
        scheduler = threading.Thread(target=self.scheduler_thread, daemon=True)
        scheduler.start()

        # Wait for the scheduler to finish processing all queries
        with self.condition:
            while not self.finished:
                self.condition.wait()

        end_time = time.time()
        total_exec_time = end_time - start_time
        logging.info(f"Memory-Based Strategy Total Execution Time: {total_exec_time:.2f} seconds.")
        return total_exec_time
    
    def scheduler_thread(self):
        """
        Scheduler thread that continuously monitors the priority queue and submits queries
        to the executor when sufficient memory is available.
        """
        while True:
            with self.condition:
                while self.ready_queue.is_empty() and self.active_queries > 0:
                    self.condition.wait()

                if self.ready_queue.is_empty() and self.active_queries == 0:
                    self.finished = True
                    self.condition.notify_all()
                    logging.debug(f"Scheduler: All queries completed. Notified all.")
                    break

                current_time = time.time()
                ready_queries = self.ready_queue.pop_ready_queries(current_time)

            if ready_queries: 
                for prioritized_query in ready_queries:
                    # Submit the query to the executor
                    with self.condition:
                        self.active_queries += 1
                    self.wait_for_available_memory(prioritized_query, self.total_memory_kb)
                    
                    future = self.executor.submit(
                        execute_query,
                        self.executor,
                        self.engine,
                        self.ready_queue,
                        prioritized_query,
                        self.results,
                        self.lock,
                        'memory_based',
                        self.max_retries,
                        self.base_wait_time,
                        self.total_memory_kb,
                        self.exp,
                        self.exp_num
                    )
                    # Attach a callback to handle query completion
                    future.add_done_callback(self.query_complete_callback)
                    # release_lock
            else: 
                with self.condition:
                    # No ready queries, determine the next wait time
                    next_time = self.ready_queue.peek_next_available_time()
                    if next_time:
                        wait_time = max(next_time - current_time, 0)
                        self.condition.wait(timeout=wait_time)
                    else:
                        # No queries left
                        if self.active_queries == 0:
                            self.finished = True
                            self.condition.notify_all()
                            break


    def query_complete_callback(self, future: concurrent.futures.Future):
        """
        Callback function that is called when a query execution is complete.
        It handles the results and re-enqueues the query if necessary.

        :param future: The Future object representing the executed query.
        """
        try:
            prioritized_query, success, error_message = future.result()
            query_id = prioritized_query.query.id
            if not success:
                
                if prioritized_query.retry_count < self.max_retries:
                    # Re-enqueue the query with higher priority
                    base_wait_time = 1.1
                    wait_time = min(base_wait_time ** prioritized_query.retry_count, 2)
                    prioritized_query.next_available_time = time.time() + wait_time
                    # logging.debug(f"Scheduler: Query {query_id} failed with error {error_message}. Re-enqueuing with higher priority and sleep {wait_time} seconds...")
                    logging.debug(f"Scheduler: Query {query_id} failed with error {error_message}. Re-enqueuing with higher priority... success_count: {self.success_count}")
                    # time.sleep(wait_time)
                    self.ready_queue.push(prioritized_query)
                else:
                    # Update result_dict with failure and total time
                    with self.lock:
                        self.results[query_id] = {
                            'execution_time': float('inf'),
                            'success': False,
                            'error_message': error_message
                        }
                    logging.error(f"Scheduler: Query {query_id} failed after {self.max_retries} retries.")
            else: # success
                self.success_count += 1
                pass

        except Exception as e:
            logging.error(f"Scheduler: Error in query execution callback: {e}")
        finally:
            with self.condition:
                self.active_queries -= 1
                # actual_active_queries = self.get_actual_active_queries()
                # logging.debug(f"Scheduler: Query {query_id} success. Currently active queries: {self.active_queries}, self.ready_queue.is_empty(): {self.ready_queue.is_empty()}. Actual active queries: {actual_active_queries}.")
                if success:
                    logging.debug(f"Scheduler: Query {query_id} success with retry {prioritized_query.retry_count}. Currently active queries: {self.active_queries}, self.ready_queue.is_empty(): {self.ready_queue.is_empty()}. success_count: {self.success_count}.")
                else:
                    logging.debug(f"Scheduler: Query {query_id} failed with retry {prioritized_query.retry_count}. Currently active queries: {self.active_queries}, self.ready_queue.is_empty(): {self.ready_queue.is_empty()}. success_count: {self.success_count}.")
                
                if self.active_queries ==0 and self.ready_queue.is_empty():
                    self.finished=True
                    logging.debug(f"Scheduler: All queries completed. Notified all (from complete_callback).")
                    self.condition.notify_all()
                else:
                    self.condition.notify()
    
    def get_actual_active_queries(self):
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM pg_stat_activity WHERE state = 'active';"))
            return result.scalar()
        
    def wait_for_available_memory(self, prioritized_query: PrioritizedQuery, total_memory_kb: float):
        base_wait_time = 2
        while True:
            # logging.info(f"Query {prioritized_query.query.id}: getting postgres memory usage")
            current_memory_usage = get_postgres_memory_usage(self.shared_buffers_kb)
            available_memory = total_memory_kb - current_memory_usage
            available_memory = max(available_memory, 0)
            logging.debug(f"Query {prioritized_query.query.id}: Total memory: {total_memory_kb}, Current memory usage: {current_memory_usage}KB, Available memory: {available_memory}KB, peak memory: {prioritized_query.query.explain_json_plan['pred_peakmem']}KB")
            if prioritized_query.query.explain_json_plan['peakmem'] <= available_memory:
                return
            else:
                logging.debug(f"Query {prioritized_query.query.id} is waiting for available memory with retry {prioritized_query.retry_count}.  Currently active queries: {self.active_queries}, self.ready_queue.is_empty(): {self.ready_queue.is_empty()}. success_count: {self.success_count}.")
                wait_time = min(base_wait_time ** prioritized_query.retry_count, 32)
                time.sleep(wait_time)  # Wait for available memory to increase


    # ----------------------------
    # Function to Predict Peak Memory
    # ----------------------------
    def predict_peak_memory(self, explain_json_plan: Dict[str, Any]) -> int:
        """
        Predicts the peak memory usage of a query based on its explain plan.
        Replace this mock function with actual logic based on your explain plans.

        :param explain_json_plan: Dictionary containing the explain plan of the query.
        :return: Estimated peak memory usage in KB.
        """
        # Example: Extract 'peakmem' from the explain plan if available
        nodes = []
        edges = []
        parse_plan(explain_json_plan, self.statistics, nodes=nodes, edges=edges)

        # Convert lists to tensors
        x = torch.tensor(nodes, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        data = Data(x=x, edge_index=edge_index)
        # move data to device
        data = data.to(self.device)
        pred_mem, _ = self.model(data)
        pred_mem = pred_mem * self.mem_scale + self.mem_center
        explain_json_plan['pred_peakmem'] = pred_mem
        return pred_mem

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
            explain_json_plan=plan  # Directly assign the plan dict
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
            pool_recycle=1800      # Recycle connections after 30 minutes
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
    admin_connection_buffer = 10
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

    postgres_background_memory_kb = get_postgres_background_memory_usage()
    available_memory_kb += postgres_background_memory_kb
    

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
    plan_file = f'/home/wuy/DB/pg_mem_data/{args.dataset}/val_plans.json'
    queries = load_queries(plan_file, total_query_memory_limit_kb)
    queries = queries[:args.num_queries]  # Limit to 100 queries for testing
    if not queries:
        logging.error("No queries to execute. Exiting.")
        engine.dispose()
        return

    # ----------------------------
    # Initialize ThreadPoolExecutor with max_workers equal to adjusted_max_connections
    # ----------------------------
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=adjusted_max_connections)
    logging.info(f"Initialized ThreadPoolExecutor with {adjusted_max_connections} workers.")

    max_retries = 100


    

    # ----------------------------
    # Execute Naive Strategy Multiple Times
    # ----------------------------
    if not args.no_naive:
        naive_total_time_list = []
        naive_waiting_sum_list = []
        for i in range(args.exp_num):
            logging.info(f"\nExecuting Naive Strategy - Run {i+1}/{args.exp_num}:")
            # Initialize a new NaiveStrategy instance for each run
            naive_strategy = NaiveStrategy(
                engine=engine,
                queries=queries,
                executor=executor,
                max_retries=max_retries,  # Set as needed
                base_wait_time=1.1,
                exp = i,
                exp_num = args.exp_num
            )
            naive_total_time = naive_strategy.execute()
            naive_total_time_list.append(naive_total_time)

            # Log memory spill after strategy execution
            log_memory_spill(engine, 'naive')

            # Calculate sum of waiting times
            naive_waiting_sum = sum(
                info['total_time'] for info in naive_strategy.results.values() if 'total_time' in info
            )
            naive_waiting_sum_list.append(naive_waiting_sum)
        
    with open('/home/wuy/DB/pg_mem_data/combined_statistics_workload.json') as f:
        statistics = json.load(f)

    model = GIN(hidden_channels=32, out_channels=1, num_layers=6, num_node_features=21, dropout=0.5)
    logging.info(f"Loading checkpoint")
    model.load_state_dict(torch.load('GIN_carcinogenesis_credit_employee_financial_geneea_tpcds_sf1_mem__best.pth'))
    model = model.to(args.device)
    model.eval()
    logging.info(f"Model loaded")


    
    # ----------------------------
    # Execute Memory-Based Strategy Multiple Times
    # ----------------------------
    memory_based_total_time_list = []
    memory_based_waiting_sum_list = []
    for i in range(args.exp_num):
        logging.info(f"\nExecuting Memory-Based Strategy - Run {i+1}/{args.exp_num}:")
        # Initialize a new MemoryBasedStrategy instance for each run
        memory_based_strategy = MemoryBasedStrategy(
            model,
            statistics,
            engine=engine,
            queries=queries,
            total_memory_kb=total_query_memory_limit_kb,
            work_mem_kb=work_mem_kb,
            shared_buffers_kb=shared_buffers_kb,
            executor=executor,
            max_retries=max_retries,  # Enable retries similar to Naive Strategy
            base_wait_time=1.1,  # Set as needed
            exp = i,
            exp_num = args.exp_num,
            device = args.device
        )
        try:
            memory_based_total_time = memory_based_strategy.execute()
        except Exception as e:
            logging.error(f"Memory-Based Strategy failed: {e}")
            memory_based_total_time = float('inf')
        memory_based_total_time_list.append(memory_based_total_time)

        # Log memory spill after strategy execution
        log_memory_spill(engine, 'memory-based')

        # Calculate sum of waiting times
        memory_based_waiting_sum = sum(
            info['total_time'] for info in memory_based_strategy.results.values() if 'total_time' in info
        )
        memory_based_waiting_sum_list.append(memory_based_waiting_sum)


    # ----------------------------
    # Compare Performance
    # ----------------------------
    logging.info("\nComparison of Strategies:")
    mean_naive_time = mean(naive_total_time_list) if not args.no_naive else 0
    mean_memory_based_time = mean(memory_based_total_time_list)
    mean_naive_waiting = mean(naive_waiting_sum_list) if not args.no_naive else 0
    mean_memory_based_waiting = mean(memory_based_waiting_sum_list)

    if not args.no_naive:
        logging.info(f"Naive Strategy Average Total Execution Time: {mean_naive_time:.2f} seconds.")
        logging.info(f"Naive Strategy Average Sum of Waiting Times: {mean_naive_waiting:.2f} seconds.")
    
    logging.info(f"Memory-Based Strategy Average Total Execution Time: {mean_memory_based_time:.2f} seconds.")
    logging.info(f"Memory-Based Strategy Average Sum of Waiting Times: {mean_memory_based_waiting:.2f} seconds.")

    if not args.no_naive:
        if mean_memory_based_time < mean_naive_time:
            logging.info("Memory-Based Strategy is faster on average.")
        elif mean_memory_based_time > mean_naive_time:
            logging.info("Naive Strategy is faster on average.")
        else:
            logging.info("Both strategies have the same average execution time.")
        
        if mean_memory_based_waiting < mean_naive_waiting:
            logging.info("Memory-Based Strategy has a lower average sum of waiting times.")
        elif mean_memory_based_waiting > mean_naive_waiting:
            logging.info("Naive Strategy has a lower average sum of waiting times.")
        else:
            logging.info("Both strategies have the same average sum of waiting times.")


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
