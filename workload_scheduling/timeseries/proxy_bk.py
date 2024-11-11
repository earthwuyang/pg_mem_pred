# proxy.py

import threading
import time
import uuid
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import concurrent.futures
import heapq
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

import torch
from torch_geometric.data import Data
from utils import Query, PrioritizedQuery, PriorityQueue, QueryRequest
from parse_plan import parse_plan
from GIN import GIN

# ----------------------------
# Setup Logging
# ----------------------------
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')



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




class MemoryBasedStrategy:
    def __init__(
        self,
        model,
        statistics,
        engine: Engine,
        executor: concurrent.futures.ThreadPoolExecutor,
        total_memory_kb: int,
        work_mem_kb: int,
        shared_buffers_kb: int,
        max_retries: int = 5,
        base_wait_time: float = 2.0,
        exp: int = 0,
        exp_num: int = 0,
        device: str = 'cpu'
    ):
        """
        Initializes the MemoryBasedStrategy.

        :param model: PyTorch model to predict peak memory.
        :param statistics: Statistics for memory scaling.
        :param engine: SQLAlchemy Engine instance.
        :param executor: ThreadPoolExecutor for concurrency.
        :param total_memory_kb: Total memory allocated to PostgreSQL in KB.
        :param work_mem_kb: work_mem setting in KB.
        :param shared_buffers_kb: shared_buffers setting in KB.
        :param max_retries: Maximum number of retries for each query.
        :param base_wait_time: Base wait time for retries in seconds.
        :param device: Device for PyTorch model ('cpu' or 'cuda').
        """
        self.model = model
        self.statistics = statistics
        self.mem_scale = statistics.get('peakmem', {}).get('scale', 1.0)
        self.mem_center = statistics.get('peakmem', {}).get('center', 0.0)
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
        self.exp = exp
        self.exp_num = exp_num
        self.device = device
        
        # Initialize the priority queue
        self.ready_queue = PriorityQueue()
        self.success_count = 0
        
        # Condition variable to synchronize scheduler and executor
        self.condition = threading.Condition()
        self.active_queries = 0
        self.finished = False

        # Start the scheduler thread
        scheduler = threading.Thread(target=self.scheduler_thread, daemon=True)
        scheduler.start()

    def wait_for_scheduler(self):
        """
        Blocks until all queries have been processed and the scheduler thread has completed.
        """
        with self.condition:
            while not self.finished:
                self.condition.wait()

    def execute(self, query: Query):
        """
        Adds a query to the priority queue after predicting its peak memory.

        :param query: Query object to be executed.
        """
        logging.debug(f"Memory-Based Strategy: Received query {query.id}.")
        peak_memory = self.predict_peak_memory(query.explain_json_plan)
        query.pred_peakmem = peak_memory
        if peak_memory > self.total_memory_kb:
            logging.warning(
                f"Memory-Based Strategy: Query {query.id} requires more memory "
                f"({peak_memory} KB) than available ({self.total_memory_kb} KB). Skipping."
            )
            with self.lock:
                self.results[query.id] = {
                    'execution_time': float('inf'),
                    'total_time': float('inf'),
                    'success': False,
                    'error_message': 'Exceeds memory limit.'
                }
            return
        
        # Calculate initial priority (example: higher peakmem has higher priority)
        priority_value = -peak_memory  # Negative because heapq is min-heap
        prioritized_query = PrioritizedQuery(
            priority=priority_value,
            query=query,
            enqueue_time=time.time(),
            next_available_time=time.time()
        )
        self.ready_queue.push(prioritized_query)
        with self.condition:
            self.condition.notify()
    
    def scheduler_thread(self):
        """
        Scheduler thread that continuously monitors the priority queue and submits queries
        to the executor when sufficient memory is available.
        """
        while True:
            logging.info(f"Scheduler: Checking for ready queries.")
            with self.condition:
                while self.ready_queue.is_empty():
                    self.condition.wait(timeout=0.001)
                    # logging.debug(f"Scheduler: No ready queries. Waiting...")

                current_time = time.time()
                ready_queries = self.ready_queue.pop_ready_queries(current_time)

            if ready_queries:
                for prioritized_query in ready_queries:
                    with self.condition:
                        self.active_queries += 1
                    self.wait_for_available_memory(prioritized_query)
                    
                    future = self.executor.submit(
                        self.execute_query,
                        self.engine,
                        prioritized_query,
                        self.results,
                        self.lock,
                        self.max_retries,
                        self.base_wait_time
                    )
                    future.add_done_callback(self.query_complete_callback)
            else:
                with self.condition:
                    next_time = self.ready_queue.peek_next_available_time()
                    if next_time:
                        wait_time = max(next_time - current_time, 0)
                        self.condition.wait(timeout=wait_time)
                    

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
                    # Re-enqueue the query with higher priority and increased wait time
                    wait_time = min(self.base_wait_time ** prioritized_query.retry_count, 32)
                    prioritized_query.next_available_time = time.time() + wait_time
                    prioritized_query.retry_count += 1
                    logging.debug(
                        f"Scheduler: Query {query_id} failed with error '{error_message}'. "
                        f"Re-enqueuing with retry count {prioritized_query.retry_count}."
                    )
                    self.ready_queue.push(prioritized_query)
                else:
                    # Update results with failure
                    with self.lock:
                        self.results[query_id] = {
                            'execution_time': float('inf'),
                            'success': False,
                            'error_message': error_message
                        }
                    logging.error(
                        f"Scheduler: Query {query_id} failed after {self.max_retries} retries."
                    )
            else:
                self.success_count += 1
                logging.info(
                    f"Scheduler: Query {query_id} executed successfully in {self.results[query_id]['execution_time']} seconds. "
                    f"Total successful queries: {self.success_count}."
                )

        except Exception as e:
            logging.error(f"Scheduler: Error in query execution callback: {e}")
        finally:
            with self.condition:
                self.active_queries -= 1
                if self.active_queries == 0 and self.ready_queue.is_empty():
                    self.finished = True
                    self.condition.notify_all()
    
    def wait_for_available_memory(self, prioritized_query: PrioritizedQuery):
        """
        Waits until enough memory is available to execute the query.

        :param prioritized_query: The query to wait for.
        """
        base_wait_time = 2
        while True:
            current_memory_usage = get_postgres_memory_usage(self.shared_buffers_kb)
            available_memory = self.total_memory_kb - current_memory_usage
            available_memory = max(available_memory, 0)
            peakmem = prioritized_query.query.pred_peakmem
            logging.debug(
                f"Query {prioritized_query.query.id}: Total memory: {self.total_memory_kb} KB, "
                f"Current memory usage: {current_memory_usage} KB, "
                f"Available memory: {available_memory} KB, Peak memory required: {peakmem} KB."
            )
            if peakmem <= available_memory:
                logging.debug(
                    f"Query {prioritized_query.query.id}: Sufficient memory available. Proceeding to execute."
                )
                return
            else:
                wait_time = min(base_wait_time ** prioritized_query.retry_count, 32)
                logging.debug(
                    f"Query {prioritized_query.query.id}: Waiting for memory. Sleeping for {wait_time} seconds."
                )
                time.sleep(wait_time)

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
        # return explain_json_plan.get('peakmem', 0)
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
        pred_mem = pred_mem.item() * self.mem_scale + self.mem_center
        explain_json_plan['pred_peakmem'] = pred_mem
        return pred_mem

    
    def execute_query(
        self,
        engine: Engine,
        prioritized_query: PrioritizedQuery,
        results: Dict[str, Any],
        lock: threading.Lock,
        max_retries: int,
        base_wait_time: float
    ):
        """
        Executes the SQL query against PostgreSQL.

        :param engine: SQLAlchemy Engine instance.
        :param prioritized_query: The prioritized query to execute.
        :param results: Shared dictionary to store results.
        :param lock: Lock for thread-safe operations.
        :param max_retries: Maximum number of retries.
        :param base_wait_time: Base wait time for retries.
        :return: Tuple containing the prioritized query, success status, and error message.
        """
        query = prioritized_query.query
        logging.debug(f"Executing query {query.id}")
        start_time = time.time()
        try:
            with engine.connect() as conn:
                conn.execute(text(query.sql))
            execution_time = time.time() - start_time
            with lock:
                self.results[query.id] = {
                    'execution_time': execution_time,
                    'total_time': execution_time,
                    'success': True,
                    'error_message': None
                }
            return prioritized_query, True, None
        except Exception as e:
            execution_time = time.time() - start_time
            with lock:
                self.results[query.id] = {
                    'execution_time': execution_time,
                    'total_time': execution_time,
                    'success': False,
                    'error_message': str(e)
                }
            return prioritized_query, False, str(e)

# ----------------------------
# Define Helper Functions
# ----------------------------
import psutil
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



# ----------------------------
# Define FastAPI App and Endpoints
# ----------------------------
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='tpcds_sf1', help='Name of the dataset (e.g., tpch, tpcds)')
parser.add_argument('--device', type=str, default='cpu', help='Device to run the workload (e.g., cpu, gpu)')

parser.add_argument('--num_queries', type=int, default=100, help='Number of queries to execute')
args = parser.parse_args()

# Initialize the FastAPI app
app = FastAPI()



connection_params = {
    'dbname': args.dataset,
    'user': 'wuy',
    'password': 'wuy',
    'host': 'localhost',
    'port': 5431
}

# Initialize SQLAlchemy Engine
DATABASE_URL = "postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}".format(**connection_params),
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
    exit()

# Retrieve PostgreSQL memory settings
memory_settings = get_postgres_memory_settings(engine)
if not memory_settings:
    logging.error("Failed to retrieve memory settings. Exiting.")
    exit()

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
    exit()

logging.info(f"Adjusted max_connections for connection pool: {adjusted_max_connections}")

total_query_memory_limit_kb = 3 * 1024**2  # 3 GB for example




# ----------------------------
# Initialize ThreadPoolExecutor with max_workers equal to adjusted_max_connections
# ----------------------------
executor = concurrent.futures.ThreadPoolExecutor(max_workers=adjusted_max_connections)
logging.info(f"Initialized ThreadPoolExecutor with {adjusted_max_connections} workers.")

max_retries = 10000
import json
statistics_file = f'/home/wuy/DB/pg_mem_data/{args.dataset}/combined_statistics_workload.json'
statistics_file = f'/home/wuy/DB/pg_mem_data/airline/statistics_workload_combined.json'
with open(statistics_file, 'r') as f:
    statistics = json.load(f)

# Initialize the MemoryBasedStrategy

        
with open('/home/wuy/DB/pg_mem_data/combined_statistics_workload.json') as f:
    statistics = json.load(f)

model = GIN(hidden_channels=32, out_channels=1, num_layers=6, num_node_features=21, dropout=0.5)
logging.info(f"Loading checkpoint")
model.load_state_dict(torch.load('../GIN_carcinogenesis_credit_employee_financial_geneea_tpcds_sf1_mem__best.pth'))
model = model.to(args.device)
model.eval()
logging.info(f"Model loaded")

memory_strategy = MemoryBasedStrategy(
    model=model,
    statistics=statistics,
    engine=engine,
    executor=executor,
    total_memory_kb=total_query_memory_limit_kb,  # 10 MB for example
    work_mem_kb=4 * 1024,       # 4 MB
    shared_buffers_kb=2 * 1024,  # 2 MB
    device='cpu'
)

@app.post("/submit_query")
def submit_query(query_request: QueryRequest):
    """
    Endpoint to receive SQL queries from clients.

    :param query_request: QueryRequest object containing the SQL query.
    :return: Response indicating the query has been received.
    """
    sql = query_request.sql
    id = query_request.id
    explain_json_plan = query_request.explain_json_plan
    query = Query(sql=sql, id=id, explain_json_plan=explain_json_plan)

    # Execute the query using MemoryBasedStrategy
    memory_strategy.execute(query)

    return {"query_id": query.id, "status": "submitted"}

@app.get("/query_status/{query_id}")
def query_status(query_id: str):
    """
    Endpoint to check the status of a submitted query.

    :param query_id: The unique ID of the query.
    :return: Status of the query execution.
    """
    query_id = int(query_id)
    result = memory_strategy.results.get(query_id)
    if not result:
        # logging.debug(f"Query {query_id} not found in results.")
        raise HTTPException(status_code=404, detail="Query ID not found.")
    # else:
    #     logging.info(f"######################## Query {query_id} status: {result['success']} #######################")
    return { "query_id": query_id, "result": result }

# ----------------------------
# Run the FastAPI App
# ----------------------------

# To run the app, use the command:
# uvicorn proxy:app --host 0.0.0.0 --port 8000

import uvicorn

if __name__ == "__main__":
    uvicorn.run("proxy:app", host="0.0.0.0", port=8000, reload=True)