# proxy_FCFS.py.py   naive strategy implementation for timeseries workload scheduling

import threading
import time
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
from utils import Query, PrioritizedQuery, PriorityQueue, QueryRequest
from fastapi.responses import JSONResponse

# ----------------------------
# Setup Logging
# ----------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
logging.getLogger('uvicorn.access').setLevel(logging.WARNING)

# ---------
# ----------------------------
# Helper Functions
# ----------------------------

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

def get_postgres_memory_usage(shared_buffers_kb: int) -> int:
    """
    Retrieves the current memory usage of PostgreSQL.

    :param shared_buffers_kb: Shared buffers setting in KB.
    :return: Current memory usage in KB.
    """
    # Implement actual memory usage retrieval, e.g., via system metrics or PostgreSQL statistics
    # For demonstration, returning a mock value
    return 0  # Replace with actual implementation

# ----------------------------
# Naive Strategy Implementation
# ----------------------------
class NaiveStrategy:
    def __init__(
        self,
        engine: Engine,
        executor: concurrent.futures.ThreadPoolExecutor,
        max_retries: int = 5,
        base_wait_time: float = 2.0,
        exp: int = 0,
        exp_num: int = 0
    ):
        """
        Initializes the NaiveStrategy.

        :param engine: The SQLAlchemy Engine instance.
        :param executor: ThreadPoolExecutor to manage concurrency.
        :param max_retries: Maximum number of retries for each query.
        :param base_wait_time: Base wait time for retries in seconds.
        :param exp: Experiment identifier (if any).
        :param exp_num: Experiment number (if any).
        """
        self.engine = engine
        self.results = {}
        self.lock = threading.Lock()
        self.executor = executor
        self.max_retries = max_retries
        self.base_wait_time = base_wait_time
        self.exp = exp
        self.exp_num = exp_num

        self.success_count = 0

    def execute(self, query: Query):
        """
        Executes a single query concurrently without considering memory constraints.
        Relies on naive_execute_query to handle retries.

        :param query: Query object to execute.
        """
        start_time = time.time()
        # Submit the query for execution
        future = self.executor.submit(self.naive_execute_query, query, start_time)
        future.add_done_callback(lambda f: self.handle_execution_result(f, query.id))
        logging.debug(f"NaiveStrategy: Submitted Query {query.id} for execution.")

    def naive_execute_query(self, query: Query, start_time: float):
        """
        Executes a single query in naive strategy and records execution and waiting times.
        Handles retries.

        :param query: Query object to execute.
        :param start_time: Time when the query was submitted.
        :return: Tuple containing (query_id, success, error_message)
        """
        query_id = query.id
        retry_count = 0

        while retry_count < self.max_retries:
            try:
                exec_start = time.time()
                with self.engine.connect() as conn:
                    logging.debug(f"NaiveStrategy: Executing Query {query_id}, attempt {retry_count + 1}")
                    conn.execute(text(query.sql))
                exec_end = time.time()
                exec_time = exec_end - exec_start
                total_time = exec_end - start_time
                self.success_count += 1
                with self.lock:
                    self.results[query_id] = {
                        'execution_time': exec_time,
                        'total_time': total_time,
                        'success': True,
                        'error_message': None
                    }

                logging.info(f"NaiveStrategy: Query {query_id} executed successfully in {exec_time:.2f} seconds. Total time: {total_time:.2f} seconds. success_count: {self.success_count}")
                return (query_id, True, None)

            except Exception as e:
                retry_count += 1
                wait_time = min(self.base_wait_time ** retry_count, 32)
                logging.warning(f"NaiveStrategy: Query {query_id} failed on attempt {retry_count} with error: {e}. Retrying after {wait_time} seconds. success_count: {self.success_count}")
                time.sleep(wait_time)

        # If all retries failed
        total_time = time.time() - start_time
        with self.lock:
            self.results[query_id] = {
                'execution_time': float('inf'),
                'total_time': total_time,
                'success': False,
                'error_message': "Max retries exceeded."
            }
        logging.error(f"NaiveStrategy: Query {query_id} failed after {self.max_retries} retries.")

        return (query_id, False, "Max retries exceeded.")

    def handle_execution_result(self, future: concurrent.futures.Future, query_id: int):
        """
        Handles the result of the query execution.

        :param future: Future object representing the execution.
        :param query_id: ID of the query.
        """
        try:
            result = future.result()
            # Result has already been handled in naive_execute_query
            logging.debug(f"NaiveStrategy: Query {query_id} execution result handled.")
        except Exception as e:
            logging.error(f"NaiveStrategy: Unexpected error while executing Query {query_id}: {e}")
            with self.lock:
                self.results[query_id] = {
                    'execution_time': float('inf'),
                    'total_time': float('inf'),
                    'success': False,
                    'error_message': str(e)
                }

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
# Initialize FastAPI App and Endpoints
# ----------------------------
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='tpcds_sf1', help='Name of the dataset (e.g., tpch, tpcds)')
parser.add_argument('--device', type=str, default='cpu', help='Device to run the workload (e.g., cpu, gpu)')
parser.add_argument('--num_queries', type=int, default=100, help='Number of queries to run')
args = parser.parse_args()

# Initialize the FastAPI app
app = FastAPI()

# Database connection parameters
connection_params = {
    'dbname': args.dataset,
    'user': 'wuy',
    'password': 'wuy',
    'host': 'localhost',
    'port': 5431
}

# Initialize SQLAlchemy Engine
DATABASE_URL = "postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}".format(**connection_params)
try:
    engine = create_engine(
        DATABASE_URL,
        pool_size=1000,          # Adjust based on max_connections
        max_overflow=0,          # No additional connections beyond pool_size
        pool_timeout=30,         # Timeout for getting connection
        pool_recycle=1800        # Recycle connections after 30 minutes
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

# Initialize ThreadPoolExecutor with max_workers equal to adjusted_max_connections
executor = concurrent.futures.ThreadPoolExecutor(max_workers=adjusted_max_connections)
logging.info(f"Initialized ThreadPoolExecutor with {adjusted_max_connections} workers.")

# Load statistics (replace with actual path and ensure the file exists)
# statistics_file = f'/home/wuy/DB/pg_mem_data/{args.dataset}/combined_statistics_workload.json'
statistics_file = f'/home/wuy/DB/pg_mem_data/combined_statistics_workload.json'

try:
    with open(statistics_file, 'r') as f:
        statistics = json.load(f)
    logging.info("Loaded statistics successfully.")
except FileNotFoundError:
    logging.error(f"Statistics file not found: {statistics_file}")
    exit()
except json.JSONDecodeError as jde:
    logging.error(f"JSON decode error in statistics file: {jde}")
    exit()

# Initialize the NaiveStrategy
naive_strategy = NaiveStrategy(
    engine=engine,
    executor=executor,
    max_retries=10000,          # Adjust as needed
    base_wait_time=2.0,         # Adjust as needed
    exp=0,
    exp_num=0
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

    # Execute the query using NaiveStrategy
    naive_strategy.execute(query)

    return {"query_id": query.id, "status": "submitted"}

@app.get("/query_status/{query_id}")
def query_status(query_id: str):
    """
    Endpoint to check the status of a submitted query.

    :param query_id: The unique ID of the query.
    :return: Status of the query execution.
    """
    try:
        query_id = int(query_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid query ID format. Must be an integer.")
    
    result = naive_strategy.results.get(query_id)
    if not result:
        # Query might still be in progress or not submitted
        return JSONResponse(status_code=202, content={"message": "Query is still being processed or has not been started yet", "status": "in progress"})
    
    return {"query_id": query_id, "result": result}

@app.post("/restart")
def restart_strategy():
    """
    Endpoint to reset the memory-based strategy state.
    Clears the results to start fresh for each new client run.
    """
    naive_strategy.results.clear()  # Reset the results dictionary
    naive_strategy.success_count = 0  # Reset the success count
    logging.info("Naive Strategy results have been reset.")
    return JSONResponse(status_code=200, content={"message": "Memory strategy results have been reset."})

# ----------------------------
# Run the FastAPI App
# ----------------------------

import uvicorn

if __name__ == "__main__":
    uvicorn.run("proxy_FCFS:app", host="0.0.0.0", port=8000, reload=True)
