# proxy.py memory based strategy

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
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import psutil
import torch
from torch_geometric.data import Data
from utils import Query, PrioritizedQuery, PriorityQueue, QueryRequest
from parse_plan import parse_plan
from GIN import GIN

# ----------------------------
# Setup Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logging.getLogger('uvicorn.access').setLevel(logging.WARNING)

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



# ----------------------------
# MemoryBasedStrategy with Thread-Per-Query
# ----------------------------
class MemoryBasedStrategy:
    def __init__(
        self,
        model,
        statistics,
        engine: Engine,
        executor: concurrent.futures.ThreadPoolExecutor,
        container_name: str,
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
        :param container_name: Name of the PostgreSQL Docker container.
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
        self.device = device

        # Initialize Docker client once
        self.threads = []
        self.success_count = 0
        import docker
        self.client = docker.from_env()
        try:
            self.container = self.client.containers.get(container_name)
            self.container_id = self.container.id
            logging.info(f"Connected to Docker container '{container_name}' with id {self.container_id}.")
        except docker.errors.NotFound:
            logging.error(f"Docker container '{container_name}' not found.")
            raise
        except Exception as e:
            logging.error(f"Error connecting to Docker container '{container_name}': {e}")
            raise


    import os

    def get_postgres_memory_usage(self):
        """
        Gets the memory usage of the Docker container running PostgreSQL by reading the 
        memory usage from cgroup files.

        :return: Total memory usage in KB or None if an error occurs.
        """
        try:
            # Path to the memory usage file in cgroup for this Docker container
            cgroup_memory_usage_path = f"/sys/fs/cgroup/memory/docker/{self.container_id}/memory.usage_in_bytes"
            
            # Read the current memory usage in bytes
            with open(cgroup_memory_usage_path, 'r') as file:
                memory_usage_bytes = int(file.read().strip())
            
            # Convert memory usage to KB
            process_specific_memory_kb = memory_usage_bytes // 1024
            
            # Calculate the total memory usage including shared_buffers
            total_memory_kb = self.shared_buffers_kb + process_specific_memory_kb
            return total_memory_kb

        except Exception as e:
            print(f"Error reading memory usage from cgroup: {e}")
            return None

    def execute(self, query: Query):
        """
        Handles the incoming query by starting a new thread to execute it.

        :param query: Query object to be executed.
        """
        logging.info(f"Memory-Based Strategy: Received query {query.id}.")
        thread = threading.Thread(target=self.handle_query, args=(query,))
        thread.start()
        self.threads.append(thread)

    def handle_query(self, query: Query):
        """
        Handles the execution of a single query, waiting for available memory.

        :param query: Query object to be executed.
        """
        peak_memory = self.predict_peak_memory(query.explain_json_plan)
        query.pred_peakmem = peak_memory
        success=False

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

        retry_count = 0
        while retry_count <= self.max_retries:
            current_memory_kb = self.get_postgres_memory_usage()
            available_memory = self.total_memory_kb - current_memory_kb

            logging.info(
                f"Query {query.id}:Total memory: {self.total_memory_kb/1024**2} GB, Current memory: {current_memory_kb/1024**2} GB, Available memory: {available_memory} KB, "
                f"Peak memory required: {peak_memory:.2f} KB."
            )

            if peak_memory <= available_memory:

                # Execute the query
                start_time = time.time()
                success = False
                error_message = None
                try:
                    with self.engine.connect() as conn:
                        conn.execute(text(query.sql))
                    execution_time = time.time() - start_time
                    success = True
                    self.success_count += 1
                    logging.info(
                        f"Memory-Based Strategy: Query {query.id} executed successfully in {execution_time:.2f} seconds, success_count={self.success_count}"
                    )
                except Exception as e:
                    retry_count += 1
                    execution_time = time.time() - start_time
                    success = False
                    error_message = str(e)
                    # logging.error(
                    #     f"Memory-Based Strategy: Query {query.id} failed after {execution_time:.2f} seconds with error: {e}"
                    # )

                # Update results
                with self.lock:
                    self.results[query.id] = {
                        'execution_time': execution_time,
                        'total_time': execution_time,  # Simplification; can track total wait time if needed
                        'success': success,
                        'error_message': error_message
                    }

                if not success and retry_count < self.max_retries:
                    retry_count += 1
                    wait_time = min(self.base_wait_time ** retry_count, 32)
                    logging.info(
                        f"Memory-Based Strategy: Retrying query {query.id} after {wait_time} seconds (Retry {retry_count}/{self.max_retries}), success_count={self.success_count}"
                    )
                    time.sleep(wait_time)
                    continue
                break
            else:
                # Not enough memory; wait and retry
                # retry_count += 1
                wait_time = min(self.base_wait_time ** retry_count, 32)
                logging.info(
                    f"Memory-Based Strategy: Waiting for memory to execute query {query.id}. "
                    f"Retrying after {wait_time} seconds (Retry {retry_count}/{self.max_retries}), success_count={self.success_count}"
                )
                time.sleep(wait_time)

        if not success:
            with self.lock:
                self.results[query.id] = {
                    'execution_time': float('inf'),
                    'total_time': float('inf'),
                    'success': False,
                    'error_message': error_message or 'Max retries exceeded.'
                }

    def predict_peak_memory(self, explain_json_plan: Dict[str, Any]) -> int:
        """
        Predicts the peak memory usage of a query based on its explain plan.

        :param explain_json_plan: Dictionary containing the explain plan of the query.
        :return: Estimated peak memory usage in KB.
        """
        # Example: Extract 'peakmem' from the explain plan if available
        # Replace with actual prediction logic using your model
        # explain_json_plan['pred_peakmem'] = explain_json_plan.get('peakmem', 0)
        # return explain_json_plan.get('peakmem', 0)
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

total_query_memory_limit_kb = 1 * 1024**2  # 3 GB for example




# ----------------------------
# Initialize ThreadPoolExecutor with max_workers equal to adjusted_max_connections
# ----------------------------
executor = concurrent.futures.ThreadPoolExecutor(max_workers=adjusted_max_connections)
logging.info(f"Initialized ThreadPoolExecutor with {adjusted_max_connections} workers.")

max_retries = 10000
import json
# statistics_file = f'/home/wuy/DB/pg_mem_data/{args.dataset}/combined_statistics_workload.json'
statistics_file = f'/home/wuy/DB/pg_mem_data/combined_statistics_workload.json'
with open(statistics_file, 'r') as f:
    statistics = json.load(f)

# Initialize the MemoryBasedStrategy

        
with open('/home/wuy/DB/pg_mem_data/combined_statistics_workload.json') as f:
    statistics = json.load(f)

model = GIN(hidden_channels=32, out_channels=1, num_layers=6, num_node_features=23, dropout=0.5)
logging.info(f"Loading checkpoint")
# model.load_state_dict(torch.load('../GIN_carcinogenesis_credit_employee_financial_geneea_tpcds_sf1_mem__best.pth'))
model.load_state_dict(torch.load('../GIN_airline_credit_carcinogenesis_employee_hepatitis_mem__best.pth'))
model = model.to(args.device)
model.eval()
logging.info(f"Model loaded")

memory_strategy = MemoryBasedStrategy(
    model=model,
    statistics=statistics,
    engine=engine,
    executor=executor,
    container_name='my_postgres',
    total_memory_kb=total_query_memory_limit_kb,  # 10 MB for example
    work_mem_kb=4 * 1024,       # 4 MB
    shared_buffers_kb=2 * 1024,  # 2 MB
    max_retries=max_retries,
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
        # raise HTTPException(status_code=404, detail="Query ID not found.")
        return JSONResponse(status_code=202, content={"message": "Query is still being processed or has not been started yet", "status": "in progress"})
    # else:
    #     logging.info(f"######################## Query {query_id} status: {result['success']} #######################")
    return { "query_id": query_id, "result": result }

@app.post("/restart")
def restart_strategy():
    """
    Endpoint to reset the memory-based strategy state.
    Clears the results to start fresh for each new client run.
    """
    memory_strategy.results.clear()  # Reset the results dictionary
    memory_strategy.threads.clear()  # Reset the threads list
    memory_strategy.success_count = 0  # Reset the success count
    
    logging.info("Memory-Based Strategy results have been reset.")
    return JSONResponse(status_code=200, content={"message": "Memory strategy results have been reset."})

# ----------------------------
# Run the FastAPI App
# ----------------------------

# To run the app, use the command:
# uvicorn proxy:app --host 0.0.0.0 --port 8000

import uvicorn

if __name__ == "__main__":
    uvicorn.run("proxy:app", host="0.0.0.0", port=8000, reload=True)