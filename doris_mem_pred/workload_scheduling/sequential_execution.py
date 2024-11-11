from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import logging
import sys
import argparse
import json
from typing import List
import time

class Query:
    def __init__(self, id: int, sql: str, explain_json_plan: dict):
        self.id = id
        self.sql = sql
        self.explain_json_plan = explain_json_plan


def load_queries(plan_file: str, total_query_memory_limit_mb: int, exec_mme_limit: int) -> List[Query]:
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
        plan['peak_memory_mb'] =  int(plan.get('peak_memory_bytes', 0)) // 1024**2
        if plan.get('peak_memory_mb', 0) >= min(total_query_memory_limit_mb, exec_mme_limit):
            logging.warning(f"Query {idx+1} requires more memory ({plan.get('peakmem')} MB) than available ({min(total_query_memory_limit_mb, exec_mme_limit)} MB). Skipping.")
            continue

        Q = Query(
            id=idx + 1,
            sql=plan['stmt'],
            explain_json_plan=plan  # Directly assign the plan dict
        )
        queries.append(Q)
    logging.info(f"Total queries loaded: {len(queries)}")
    return queries

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# PostgreSQL connection parameters
connection_params = {
    "host": '101.6.5.215',
    "port": 9030,
    "user": 'root',
    "password": '',
    "database": 'tpcds'
}

# Create SQLAlchemy Engine with connection pool
try:
    engine = create_engine(
        "mysql+pymysql://{user}:{password}@{host}:{port}/{database}".format(**connection_params),
        pool_size=100,          # Adjust based on max_connections
        max_overflow=0,        # No additional connections beyond pool_size
        pool_timeout=30,       # Timeout for getting connection
        pool_recycle=1800      # Recycle connections after 30 minutes
    )
except Exception as e:
    logging.error(f"Failed to create SQLAlchemy engine: {e}")
    sys.exit(1)
import psutil

parser = argparse.ArgumentParser(description='Run queries sequentially.')
parser.add_argument('--num_queries', type=int, default=100, help='Number of queries to execute.')
args = parser.parse_args()

available_memory_mb = psutil.virtual_memory().available // 1024**2

postgres_background_memory_kb = 0
available_memory_mb += postgres_background_memory_kb


total_query_memory_limit_mb = available_memory_mb

logging.info(f"Adjusted total memory limit for query operations: {total_query_memory_limit_mb} KB")

exec_mem_limit = 2048 # MB

# Load queries from JSON file
plan_file = f'../data.json'
queries = load_queries(plan_file, total_query_memory_limit_mb, exec_mem_limit)
queries = queries[:args.num_queries]  # Limit to 100 queries for testing
if not queries:
    logging.error("No queries to execute. Exiting.")
    engine.dispose()
    exit(1)
from tqdm import tqdm
begin = time.time()
# Execute queries sequentially
success_count = 0
for query in tqdm(queries):
    # logging.info(f"Executing query {query.id}: {query.sql}")
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query.sql))
            # logging.info(f"Query {query.id} executed successfully.")
            success_count += 1
    except Exception as e:
        logging.error(f"Failed to execute query {query.id}: {e}")
        engine.dispose()
        continue

end = time.time()
logging.info(f"Total time taken: {end - begin} seconds, success count: {success_count} queries")
logging.info("All queries executed successfully.")
engine.dispose()