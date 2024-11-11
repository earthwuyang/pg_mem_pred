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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

parser = argparse.ArgumentParser(description='Run queries sequentially.')
parser.add_argument('--begin', type=int, default=0, help='Start query id (inclusive).')
parser.add_argument('--num_queries', type=int, default=100, help='Number of queries to execute.')
parser.add_argument('--dataset', type=str, default='tpcds_sf1', help='Name of the dataset to use.')
args = parser.parse_args()

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
        pool_size=100,          # Adjust based on max_connections
        max_overflow=0,        # No additional connections beyond pool_size
        pool_timeout=30,       # Timeout for getting connection
        pool_recycle=1800      # Recycle connections after 30 minutes
    )
except Exception as e:
    logging.error(f"Failed to create SQLAlchemy engine: {e}")
    sys.exit(1)
import psutil



available_memory_kb = psutil.virtual_memory().available // 1024

postgres_background_memory_kb = get_postgres_background_memory_usage()
available_memory_kb += postgres_background_memory_kb


total_query_memory_limit_kb = available_memory_kb

logging.info(f"Adjusted total memory limit for query operations: {total_query_memory_limit_kb} KB")


# Load queries from JSON file
plan_file = f'/home/wuy/DB/pg_mem_data/{args.dataset}/val_plans.json'
queries = load_queries(plan_file, total_query_memory_limit_kb)
queries = queries[args.begin:args.num_queries]  # Limit to 100 queries for testing
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