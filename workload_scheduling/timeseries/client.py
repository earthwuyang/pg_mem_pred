# client.py

import requests
import time
import random
import uuid
import threading
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json

from utils import Query

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('client.log', mode='w'), logging.StreamHandler()])

PROXY_URL = "http://localhost:8000"  # Update if proxy is running elsewhere


# ----------------------------
# Function to Load Queries from JSON File
# ----------------------------
def load_queries(plan_file: str, total_query_memory_limit_kb: int):
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


import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='tpcds_sf1', help='Dataset to use (tpch, tpcds, or your own dataset)')
argparser.add_argument('--num_queries', type=int, default=100, help='Number of queries to execute')
args = argparser.parse_args()
total_query_memory_limit_kb = 3 * 1024**2  # 3GB memory limit for query operations
# Load queries from JSON file
plan_file = f'/home/wuy/DB/pg_mem_data/{args.dataset}/val_plans.json'
queries = load_queries(plan_file, total_query_memory_limit_kb)
queries = queries[:args.num_queries]  # Limit to 100 queries for testing

monitoring_threads = []

def submit_queries():
    """
    Submits queries to the proxy at random intervals.
    """
    
    for qid, q in enumerate(queries):
        sql = q.sql
        try:
            q.submit_time = time.time()
            while True:
                response = requests.post(f"{PROXY_URL}/submit_query", json={"sql": sql, "id": qid, "explain_json_plan": q.explain_json_plan})
                if response.status_code == 200:
                    break
                print(f"Failed to submit query. Retrying in 1 second.")
                time.sleep(1)
            
            if response.status_code == 200:
                data = response.json()
                query_id = data.get("query_id")
                # logging.info(f"Submitted Query ID: {query_id} | SQL: {sql}")
                logging.debug(f"Submitted Query ID: {query_id}")
                # Start a thread to monitor the query status
                monitor_thread = threading.Thread(target=monitor_query, args=(query_id,), daemon=True)
                monitor_thread.start()
                monitoring_threads.append(monitor_thread)

            else:
                logging.error(f"Failed to submit query. Status Code: {response.status_code}, Response: {response.text}")
        except Exception as e:
            logging.error(f"Exception while submitting query: {e}")
        
        
        time.sleep(random.random()/10) # max sleep 100ms
    

def monitor_query(query_id: str):
    """
    Monitors the status of a submitted query until it completes.

    :param query_id: The unique ID of the query to monitor.
    """
    while True:
        try:
            response = requests.get(f"{PROXY_URL}/query_status/{query_id}")
            if response.status_code == 200:
                data = response.json()
                result = data.get("result")
                if result['success']:
                    queries[query_id].end_time = time.time()
                    logging.info(
                        f"Query ID: {query_id} executed successfully in {result['execution_time']:.2f} seconds. Latency: {queries[query_id].end_time - queries[query_id].submit_time:.2f} seconds."
                    )
                    break
                else:
                    queries[query_id].end_time = time.time()
                    logging.warning(
                        f"Query ID: {query_id} failed. Error: {result['error_message']}"
                    )
                # break  # Exit the monitoring loop
            elif response.status_code == 404:
                # logging.warning(f"Query ID: {query_id} not found. It might still be in queue.")
                pass
            else:
                logging.debug(f"Failed to get status for Query ID: {query_id}. Status Code: {response.status_code}")
        except Exception as e:
            logging.error(f"Exception while checking status for Query ID: {query_id}: {e}")
        
        # Wait for 1 second before checking again
        time.sleep(0.01)

def reset_proxy():
    """
    Sends a request to the proxy to reset the memory strategy results.
    """
    try:
        response = requests.post(f"{PROXY_URL}/restart")
        if response.status_code == 200:
            logging.info("Proxy memory strategy results reset successfully.")
        else:
            logging.error(f"Failed to reset proxy memory strategy. Status Code: {response.status_code}")
    except Exception as e:
        logging.error(f"Exception while resetting proxy: {e}")

def stop_proxy():
    """
    Sends a request to the proxy to stop the server.
    """
    try:
        response = requests.post(f"{PROXY_URL}/stop")
        if response.status_code == 200:
            logging.info("Proxy server stopped successfully.")
        else:
            logging.error(f"Failed to stop proxy server. Status Code: {response.status_code}")
    except Exception as e:
        logging.error(f"Exception while stopping proxy: {e}")


def main():
    """
    Starts the query submission in a separate thread.
    """
    reset_proxy()
    logging.info(f"Starting client for {args.num_queries} queries.")
    
    submission_thread = threading.Thread(target=submit_queries, daemon=True)
    begin = time.time()
    submission_thread.start()
    
    # Keep the main thread alive
    try:
        # wait for the submission thread to complete
        submission_thread.join()
        for thread in monitoring_threads:
            thread.join()
        end = time.time()
        logging.info(f"Total time taken: {end-begin:.2f} seconds for {args.num_queries} queries.")
        average_latency = sum([q.end_time - q.submit_time for q in queries]) / len(queries)
        logging.info(f"Average latency: {average_latency:.2f} seconds.")
        stop_proxy()
    except KeyboardInterrupt:
        logging.info("Client shutting down.")

if __name__ == "__main__":
    random.seed(1)  # For reproducibility
    main()
