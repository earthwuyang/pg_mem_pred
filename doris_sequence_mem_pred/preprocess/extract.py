import re
import json
import pymysql
from tqdm import tqdm

def get_explain_plan(connection, sql):
    cursor = connection.cursor()
    cursor.execute(f"EXPLAIN optimized plan {sql}")
    result = cursor.fetchall()
    plan = [row[0] for row in result]
    return plan

# Connect to the database
# Connection parameters
host = '101.6.5.215'
port = 9030  # Default Doris port for MySQL protocol
user = 'root'
password = ''
database = 'tpcds'

# Establish the connection
connection = pymysql.connect(host=host, port=port, user=user, password=password, database=database)

fe_log_path = '/home/wuy/doris-master/output/fe/log'
import os
for file in os.listdir(fe_log_path):
    if file.startswith('fe.audit.log'):
        file_path = os.path.join(fe_log_path, file)
        print(f"Processing {file_path}")
        with open(file_path, 'r') as f:
            log_lines = f.readlines()

        # Define a pattern to match the required fields
        pattern = r"Time\(ms\)=(\d+).*?Stmt=(.*?)\|.*?peakMemoryBytes=(\d+)"

        data_list = []

        # Extract and print the desired information
        for line in tqdm(log_lines):
            match = re.search(pattern, line)
            if match:
                time = match.group(1)
                stmt = match.group(2).split('\\n')[0]
                peak_memory = match.group(3)
                if stmt.startswith("SELECT") and 'timeout' not in line:
                    data={}
                    data['time_ms'] = time
                    data['stmt'] = stmt
                    data['peak_memory_bytes'] = peak_memory
                    try:
                        data['plan'] = get_explain_plan(connection, stmt)
                    except:
                        print(f"Failed to get explain plan for {stmt}")
                        continue
                    data_list.append(data)
                # print(f"Time: {time} ms, Stmt: {stmt}, PeakMemoryBytes: {peak_memory} bytes")
    
# Save the extracted data to a json file
with open('data.json', 'w') as f:
    json.dump(data_list, f, indent=4)
        
