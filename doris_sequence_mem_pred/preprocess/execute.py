import pymysql
from tqdm import tqdm

# Connection parameters
host = '101.6.5.215'
port = 9030  # Default Doris port for MySQL protocol
user = 'root'
password = ''
database = 'tpcds'

# Establish the connection
connection = pymysql.connect(host=host, port=port, user=user, password=password, database=database)

workload_file = 'workloads/workload_100k_s1_group_order_by_complex.sql'
# Read the workload file
with open(workload_file, 'r') as f:
    workload = f.readlines()

# set timeout for doris
with connection.cursor() as cursor:
    cursor.execute('set query_timeout=30') # set timeout to 1s

count = 0
# Execute the workload
for query in tqdm(workload):
    if count == 50000:
        break
    if query.strip():
        try:
            with connection.cursor() as cursor:
                query=query.replace('"', '')
                # print(f'Executing query: {query}')
                cursor.execute(query)

                cursor.fetchall()
                count += 1
        except Exception as e:
            print(f'Error executing query: {query}')
            print(e)
           
# Close the connection
connection.close()

