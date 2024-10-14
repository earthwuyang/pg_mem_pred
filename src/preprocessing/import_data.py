import os
import psycopg2
from psycopg2 import sql
import json




def connect_db():
    try:
        conn = psycopg2.connect(
            **conn_params
        )
        return conn
    except Exception as e:
        print(f"conn fails: {e}")
        return None

def preprocess_csv(csv_file):
    # Create a temporary file to write the processed data
    temp_csv_file = csv_file + ".tmp"
    
    with open(csv_file, 'r') as infile, open(temp_csv_file, 'w') as outfile:
        # Read each line in the original file
        for line in infile:
            # Replace 'None' with an empty string
            processed_line = line.replace('None', '')
            # Write the processed line to the temporary file
            outfile.write(processed_line)
    
    return temp_csv_file

def import_data(conn, table_name, csv_file):
    try:
        # Preprocess the CSV file to handle 'None' values
        temp_csv_file = preprocess_csv(csv_file)
        
        with conn.cursor() as cur:
            # Correct SQL COPY command construction
            copy_sql = sql.SQL("""
                COPY {table} FROM STDIN WITH CSV HEADER DELIMITER '{sep}' NULL AS '';
                """
            ).format(
                table=sql.Identifier(table_name),
                sep=sql.SQL(sep)  # Passing sep directly without extra quotes
            )
            
            # Open the preprocessed CSV file and execute the COPY command
            with open(temp_csv_file, 'r') as f:
                cur.copy_expert(copy_sql, f)
                
        # Commit after successful load
        conn.commit()
        print(f"load table: {table_name}")
    except Exception as e:
        # Rollback in case of error
        conn.rollback()
        print(f"load table {table_name} fails: {e}")

    finally:
        # Remove the temporary file after use
        if os.path.exists(temp_csv_file):
            os.remove(temp_csv_file)

def main(dataset):
    # Load schema file
    with open(os.path.join(os.path.dirname(__file__), f'../../zsce/cross_db_benchmark/datasets/{dataset}/schema.json'), 'r') as f:
        schema = json.load(f)

    database_name = schema['name']
    tables = schema['tables']
    sep = schema['csv_kwargs']['sep']

    dataset = database_name

    with open(os.path.join(os.path.dirname(__file__), '../../../../../conn.json')) as f:
        conn_info = json.load(f)
    conn_params = {
        "dbname": conn_info['user'],
        "user": conn_info['user'],
        "password": conn_info['password'],
        "host": conn_info['host'],
        "port": conn_info['port']
        }


    conn = connect_db()
    if conn is None:
        print("cannot connect to database")
        return

    conn.autocommit = True
    with conn.cursor() as cur:
        try:
            cur.execute(f"CREATE DATABASE {dataset};")
        except Exception as e:
            print(f"database {dataset} already exists, skip creating.")

    conn_params = {
        "dbname": dataset,
        "user": conn_info['user'],
        "password": conn_info['password'],
        "host": conn_info['host'],
        "port": conn_info['port']
        }

    DATA_DIR = f'/home/wuy/DB/pg_mem_data/datasets/{dataset}'

    conn = connect_db()
    conn.autocommit = True
    if conn is None:
        print("cannot connect to database")
        return

    # create database if not exists
    with conn.cursor() as cur:
        # create tables
        sql_file_path = os.path.join(os.path.dirname(__file__), f"../schema_sql/postgres.sql")
        with open(sql_file_path, 'r') as f:
            cur.execute(f.read())
        conn.commit()
        print(f"table creating sqls executed of dataset {dataset}.")


    for table in tables:
        csv_file = os.path.join(DATA_DIR, f"{table}.csv")

        if os.path.exists(csv_file):
            print(f"loading: {table}")
            import_data(conn, table, csv_file)
        else:
            print(f"file {csv_file} doesnot exist, skip loading {table} for dataset {dataset}。")

    conn.close()

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from database_list import database_list

    for dataset in database_list:
        main(dataset)
