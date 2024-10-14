import os
import psycopg2
from psycopg2 import sql
import json
import multiprocessing


def connect_db(conn_params):
    try:
        conn = psycopg2.connect(**conn_params)
        return conn
    except Exception as e:
        print(f"Connection fails: {e}")
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


def import_data(conn_params, table_name, csv_file, sep):
    conn = connect_db(conn_params)
    if conn is None:
        return

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
                sep=sql.SQL(sep)
            )
            
            # Open the preprocessed CSV file and execute the COPY command
            with open(temp_csv_file, 'r') as f:
                cur.copy_expert(copy_sql, f)
                
        # Commit after successful load
        conn.commit()
        print(f"Table {table_name} of dataset {conn_params['dbname']} loaded successfully.")
    except Exception as e:
        # Rollback in case of error
        conn.rollback()
        print(f"Failed to load table {table_name}: {e}")
    finally:
        # Remove the temporary file after use
        if os.path.exists(temp_csv_file):
            os.remove(temp_csv_file)
        conn.close()


def process_table(conn_params, dataset, table, sep):
    DATA_DIR = f'/home/wuy/DB/pg_mem_data/datasets/{dataset}'
    csv_file = os.path.join(DATA_DIR, f"{table}.csv")

    if os.path.exists(csv_file):
        print(f"Loading table {table} for dataset {dataset}...")
        import_data(conn_params, table, csv_file, sep)
    else:
        print(f"File {csv_file} does not exist, skipping table {table} for dataset {dataset}.")


def import_dataset(dataset, conn_params):
    # Load schema file
    with open(os.path.join(os.path.dirname(__file__), f'../../zsce/cross_db_benchmark/datasets/{dataset}/schema.json'), 'r') as f:
        schema = json.load(f)

    database_name = schema['name']
    tables = schema['tables']
    sep = schema['csv_kwargs']['sep']

    # Create database connection parameters for the current dataset
    conn_params["dbname"] = conn_params['user']

    # Connect to PostgreSQL to create database and tables
    conn = connect_db(conn_params)
    if conn is None:
        print("Cannot connect to PostgreSQL server.")
        return

    conn.autocommit = True
    with conn.cursor() as cur:
        try:
            cur.execute(f"CREATE DATABASE {database_name};")
            print(f"Database {database_name} created successfully.")
        except Exception as e:
            print(f"Database {database_name} already exists, skipping creation.")
    
    conn.close()

    conn_params["dbname"] = database_name
    # Reconnect to the newly created database to create tables
    conn = connect_db(conn_params)
    if conn is None:
        return

    # Create tables for the current dataset
    with conn.cursor() as cur:
        sql_file_path = os.path.join(os.path.dirname(__file__), f"../../zsce/cross_db_benchmark/datasets/{dataset}/schema_sql/postgres.sql")
        with open(sql_file_path, 'r') as f:
            try:
                cur.execute(f.read())
            except Exception as e:
                print(f"Failed to execute table creation SQL for dataset {database_name}: {e}")
        conn.commit()
        print(f"Table creation SQL executed for dataset {database_name}.")

    # Use multiprocessing to process tables in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(process_table, [(conn_params, dataset, table, sep) for table in tables])

    # analyze the dataset after loading
    with conn.cursor() as cur:
        try:
            cur.execute(f"ANALYZE;")
            print(f"Dataset {database_name} analyzed successfully.")
        except Exception as e:
            print(f"Failed to analyze dataset {database_name}: {e}")

def main():
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from database_list import database_list

    # Load connection information
    with open(os.path.join(os.path.dirname(__file__), '../../conn.json')) as f:
        conn_info = json.load(f)

    conn_params = {
        "user": conn_info['user'],
        "password": conn_info['password'],
        "host": conn_info['host'],
        "port": conn_info['port']
    }

    # Process each dataset one by one and use multiprocessing for tables
    for dataset in database_list:
        import_dataset(dataset, conn_params)


if __name__ == "__main__":
    main()
