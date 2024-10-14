import os
import psycopg2
from psycopg2 import sql
import json
import multiprocessing
from functools import partial

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
    
    try:
        with open(csv_file, 'r') as infile, open(temp_csv_file, 'w') as outfile:
            # Read each line in the original file
            for line in infile:
                # Replace 'None' with an empty string
                processed_line = line.replace('None', '')
                # Write the processed line to the temporary file
                outfile.write(processed_line)
    except Exception as e:
        print(f"Error preprocessing CSV file {csv_file}: {e}")
        return None
    
    return temp_csv_file

def import_data(conn_params, table_name, csv_file, sep):
    conn = connect_db(conn_params)
    if conn is None:
        return

    try:
        # Preprocess the CSV file to handle 'None' values
        temp_csv_file = preprocess_csv(csv_file)
        if temp_csv_file is None:
            print(f"Preprocessing failed for {csv_file}, skipping import.")
            return
        
        with conn.cursor() as cur:
            # Correct SQL COPY command construction
            copy_sql = sql.SQL("""
                COPY {table} FROM STDIN WITH CSV HEADER DELIMITER '{sep}' NULL AS '';
                """
            ).format(
                table=sql.Identifier(table_name),
                sep=sql.Literal(sep)
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
        if temp_csv_file and os.path.exists(temp_csv_file):
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
    schema_path = os.path.join(os.path.dirname(__file__), f'../../zsce/cross_db_benchmark/datasets/{dataset}/schema.json')
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
    except Exception as e:
        print(f"Error loading schema for dataset {dataset}: {e}")
        return

    database_name = schema.get('name')
    tables = schema.get('tables', [])
    sep = schema.get('csv_kwargs', {}).get('sep', ',')
    sql_file_path = os.path.join(os.path.dirname(__file__), f'../../zsce/cross_db_benchmark/datasets/{dataset}/schema_sql/postgres.sql')

    if not database_name:
        print(f"No database name found in schema for dataset {dataset}, skipping.")
        return

    # Create database connection parameters for the current dataset
    base_conn_params = conn_params.copy()
    base_conn_params["dbname"] = conn_params['user']  # Connect to default 'postgres' database to create new one

    # Connect to PostgreSQL to create database
    conn = connect_db(base_conn_params)
    if conn is None:
        print("Cannot connect to PostgreSQL server.")
        return

    conn.autocommit = True
    with conn.cursor() as cur:
        try:
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))
            print(f"Database {database_name} created successfully.")
        except psycopg2.errors.DuplicateDatabase:
            print(f"Database {database_name} already exists, skipping creation.")
        except Exception as e:
            print(f"Error creating database {database_name}: {e}")
            conn.close()
            return
    conn.close()

    # Update connection parameters to connect to the newly created database
    dataset_conn_params = conn_params.copy()
    dataset_conn_params["dbname"] = database_name

    # Connect to the new database to create tables
    conn = connect_db(dataset_conn_params)
    if conn is None:
        print(f"Cannot connect to database {database_name}.")
        return

    with conn.cursor() as cur:
        try:
            with open(sql_file_path, 'r') as f:
                cur.execute(f.read())
            conn.commit()
            print(f"Table creation SQL executed for dataset {database_name}.")
        except Exception as e:
            print(f"Failed to execute table creation SQL for dataset {database_name}: {e}")
            conn.rollback()
    conn.close()

    # Use multiprocessing to process tables in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(
            process_table,
            [(dataset, dataset_conn_params, table, sep) for table in tables]
        )

def main():
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from database_list import database_list

    # Load connection information
    conn_info_path = os.path.join(os.path.dirname(__file__), '../../conn.json')
    try:
        with open(conn_info_path, 'r') as f:
            conn_info = json.load(f)
    except Exception as e:
        print(f"Error loading connection info from {conn_info_path}: {e}")
        return

    required_keys = {'user', 'password', 'host', 'port'}
    if not required_keys.issubset(conn_info.keys()):
        print(f"Connection info missing required keys: {required_keys - set(conn_info.keys())}")
        return

    conn_params = {
        "user": conn_info['user'],
        "password": conn_info['password'],
        "host": conn_info['host'],
        "port": conn_info['port']
    }

    # Use multiprocessing to import multiple datasets in parallel
    with multiprocessing.Pool(processes=min(len(database_list), multiprocessing.cpu_count())) as pool:
        pool.starmap(
            import_dataset,
            [(dataset, conn_params) for dataset in database_list]
        )

if __name__ == "__main__":
    main()
