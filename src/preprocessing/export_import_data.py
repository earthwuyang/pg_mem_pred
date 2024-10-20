import mysql.connector
import os
import json
import multiprocessing
import psycopg2
from psycopg2 import sql
from tqdm import tqdm


# MySQL Export Function
def export_table(dataset, mysql_dataset, table, sep, data_dir):
    print(f"Exporting table {table} from dataset {dataset}...")

    conn = mysql.connector.connect(
        host='db.relational-data.org',
        user='guest',
        password='relational',
        database=mysql_dataset
    )

    try:
        cursor = conn.cursor()

        # Get row count for progress bar
        cursor.execute(f"SELECT COUNT(*) FROM {mysql_dataset}.{table}")
        num_rows = cursor.fetchone()[0]

        # Fetch all data from the table
        cursor.execute(f"SELECT * FROM {mysql_dataset}.{table}")
        all_rows = cursor.fetchall()

        # Output CSV file path
        output_file = os.path.join(data_dir, dataset, f"{table}.csv")
        if os.path.exists(output_file):
            cursor.close()
            conn.close()
            return

        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Write to CSV file
        with open(output_file, 'w') as f:
            print(f"Exporting {table} of {dataset}, total rows: {num_rows}")
            for row in tqdm(all_rows, total=num_rows, desc=f"Exporting {table} of {dataset}"):
                f.write(sep.join(map(str, row)) + '\n')
        print(f"Exported {table} of {dataset} to {output_file}")

        cursor.close()
    except Exception as e:
        print(f"Error exporting table {table} from dataset {dataset}: {e}")
    finally:
        conn.close()


# PostgreSQL Import Function
def preprocess_csv(csv_file):
    temp_csv_file = csv_file + ".tmp"
    with open(csv_file, 'r') as infile, open(temp_csv_file, 'w') as outfile:
        for line in infile:
            processed_line = line.replace('None', '')
            outfile.write(processed_line)
    return temp_csv_file


def connect_db(conn_params):
    try:
        conn = psycopg2.connect(**conn_params)
        return conn
    except Exception as e:
        print(f"Connection fails: {e}")
        return None


def import_data(conn_params, table_name, csv_file, sep, dataset):
    specific_conn_params = conn_params.copy()
    specific_conn_params['dbname'] = dataset

    conn = connect_db(conn_params)
    if conn is None:
        return

    try:
        temp_csv_file = preprocess_csv(csv_file)
        with conn.cursor() as cur:
            copy_sql = sql.SQL("""
                COPY {table} FROM STDIN WITH CSV HEADER DELIMITER '{sep}' NULL AS '';
            """).format(
                table=sql.Identifier(table_name),
                sep=sql.SQL(sep)
            )
            with open(temp_csv_file, 'r') as f:
                cur.copy_expert(copy_sql, f)

        conn.commit()
        print(f"Table {table_name} loaded successfully.")
    except Exception as e:
        conn.rollback()
        print(f"Failed to load table {table_name}: {e}")
    finally:
        if os.path.exists(temp_csv_file):
            os.remove(temp_csv_file)
        conn.close()


def process_table(dataset, conn_params, table, sep):
    DATA_DIR = f'/home/wuy/DB/pg_mem_data/datasets/{dataset}'
    csv_file = os.path.join(DATA_DIR, f"{table}.csv")

    if os.path.exists(csv_file):
        print(f"Loading table {table} for dataset {dataset}...")
        import_data(conn_params, table, csv_file, sep, dataset)
    else:
        print(f"File {csv_file} does not exist, skipping table {table} for dataset {dataset}.")


# Combined Export and Import Function
def export_and_import_table(dataset, mysql_dataset, table, sep, data_dir, conn_params):
    export_table(dataset, mysql_dataset, table, sep, data_dir)  # Export table from MySQL
    process_table(dataset, conn_params, table, sep)  # Import table into PostgreSQL


def export_and_import_dataset(dataset, mysql_dataset, conn_params):
    # Load schema
    with open(os.path.join(os.path.dirname(__file__), f'../../zsce/cross_db_benchmark/datasets/{dataset}/schema.json'), 'r') as f:
        schema = json.load(f)

    tables = schema['tables']
    sep = schema['csv_kwargs']['sep']
    data_dir = '/home/wuy/DB/pg_mem_data/datasets'

    # Create database if not exists
    conn = connect_db(conn_params)
    if conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            try:
                cur.execute(f"CREATE DATABASE {dataset};")
                print(f"Database {dataset} created successfully.")
            except Exception as e:
                print(f"Database {dataset} already exists.")

        conn.close()

    conn_params['dbname'] = dataset
    conn = connect_db(conn_params)
    if conn:
        # Create tables for the current dataset
        with conn.cursor() as cur:
            sql_file_path = os.path.join(os.path.dirname(__file__), f"../../zsce/cross_db_benchmark/datasets/{dataset}/schema_sql/postgres.sql")
            with open(sql_file_path, 'r') as f:
                try:
                    cur.execute(f.read())
                except Exception as e:
                    print(f"Failed to execute table creation SQL for dataset {dataset}: {e}")
            conn.commit()
            print(f"Table creation SQL executed for dataset {dataset}.")

    # Use multiprocessing to export and import tables in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(export_and_import_table, [(dataset, mysql_dataset, table, sep, data_dir, conn_params) for table in tables])


# Main function to run the combined tasks
def main():
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from database_list import database_list, mysql_database_list

    # Load connection info
    with open(os.path.join(os.path.dirname(__file__), '../../conn.json')) as f:
        conn_info = json.load(f)

    conn_params = {
        "user": conn_info['user'],
        "password": conn_info['password'],
        "host": conn_info['host'],
        "port": conn_info['port'],
        "dbname": conn_info['user']
    }

    # Process each dataset one by one
    for i, dataset in enumerate(database_list):
        mysql_dataset = mysql_database_list[i]
        export_and_import_dataset(dataset, mysql_dataset, conn_params)


if __name__ == "__main__":
    main()