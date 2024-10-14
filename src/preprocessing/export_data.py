import mysql.connector
from tqdm import tqdm
import json
import os
import multiprocessing


def export_dataset(dataset, mysql_dataset):
    print(f"Exporting {dataset} from {mysql_dataset}...")
    conn = mysql.connector.connect(
        host='db.relational-data.org',
        user='guest',
        password='relational',
        database=mysql_dataset
    )

    # Load schema file
    with open(os.path.join(os.path.dirname(__file__), f'../../zsce/cross_db_benchmark/datasets/{dataset}/schema.json'), 'r') as f:
        schema = json.load(f)

    database_name = schema['name']
    tables = schema['tables']
    sep = schema['csv_kwargs']['sep']

    data_dir = '/home/wuy/DB/pg_mem_data/datasets'
    for table in tables:
        cursor = conn.cursor()

        # Get row count for progress bar
        cursor.execute(f"SELECT count(*) FROM {table}")
        num_rows = cursor.fetchone()[0]

        # Fetch data from the table
        cursor.execute(f"SELECT * FROM {table}")

        output_file = os.path.join(data_dir, database_name, f"{table}.csv")
        if os.path.exists(output_file):
            print(f"{output_file} already exists, skipping")
            continue

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Write to CSV file
        with open(output_file, 'w') as f:
            for row in tqdm(cursor, total=num_rows, desc=f"Exporting {table} of {database_name}"):
                f.write(sep.join(map(str, row)) + '\n')

        cursor.close()

    conn.close()


def main():
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from database_list import database_list, mysql_database_list

    # Create a pool of workers to process datasets in parallel
    mp_count = multiprocessing.cpu_count()
    # mp_count = 1  # Disable multiprocessing for now
    with multiprocessing.Pool(processes=mp_count) as pool:
        # Iterate over datasets and assign a process to each dataset
        pool.starmap(export_dataset, zip(database_list, mysql_database_list))


if __name__ == '__main__':
    main()
