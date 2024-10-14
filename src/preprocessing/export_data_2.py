import mysql.connector
from tqdm import tqdm
import json
import os
import multiprocessing


def export_table(dataset, mysql_dataset, table, sep, data_dir):
    print(f"Exporting table {table} from dataset {dataset}...")

    # Create a new MySQL connection for each table process
    conn = mysql.connector.connect(
        host='db.relational-data.org',
        user='guest',
        password='relational',
        database=mysql_dataset
    )

    try:
        # Fetch data from the table
        cursor = conn.cursor()

        # Get row count for progress bar
        cursor.execute(f"SELECT COUNT(*) FROM {mysql_dataset}.{table}")
        num_rows = cursor.fetchone()[0]

        # Fetch all data from the table
        cursor.execute(f"SELECT * FROM {mysql_dataset}.{table}")
        
        # Fetch all rows before starting to write to avoid unread result errors
        all_rows = cursor.fetchall()

        # Output CSV file path
        output_file = os.path.join(data_dir, dataset, f"{table}.csv")
        if os.path.exists(output_file):
            # print(f"{output_file} already exists, skipping")
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

        cursor.close()
    except Exception as e:
        print(f"Error exporting table {table} from dataset {dataset}: {e}")
    finally:
        conn.close()


def export_dataset(dataset, mysql_dataset):
    # print(f"Exporting dataset {dataset} from {mysql_dataset}...")

    # Load schema file
    with open(os.path.join(os.path.dirname(__file__), f'../../zsce/cross_db_benchmark/datasets/{dataset}/schema.json'), 'r') as f:
        schema = json.load(f)

    database_name = schema['name']
    tables = schema['tables']
    sep = schema['csv_kwargs']['sep']
    data_dir = '/home/wuy/DB/pg_mem_data/datasets'

    # Use multiprocessing to export tables in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(export_table, [(dataset, mysql_dataset, table, sep, data_dir) for table in tables])


def main():
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from database_list import database_list, mysql_database_list

    # Process each dataset sequentially, but export tables in parallel within each dataset
    for i, dataset in enumerate(database_list):
        mysql_dataset = mysql_database_list[i]
        export_dataset(dataset, mysql_dataset)


if __name__ == '__main__':
    main()
