import mysql.connector
from tqdm import tqdm
import json
import os
import multiprocessing

import os
import mysql.connector
from tqdm import tqdm
import argparse

def export_table(dataset, mysql_dataset, table, sep, data_dir, overwrite):
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
        print(f"Table {table} of {dataset} has {num_rows} rows")

        # Fetch all data from the table
        cursor.execute(f"SELECT * FROM {mysql_dataset}.{table}")
        
        # Fetch all rows before starting to write to avoid unread result errors
        all_rows = cursor.fetchall()
        print(f"fetched {len(all_rows)} rows from {table} of {dataset}")

        # Output CSV file path
        output_file = os.path.join(data_dir, dataset, f"{table}.csv")
        if not overwrite and os.path.exists(output_file):
            print(f"Skipping {table} of {dataset}, file already exists")
            cursor.close()
            conn.close()
            return

        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Write to CSV file
        with open(output_file, 'w') as f:
            print(f"Exporting {table} of {dataset}, total rows: {num_rows}")
            for row in all_rows:
                # Replace None with empty string and join with separator
                row_data = [str(col) if col is not None else '' for col in row]
                f.write(sep.join(row_data) + '\n')
        print(f"Exported {table} of {dataset} to {output_file}")

        cursor.close()
    except Exception as e:
        print(f"Error exporting table {table} from dataset {dataset}: {e}")
    finally:
        conn.close()



def export_dataset(data_dir, dataset, mysql_dataset, overwrite):
    # print(f"Exporting dataset {dataset} from {mysql_dataset}...")

    # Load schema file
    with open(os.path.join(os.path.dirname(__file__), f'../../zsce/cross_db_benchmark/datasets/{dataset}/schema.json'), 'r') as f:
        schema = json.load(f)

    database_name = schema['name']
    tables = schema['tables']
    # sep = schema['csv_kwargs']['sep']
    sep = '|'
    

    # Use multiprocessing to export tables in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(export_table, [(dataset, mysql_dataset, table, sep, data_dir, overwrite) for table in tables])
    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #     args_list = [(dataset, mysql_dataset, table, sep, data_dir, overwrite) for table in tables]
    #     for _ in pool.imap(export_table, args_list):
    #         pass  # We don't need to do anything with the results, just wait for all tasks to complete


def main():
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from database_list import database_list, mysql_database_list

    data_dir = '/home/wuy/DB/pg_mem_data/datasets'

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_dir', type=str, default=data_dir, help='Directory to store exported data')
    argparser.add_argument('--dataset', nargs='+', type=str, help='Dataset to export', default=None)
    argparser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing data', default=False)
    args = argparser.parse_args()

    
    if args.dataset is not None:
        mysql_database_index = None
        for i,ds in enumerate(mysql_database_list):
            if ds.lower() == args.dataset[0]:
                mysql_database_index = i
        export_dataset(data_dir, args.dataset[0], mysql_database_list[mysql_database_index], args.overwrite)
    else:
        # Process each dataset sequentially, but export tables in parallel within each dataset
        for i, dataset in enumerate(database_list):
            mysql_dataset = mysql_database_list[i]
            export_dataset(data_dir, dataset, mysql_dataset, args.overwrite)


if __name__ == '__main__':
    main()
