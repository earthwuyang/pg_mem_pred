import os
import psycopg2
import json


def main(dataset, conn_params):
    print(f"analyzing {dataset}")
    conn_params['dbname'] = dataset
    conn = psycopg2.connect(**conn_params)
    with conn.cursor() as cur:
        cur.execute("analyze;")
        conn.commit()
    conn.close()
        


if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from database_list import full_database_list

    # Load connection information
    with open(os.path.join(os.path.dirname(__file__), '../../conn.json')) as f:
        conn_info = json.load(f)

    conn_params = {
        "user": conn_info['user'],
        "password": conn_info['password'],
        "host": conn_info['host'],
        "port": conn_info['port']
    }

    for dataset in full_database_list:
        main(dataset, conn_params)