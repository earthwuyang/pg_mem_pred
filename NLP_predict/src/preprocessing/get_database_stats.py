import psycopg2
import json
import os
import argparse

def get_database_stats(dataset):
    database = dataset
    
    # Load connection information
    with open(os.path.join(os.path.dirname(__file__), '../../conn.json')) as f:
        conn_info = json.load(f)

    conn_params = {
        "user": conn_info['user'],
        "password": conn_info['password'],
        "host": conn_info['host'],
        "port": conn_info['port'],
        "dbname": database
    }

    # Connect to PostgreSQL
    conn = psycopg2.connect(
        **conn_params
    )
    
    cur = conn.cursor()
    
    # Query to get column stats for TPCH tables
    column_stats_query = """
    SELECT
        st.relname AS tablename,
        a.attname AS attname,
        s.null_frac,
        s.avg_width,
        s.n_distinct,
        s.correlation,
        format_type(a.atttypid, a.atttypmod) AS data_type
    FROM
        pg_class st
    JOIN
        pg_namespace ns ON st.relnamespace = ns.oid
    JOIN
        pg_attribute a ON st.oid = a.attrelid
    JOIN
        pg_stats s ON s.schemaname = ns.nspname AND s.tablename = st.relname AND s.attname = a.attname
    WHERE
        a.attnum > 0
        AND NOT a.attisdropped
        AND st.relkind = 'r' -- Only include regular tables
        AND st.relname IN ('lineitem', 'orders', 'customer', 'nation', 'region', 'part', 'supplier', 'partsupp') -- Filter for TPCH tables
    ORDER BY
        tablename, attname;
    """
    
    # Query to get table stats for TPCH tables
    table_stats_query = """
    SELECT
        st.relname AS relname,
        st.reltuples,
        st.relpages
    FROM
        pg_class st
    JOIN
        pg_namespace ns ON st.relnamespace = ns.oid
    WHERE
        st.relkind = 'r' -- Only include regular tables
        AND st.relname IN ('lineitem', 'orders', 'customer', 'nation', 'region', 'part', 'supplier', 'partsupp') -- Filter for TPCH tables
    ORDER BY
        st.relname;
    """
    
    # Execute queries and fetch results
    cur.execute(column_stats_query)
    column_stats_result = cur.fetchall()

    cur.execute(table_stats_query)
    table_stats_result = cur.fetchall()
    
    # Construct the column_stats JSON array
    column_stats = []
    for row in column_stats_result:
        column_stats.append({
            'tablename': row[0],
            'attname': row[1],
            'null_frac': row[2],
            'avg_width': row[3],
            'n_distinct': row[4],
            'correlation': row[5],
            'data_type': row[6]
        })

    # Construct the table_stats JSON array
    table_stats = []
    for row in table_stats_result:
        table_stats.append({
            'relname': row[0],
            'reltuples': row[1],
            'relpages': row[2]
        })
    
    # Create the final JSON object
    database_stats = {
        'column_stats': column_stats,
        'table_stats': table_stats
    }
    
    # Close the cursor and connection
    cur.close()
    conn.close()

    # Convert the dictionary to a JSON string (if needed)
    return database_stats


def main():
    data_dir = '/home/wuy/DB/pg_mem_data'
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', nargs='+', type=str, default=['tpch_sf1'])
    args = argparser.parse_args()
    dataset_list = args.dataset
    for dataset in dataset_list:
        database_stats = get_database_stats(dataset)
        # print(json.dumps(database_stats))
        json_file = os.path.join(data_dir, dataset, 'database_stats.json')
        with open(json_file, 'w') as f:
            json.dump(database_stats, f, indent=2)

if __name__ == '__main__':
    main()