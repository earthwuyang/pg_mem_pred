import json
import psycopg2
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

def get_relpages_reltuples(conn, table_name):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT relpages, reltuples FROM pg_class WHERE relname = %s;", (table_name,))
            result = cur.fetchone()
            relpages = result[0]
            reltuples = result[1]
        return relpages, reltuples
    except Exception as e:
        # raise Exception(f"Error fetching relpages and reltuples for {table_name}: {e}")
        print(f"Error fetching relpages and reltuples for {table_name}: {e}")
        return 0, 0

# Fetch table size
def get_table_size(conn, table_name):
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT pg_total_relation_size('\"{table_name}\"');")
            table_size = cur.fetchone()[0]  # Size in bytes
        return table_size
    except Exception as e:
        print(f"Error fetching table size for {table_name}: {e}")
        return 0

# Fetch number of columns and their data types
def get_columns_info(conn, table_name):
    try:
        with conn.cursor() as cur:
            # Fetch column names and data types
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = %s;
            """, (table_name,))
            columns = cur.fetchall()  # List of tuples: (column_name, data_type)
        
        return columns
    except Exception as e:
        print(f"Error fetching columns info for {table_name}: {e}")
        return []

# Fetch average width of a column
def get_column_features(conn, table_name, column_name):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ps.avg_width, ps.correlation, ps.n_distinct, ps.null_frac, ic.data_type
                FROM pg_stats ps
                JOIN information_schema.columns ic
                ON ps.tablename = ic.table_name AND ps.attname = ic.column_name
                WHERE ps.tablename = %s AND ps.attname = %s;
            """, (table_name, column_name))
            result = cur.fetchone()
            avg_width = result[0] if result and result[0] is not None else 0
            correlation = result[1] if result and result[1] is not None else 0
            n_distinct = result[2] if result and result[2] is not None else 0
            null_frac = result[3] if result and result[3] is not None else 0
            data_type = result[4] if result and result[4] is not None else ''
            column_features = [avg_width, correlation, n_distinct, null_frac, data_type]
        return column_features
    except Exception as e:
        print(f"Error fetching column features for {table_name}.{column_name}: {e}")
        return [0, 0, 0, 0, 0]




# Fetch all unique data types from the database to create a mapping
def get_unique_data_types(conn):
    return ['character', 'date', 'character varying', 'integer', 'numeric', 'double precision']
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT data_type
                FROM information_schema.columns
                WHERE table_schema = 'public';
            """)
            data_types = [row[0] for row in cur.fetchall()]
        return data_types
    except Exception as e:
        print(f"Error fetching unique data types: {e}")
        return []
    
def get_tables(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public';
            """)
            tables = [row[0] for row in cur.fetchall()]
        return tables
    except Exception as e:
        print(f"Error fetching tables: {e}")
        return []


def get_db_stats(dataset, conn_info):
    database = dataset
    db_stats = {}
    conn = psycopg2.connect(**conn_info)

    unique_data_types = get_unique_data_types(conn) # len 6
    db_stats['unique_data_types'] = unique_data_types
    db_stats['tables'] = {}

    relpages_list = []
    reltuples_list = []
    table_size_list = []

    # Initialize lists for scaling
    avg_widths = []
    correlations = []
    n_distincts = []
    null_fracs = []
    
    tables = get_tables(conn)
    for table in tables:
        relpages, reltuples = get_relpages_reltuples(conn, table)
        table_size = get_table_size(conn, table)

        # Collect relpages and reltuples for scaling
        relpages_list.append(relpages)
        reltuples_list.append(reltuples)
        table_size_list.append(table_size)

        
        db_stats['tables'][table] = {'relpages': relpages, 'reltuples': reltuples, 'table_size': table_size}
        db_stats['tables'][table]['column_features'] = {}


        columns = get_columns_info(conn, table)
        for column in columns:
            column_name = column[0]
            data_type = column[1]
            avg_width, correlation, n_distinct, null_frac, data_type = get_column_features(conn, table, column_name)

            # Collect values for scaling
            avg_widths.append(avg_width)
            correlations.append(correlation)
            n_distincts.append(n_distinct)
            null_fracs.append(null_frac)

            db_stats['tables'][table]['column_features'][f"{table}.{column_name}"] = {
                'avg_width': avg_width,
                'correlation': correlation,
                'n_distinct': n_distinct,
                'null_frac': null_frac,
                'data_type': data_type
            }
        

    # Scale the collected values using RobustScaler
    column_scaler = RobustScaler()
    table_scaler = RobustScaler()
    column_scaled_features = column_scaler.fit_transform(np.array([avg_widths, correlations, n_distincts, null_fracs]).T)
    table_scaled_features = table_scaler.fit_transform(np.array([relpages_list, reltuples_list, table_size_list]).T)


    # Assign scaled values back to the db_stats
    for i, column in enumerate(columns):
        column_name = column[0]
        db_stats['tables'][table]['column_features'][f"{table}.{column_name}"]['avg_width'] = column_scaled_features[i][0]
        db_stats['tables'][table]['column_features'][f"{table}.{column_name}"]['correlation'] = column_scaled_features[i][1]
        db_stats['tables'][table]['column_features'][f"{table}.{column_name}"]['n_distinct'] = column_scaled_features[i][2]
        db_stats['tables'][table]['column_features'][f"{table}.{column_name}"]['null_frac'] = column_scaled_features[i][3]

    # Update relpages and reltuples in db_stats
    for i, table_name in enumerate(tables):
        db_stats['tables'][table_name]['relpages'] = table_scaled_features[i][0]
        db_stats['tables'][table_name]['reltuples'] = table_scaled_features[i][1]
        db_stats['tables'][table_name]['table_size'] = table_scaled_features[i][2]


    return db_stats

if __name__ == '__main__':
    import psycopg2
    database = 'tpch_sf1'
    db_stats = {}
    conn = psycopg2.connect(database=database, user="wuy", password='', host='localhost')

    unique_data_types = get_unique_data_types(conn)
    db_stats['unique_data_types'] = unique_data_types
    db_stats['tables'] = {}

    tables = get_tables(conn)
    for table in tables:
        relpages, reltuples = get_relpages_reltuples(conn, table)
        table_size = get_table_size(conn, table)
        db_stats['tables'][table] = {'relpages': relpages, 'reltuples': reltuples, 'table_size': table_size}
        db_stats['tables'][table]['column_features'] = {}
        columns = get_columns_info(conn, table)
        for column in columns:
            column_name = column[0]
            data_type = column[1]
            avg_width, correlation, n_distinct, null_frac, data_type = get_column_features(conn, table, column_name)
            # print(f"{table}.{column_name}: {data_type}, {column_features}")  # Print column features for each column in the table        
            # print(f"{table}: {relpages} pages, {reltuples} tuples, {table_size} bytes")  # Print table statistics
            db_stats['tables'][table]['column_features'][f"{table}.{column_name}"] = {}
            db_stats['tables'][table]['column_features'][f"{table}.{column_name}"]['avg_width'] = avg_width
            db_stats['tables'][table]['column_features'][f"{table}.{column_name}"]['correlation'] = correlation
            db_stats['tables'][table]['column_features'][f"{table}.{column_name}"]['n_distinct'] = n_distinct
            db_stats['tables'][table]['column_features'][f"{table}.{column_name}"]['null_frac'] = null_frac
            db_stats['tables'][table]['column_features'][f"{table}.{column_name}"]['data_type'] = data_type

    print(db_stats)
    # with open('db_stats.json', 'w') as f:
    #     json.dump(db_stats, f, indent=4)

    conn.close()