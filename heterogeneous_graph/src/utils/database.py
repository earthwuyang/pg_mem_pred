
def get_relpages_reltuples(conn, table_name):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT relpages, reltuples FROM pg_class WHERE relname = %s;", (table_name,))
            result = cur.fetchone()
            relpages = result[0]
            reltuples = result[1]
        return relpages, reltuples
    except Exception as e:
        print(f"Error fetching relpages and reltuples for {table_name}: {e}")
        return 0, 0

# Fetch table size
def get_table_size(conn, table_name):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_total_relation_size(%s);", (table_name,))
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
                SELECT avg_width, correlation, n_distinct, null_frac
                FROM pg_stats
                WHERE tablename = %s AND attname = %s;
            """, (table_name, column_name))
            result = cur.fetchone()
            avg_width = result[0] if result and result[0] is not None else 0
            correlation = result[1] if result and result[1] is not None else 0
            n_distinct = result[2] if result and result[2] is not None else 0
            null_frac = result[3] if result and result[3] is not None else 0
            column_features = [avg_width, correlation, n_distinct, null_frac]
        return column_features
    except Exception as e:
        print(f"Error fetching column features for {table_name}.{column_name}: {e}")
        return [0, 0, 0, 0]




# Fetch all unique data types from the database to create a mapping
def get_unique_data_types(conn):
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