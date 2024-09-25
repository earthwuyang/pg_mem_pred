
def get_relpages_reltuples(db_stats, table_name):
    for table_stats in db_stats['table_stats']:
        if table_stats['relname'] == table_name:
            return table_stats['relpages'], table_stats['reltuples']
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
def get_column_features(db_stats, table_name, column_name):
    # print(f"table_name: {table_name}, column_name: {column_name}")
    for column_stats in db_stats['column_stats']:
        if column_stats['tablename'] == table_name and column_stats['attname'] == column_name:
            return column_stats['avg_width'], column_stats['correlation'], column_stats['n_distinct'], column_stats['null_frac']
    # print(f"return 0000")
    return 0, 0, 0, 0




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