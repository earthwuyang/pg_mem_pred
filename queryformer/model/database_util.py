# model/database_util.py

import numpy as np
import pandas as pd
import csv
import torch
import psycopg2
import logging
import os
import ast
import multiprocessing
from tqdm import tqdm
import pickle

## BFS should be enough
def floyd_warshall_rewrite(adjacency_matrix):
    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    M = adjacency_matrix.copy().astype('long')
    for i in range(nrows):
        for j in range(ncols):
            if i == j: 
                M[i][j] = 0
            elif M[i][j] == 0: 
                M[i][j] = 60
    
    for k in range(nrows):
        for i in range(nrows):
            for j in range(nrows):
                M[i][j] = min(M[i][j], M[i][k]+M[k][j])
    return M


def extract_column_stats(args):
    """
    Extracts min, max, cardinality, and number of unique values for a single table-column pair.
    
    Args:
        args (tuple): Contains (table, column, db_params, t2alias).
    
    Returns:
        dict or None: Dictionary with column statistics or None if an error occurs.
    """
    table, column, db_params, t2alias = args
    stats = {}
    
    # Skip 'sid' column if present
    if column == 'sid':
        return None
    
    try:
        # Establish a new database connection for each subprocess
        conn = psycopg2.connect(**db_params)
        conn.set_session(autocommit=True)
        cur = conn.cursor()
        
        # Construct SQL queries
        min_query = f"SELECT MIN({column}) FROM {table};"
        max_query = f"SELECT MAX({column}) FROM {table};"
        count_query = f"SELECT COUNT({column}) FROM {table} WHERE {column} IS NOT NULL;"
        distinct_query = f"SELECT COUNT(DISTINCT {column}) FROM {table} WHERE {column} IS NOT NULL;"
        
        # Execute queries and fetch results
        cur.execute(min_query)
        min_val = cur.fetchone()[0]
        
        cur.execute(max_query)
        max_val = cur.fetchone()[0]
        
        cur.execute(count_query)
        cardinality = cur.fetchone()[0]
        
        cur.execute(distinct_query)
        num_unique = cur.fetchone()[0]
        
        # Populate the stats dictionary
        stats = {
            'name': f"{t2alias.get(table, table[:2])}.{column}",
            'min': min_val,
            'max': max_val,
            'cardinality': cardinality,
            'num_unique_values': num_unique
        }
        
        logging.info(f"Extracted stats for '{table}.{column}': min={min_val}, max={max_val}, cardinality={cardinality}, unique={num_unique}")
        
        # Close the cursor and connection
        cur.close()
        conn.close()
        
        return stats
    
    except Exception as e:
        logging.error(f"Error extracting stats for '{table}.{column}': {e}")
        return None
    

def generate_column_min_max(db_params, schema, output_file, t2alias={}, max_workers=4, pool_minconn=1, pool_maxconn=10):
    """
    Connects to the PostgreSQL database, extracts min, max, cardinality, and number of unique values
    for each column in the specified tables using multiprocessing, and saves the statistics to a CSV file.
    
    Args:
        db_params (dict): Database connection parameters.
        tpcds_schema (dict): Schema dictionary mapping table names to their columns.
        output_file (str): Path to save the generated CSV file.
        t2alias (dict): Table aliases.
        max_workers (int): Maximum number of multiprocessing workers.
        pool_minconn (int): Minimum number of connections in the pool.
        pool_maxconn (int): Maximum number of connections in the pool.
    
    Returns:
        None
    """
    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    
    # List to hold table-column pairs
    table_column_pairs = []
    for table, columns in schema.items():
        for column in columns:
            if column == 'sid':
                continue  # Skip 'sid' column
            table_column_pairs.append((table, column, db_params, t2alias))
    
    # Determine the number of worker processes
    num_workers = min(len(table_column_pairs), max_workers)
    logging.info(f"Starting multiprocessing with {num_workers} workers for {len(table_column_pairs)} table-column pairs.")
    
    # Initialize a multiprocessing Pool with limited workers
    with multiprocessing.Pool(processes=num_workers) as pool_mp:
        # Use imap_unordered for better performance and to handle results as they come
        results = []
        for res in tqdm(pool_mp.imap_unordered(extract_column_stats, table_column_pairs), total=len(table_column_pairs)):
            if res is not None:
                results.append(res)
    
    # Create a DataFrame from the results
    stats_df = pd.DataFrame(results)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the DataFrame to CSV
    stats_df.to_csv(output_file, index=False)
    logging.info(f"Saved column statistics to '{output_file}'.")


def sample_table(args):
    """
    Samples a specified number of rows from a table and saves them to a CSV file.
    
    Args:
        args (tuple): Contains (table, db_params, sample_dir, num_samples).
    
    Returns:
        str: Path to the sampled CSV file or None if an error occurs.
    """
    table, db_params, sample_dir, num_samples = args
    sample_file = os.path.join(sample_dir, f"{table}_sampled.csv")
    if os.path.exists(sample_file):
        logging.info(f"Sample file '{sample_file}' already exists. Skipping.")
        return sample_file
    
    try:
        # Establish a new database connection for each subprocess
        conn = psycopg2.connect(**db_params)
        conn.set_session(autocommit=True)
        cur = conn.cursor()
        
        # Construct SQL query to sample rows
        # Using TABLESAMPLE SYSTEM for random sampling (adjust method as needed)
        # Alternatively, use ORDER BY RANDOM() LIMIT num_samples for exact sampling
        # query = f"SELECT * FROM {table} TABLESAMPLE SYSTEM ({(num_samples / 100000) * 100}) LIMIT {num_samples};"
        query = f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT {num_samples};"
        # Note: Adjust the sampling percentage based on table size
        
        cur.execute(query)
        sampled_rows = cur.fetchall()
        column_names = [desc[0] for desc in cur.description]
        
        # Save sampled rows to CSV
        df = pd.DataFrame(sampled_rows, columns=column_names)
        df.to_csv(sample_file, index=False)
        
        logging.info(f"Sampled {len(df)} rows from '{table}' and saved to '{sample_file}'.")
        
        # Close the cursor and connection
        cur.close()
        conn.close()
        
        return sample_file
    
    except Exception as e:
        logging.error(f"Error sampling table '{table}': {e}")
        return None


def sample_all_tables(db_params, schema, sample_dir='./data/tpcds/sampled_data/', num_samples=1000, max_workers=4):
    """
    Samples data for each table using multiprocessing.
    
    Args:
        db_params (dict): Database connection parameters.
        tpcds_schema (dict): Schema dictionary mapping table names to their columns.
        sample_dir (str): Directory to save sampled CSV files.
        num_samples (int): Number of samples per table.
        max_workers (int): Maximum number of multiprocessing workers.
    
    Returns:
        dict: Dictionary mapping table names to their sampled DataFrame.
    """
    # List of tables to sample
    tables = list(schema.keys())
    
    # Prepare arguments for each table
    args_list = [
        (table, db_params, sample_dir, num_samples)
        for table in tables
    ]
    
    # Determine the number of worker processes
    num_workers = min(len(tables), max_workers)
    logging.info(f"Starting multiprocessing with {num_workers} workers for sampling {len(tables)} tables.")
    
    # Initialize a multiprocessing Pool with limited workers
    with multiprocessing.Pool(processes=num_workers) as pool_mp:
        # Use imap_unordered with tqdm for progress tracking
        sampled_files = []
        for sample_file in tqdm(pool_mp.imap_unordered(sample_table, args_list), total=len(args_list)):
            if sample_file is not None:
                sampled_files.append(sample_file)
    
    # Load sampled data into a dictionary
    sampled_data = {}
    for sample_file in sampled_files:
        table = os.path.basename(sample_file).replace('_sampled.csv', '')
        df = pd.read_csv(sample_file)
        sampled_data[table] = df
    
    return sampled_data


def generate_histogram_single(args):
    """
    Generates histogram for a single table-column pair.

    Args:
        args (tuple): Contains (table, column, db_params, hist_dir, bin_number, t2alias).

    Returns:
        dict: Histogram record for the table-column.
    """
    table, column, db_params, hist_dir, bin_number, t2alias = args

    if column == 'sid':
        return None  # Skip the sample ID column

    hist_file = os.path.join(hist_dir, f"{table}_{column}_histogram.csv")

    if os.path.exists(hist_file):
        try:
            df = pd.read_csv(hist_file)
            bins = df['bins'].tolist()
            logging.info(f"Loaded histogram for '{table}.{column}' from '{hist_file}'.")
        except Exception as e:
            logging.error(f"Error loading histogram from '{hist_file}': {e}")
            bins = []
    else:
        try:
            # Establish a separate database connection for each subprocess
            conn = psycopg2.connect(**db_params)
            conn.set_session(autocommit=True)
            cur = conn.cursor()

            query = f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL"
            cur.execute(query)
            data = cur.fetchall()
            data = [row[0] for row in data if row[0] is not None]
            if not data:
                logging.warning(f"No data found for histogram generation for '{table}.{column}'.")
                cur.close()
                conn.close()
                return None

            # Compute percentiles as bin edges
            bins = np.percentile(data, np.linspace(0, 100, bin_number + 1))

            # Save histogram to CSV
            pd.DataFrame({'bins': bins.tolist()}).to_csv(hist_file, index=False)
            logging.info(f"Generated and saved histogram for '{table}.{column}' to '{hist_file}'.")

            cur.close()
            conn.close()
        except Exception as e:
            logging.error(f"Error generating histogram for '{table}.{column}': {e}")
            bins = []

    if len(bins) != 0:
        return {
            'table': table,
            'column': column,
            'bins': bins,
            'table_column': f"{t2alias.get(table, table[:2])}.{column}"
        }
    else:
        return None



def generate_histograms_entire_db(db_params, schema, hist_dir, bin_number, t2alias, max_workers):
    """
    Generates histograms for entire database tables.

    Args:
        db_params (dict): Database connection parameters.
        tpcds_schema (dict): Schema of the tpcds database.
        hist_dir (str): Directory to save histograms.
        bin_number (int): Number of bins for histograms.
        t2alias (dict): Table aliases.
        max_workers (int): Number of workers for parallel processing.

    Returns:
        pd.DataFrame: DataFrame containing histograms.
    """
    # Placeholder implementation
    # Replace this with actual histogram generation logic
    histograms = []
    for table, columns in schema.items():
        for column in columns:
            # Example: Generate dummy bins
            # Replace with actual histogram data retrieval
            bins = list(np.linspace(0, 10, bin_number))
            histograms.append({
                'table': table,
                'column': column,
                'bins': bins
            })
    hist_file_df = pd.DataFrame(histograms)
    
    # Ensure 'table_column' is unique
    hist_file_df['table_column'] = hist_file_df['table'] + '.' + hist_file_df['column']
    
    if hist_file_df['table_column'].duplicated().any():
        duplicated_entries = hist_file_df[hist_file_df['table_column'].duplicated(keep=False)]
        logging.error("Duplicate 'table_column' entries detected during histogram generation:")
        logging.error(duplicated_entries)
        raise ValueError("Duplicate 'table_column' entries found in histogram_entire.csv.")
    
    return hist_file_df

# model/database_util.py

def save_histograms(hist_file_df, save_path):
    """
    Saves the histograms DataFrame to a CSV file with comma-separated bins.

    Args:
        hist_file_df (pd.DataFrame): DataFrame containing histograms.
        save_path (str): Path to save the CSV file.
    """
    # Convert 'bins' lists to comma-separated strings
    hist_file_df['bins'] = hist_file_df['bins'].apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else x)
    
    # Create 'table_column' by concatenating 'table' and 'column' with a dot
    hist_file_df['table_column'] = hist_file_df['table'] + '.' + hist_file_df['column']
    
    # Check for duplicates
    if hist_file_df['table_column'].duplicated().any():
        duplicated_entries = hist_file_df[hist_file_df['table_column'].duplicated(keep=False)]
        logging.error("Duplicate 'table_column' entries detected during histogram generation:")
        logging.error(duplicated_entries)
        raise ValueError("Duplicate 'table_column' entries found in histogram_entire.csv.")
    
    # Save to CSV
    hist_file_df.to_csv(save_path, index=False)
    logging.info(f"Histograms saved to '{save_path}'.")



def load_entire_histograms(load_path='./data/tpcds/histogram_entire.csv'):
    """
    Loads the histograms DataFrame from a CSV file.

    Args:
        load_path (str): Path to load the histogram CSV.

    Returns:
        pd.DataFrame: DataFrame containing histograms.
    """
    if not os.path.exists(load_path):
        logging.error(f"Histogram file '{load_path}' does not exist.")
        raise FileNotFoundError(f"Histogram file '{load_path}' does not exist.")
    hist_file_df = pd.read_csv(load_path)
    # Safely parse the 'bins' column
    hist_file_df['bins'] = hist_file_df['bins'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    logging.info(f"Loaded entire table histograms from '{load_path}'.")
    return hist_file_df


def sample_single_query(args):
    """
    This function is no longer needed in the two-step sampling approach.
    Retained for reference but can be removed.
    """
    pass  # Implemented in 'generate_query_bitmaps'


def generate_query_bitmaps(query_file, alias2t, sample_dir='./data/tpcds/sampled_data/'):
    """
    Generates table sample bitmaps for each query based on pre-sampled table data.

    Args:
        query_file (pd.DataFrame): DataFrame containing queries.
        alias2t (dict): Mapping from table aliases to table names.
        sample_dir (str): Directory where sampled table CSV files are stored.

    Returns:
        list: List of dictionaries containing bitmaps for each query's tables.
    """
    # if table_sample_bitmaps exist, load them from file
    table_sample_bitmaps_file = os.path.join(sample_dir, 'table_sample_bitmaps.pkl')
    if os.path.exists(table_sample_bitmaps_file):
        with open(table_sample_bitmaps_file, 'rb') as f:
            table_sample_bitmaps = pickle.load(f)
        logging.info(f"Loaded table sample bitmaps from '{table_sample_bitmaps_file}'.")
        return table_sample_bitmaps
    
    table_sample_bitmaps = []

    for idx, row in tqdm(query_file.iterrows(), total=len(query_file)):
        query_id = idx
        tables = row['tables']
        joins = row['joins'] if 'joins' in row and row['joins'] else ''
        predicates_str = row['predicate'] if 'predicate' in row and row['predicate'] else ''

        # Initialize bitmap dictionary for this query
        bitmap_dict = {}

        involved_tables = [t.strip() for t in tables.split(',') if t.strip()]

        # Load sampled data for involved tables
        sampled_tables = {}
        for table in involved_tables:
            sample_file = os.path.join(sample_dir, f"{table}_sampled.csv")
            if os.path.exists(sample_file):
                df = pd.read_csv(sample_file)
                sampled_tables[table] = df
            else:
                logging.warning(f"Sample file '{sample_file}' does not exist for table '{table}'.")
                sampled_tables[table] = pd.DataFrame()  # Empty DataFrame

        # Ensure predicates is a string before splitting
        if isinstance(predicates_str, str) and predicates_str:
            predicates = predicates_str.split(',')
        else:
            predicates = []

        # Apply predicates to generate bitmaps
        # Assuming predicates are in the format: "(alias.column operator value)"
        # e.g., "(t.production_year < 2020)"
        for predicate in predicates:
            predicate = predicate.strip()
            if predicate.startswith('(') and predicate.endswith(')'):
                predicate = predicate[1:-1]
            else:
                logging.warning(f"Invalid predicate format '{predicate}' in query_id={query_id}.")
                continue

            try:
                # Split predicate into parts
                col, op, val = predicate.strip().split(' ', 2)
                if '.' in col:
                    alias, column = col.split('.', 1)
                else:
                    logging.warning(f"Cannot determine alias for column '{col}' in query_id={query_id}. Skipping predicate.")
                    continue

                table = alias2t.get(alias, alias)
                df = sampled_tables.get(table, pd.DataFrame())
                if df.empty:
                    # If no data, default bitmap to all zeros
                    bitmap = np.zeros(1000, dtype='uint8')  # Adjust size as needed
                else:
                    if column not in df.columns:
                        logging.warning(f"Column '{column}' not found in table '{table}' for query_id={query_id}.")
                        bitmap = np.zeros(len(df), dtype='uint8')
                    else:
                        try:
                            # Convert value to float for comparison
                            val_float = float(val)
                            if op == '=':
                                bitmap = (df[column] == val_float).astype('uint8').values
                            elif op == '<':
                                bitmap = (df[column] < val_float).astype('uint8').values
                            elif op == '>':
                                bitmap = (df[column] > val_float).astype('uint8').values
                            else:
                                logging.warning(f"Unsupported operator '{op}' in predicate of query_id={query_id}. Skipping predicate.")
                                bitmap = np.zeros(len(df), dtype='uint8')
                        except ValueError:
                            # logging.warning(f"Non-numeric value '{val}' in predicate '{predicate}' for query_id={query_id}. Skipping predicate.")
                            bitmap = np.zeros(len(df), dtype='uint8')
                bitmap_dict[f"{table}.{column}"] = bitmap
            except Exception as e:
                logging.error(f"Error parsing predicate '{predicate}' in query_id={query_id}: {e}")
                continue

        table_sample_bitmaps.append(bitmap_dict)
    # save table_sample_bitmaps to file
    with open(table_sample_bitmaps_file, 'wb') as f:
        pickle.dump(table_sample_bitmaps, f)
    logging.info(f"Saved table sample bitmaps to '{table_sample_bitmaps_file}'.")

    return table_sample_bitmaps



def formatJoin(json_node):
    """
    Formats join conditions from JSON plan nodes.
    """
    join = None
    if 'Hash Cond' in json_node:
        join = json_node['Hash Cond']
    elif 'Join Filter' in json_node:
        join = json_node['Join Filter']
    elif 'Index Cond' in json_node and not json_node['Index Cond'][-2].isnumeric():
        join = json_node['Index Cond']
    
    if join is not None:
        twoCol = join[1:-1].split(' = ')
        twoCol = [json_node['Alias'] + '.' + col if len(col.split('.')) == 1 else col for col in twoCol ] 
        join = ' = '.join(sorted(twoCol))
    
    return join

def formatFilter(plan):
    """
    Extracts and formats filters from JSON plan nodes.
    """
    alias = None
    if 'Alias' in plan:
        alias = plan['Alias']
    else:
        pl = plan
        while 'parent' in pl:
            pl = pl['parent']
            if 'Alias' in pl:
                alias = pl['Alias']
                break
    
    filters = []
    if 'Filter' in plan:
        filters.append(plan['Filter'])
    if 'Index Cond' in plan and plan['Index Cond'][-2].isnumeric():
        filters.append(plan['Index Cond'])
    if 'Recheck Cond' in plan:
        filters.append(plan['Recheck Cond'])
        
    return filters, alias

class Batch:
    def __init__(self, attn_bias, rel_pos, heights, x, y=None):
        self.heights = heights
        self.x, self.y = x, y
        self.attn_bias = attn_bias
        self.rel_pos = rel_pos
        
    def to(self, device):
        self.heights = self.heights.to(device)
        self.x = self.x.to(device)
        self.attn_bias, self.rel_pos = self.attn_bias.to(device), self.rel_pos.to(device)
        return self
    
    def __len__(self):
        return self.heights.size(0)


def collator(small_set):
    y = small_set[1]
    xs = [s['x'] for s in small_set[0]]
    
    num_graph = len(y)
    x = torch.cat(xs)
    attn_bias = torch.cat([s['attn_bias'] for s in small_set[0]])
    rel_pos = torch.cat([s['rel_pos'] for s in small_set[0]])
    heights = torch.cat([s['heights'] for s in small_set[0]])
    
    return Batch(attn_bias, rel_pos, heights, x), y


class Encoding:
    def __init__(self, column_min_max_vals, col2idx, op2idx={'>':0, '=':1, '<':2, '<=':3, '>=':4, '!=':5, '<>':6,'NA':7}):
        self.column_min_max_vals = column_min_max_vals
        self.col2idx = col2idx
        self.op2idx = op2idx
        
        idx2col = {v: k for k, v in col2idx.items()}
        self.idx2col = idx2col
        self.idx2op = {v: k for k, v in op2idx.items()}
        
        self.type2idx = {}
        self.idx2type = {}
        self.join2idx = {}
        self.idx2join = {}
        
        self.table2idx = {}
        self.idx2table = {}
    
    def normalize_val(self, column, val, log=False):
        mini, maxi = self.column_min_max_vals.get(column, (0,1))
        val_norm = 0.0
        if maxi > mini:
            val_norm = (val - mini) / (maxi - mini)
        return val_norm
    
    def encode_filters(self, filters=[], alias=None): 
        if len(filters) == 0:
            return {'colId':[self.col2idx.get('NA', 0)],
                   'opId': [self.op2idx.get('NA', 3)],
                   'val': [0.0]} 
        res = {'colId':[],'opId': [],'val': []}
        for filt in filters:
            filt = ''.join(c for c in filt if c not in '()')
            fs = filt.split(' AND ')
            for f in fs:
                try:
                    col, op, num = f.split(' ', maxsplit=2)
                    column = f"{alias}.{col}"
                    res['colId'].append(self.col2idx.get(column, self.col2idx.get('NA', 0)))
                    res['opId'].append(self.op2idx.get(op, self.op2idx.get('NA', 3)))
                    res['val'].append(self.normalize_val(column, float(num)))
                except ValueError as e:
                    # logging.warning(f"Error encoding filter '{f}': {e}. Skipping.")
                    if res['val'] == []:
                        res['val'].append(0.0)
                    if not isinstance(num, float):
                        res['val'].append(0.0)
                    continue
        return res
    
    def encode_join(self, join):
        if join not in self.join2idx:
            self.join2idx[join] = len(self.join2idx)
            self.idx2join[self.join2idx[join]] = join
        return self.join2idx[join]
    
    def encode_table(self, table):
        if table not in self.table2idx:
            self.table2idx[table] = len(self.table2idx)
            self.idx2table[self.table2idx[table]] = table
        return self.table2idx[table]
    
    def encode_type(self, nodeType):
        if nodeType not in self.type2idx:
            self.type2idx[nodeType] = len(self.type2idx)
            self.idx2type[self.type2idx[nodeType]] = nodeType
        return self.type2idx[nodeType]

HIST_BINS = 50  # Number of bins in each histogram
MAX_FILTERS = 30  # Maximum number of filters to consider
SAMPLE_SIZE = 1000  # Size of the sample data


def filterDict2Hist(hist_file, filterDict, encoding, hist_bins=HIST_BINS, max_filters=MAX_FILTERS):
    ress = []
    filter_count = 0
    for col, condition in filterDict.items():
        if filter_count >= max_filters:
            break  # Only consider up to max_filters filters
        filter_count += 1

        # Retrieve histogram bins for the column
        matching_bins = hist_file.loc[hist_file['table_column'] == col, 'bins']
        if matching_bins.empty:
            logging.warning(f"No histogram bins found for column '{col}'. Using default bins.")
            bins = np.linspace(0, 1, hist_bins + 1)  # Default bins
        else:
            bins = matching_bins.iloc[0]
            if isinstance(bins, str):
                bins = ast.literal_eval(bins)
            bins = np.array(bins)
            # Ensure bins have the correct size
            if len(bins) != hist_bins + 1:
                bins = np.linspace(bins[0], bins[-1], hist_bins + 1)

        op = condition['op']
        val = condition['value']
        mini, maxi = encoding.column_min_max_vals.get(col, (0, 1))
        val_unnorm = val * (maxi - mini) + mini

        res = np.zeros(hist_bins)
        if op == '=':
            idx = np.digitize([val_unnorm], bins) - 1
            if 0 <= idx[0] < len(res):
                res[idx[0]] = 1
        elif op == '<':
            idx = np.digitize([val_unnorm], bins) - 1
            if 0 <= idx[0] < len(res):
                res[:idx[0]] = 1
        elif op == '>':
            idx = np.digitize([val_unnorm], bins) - 1
            if 0 <= idx[0] < len(res):
                res[idx[0]:] = 1
        elif op == '<=':
            idx = np.digitize([val_unnorm], bins) - 1
            if 0 <= idx[0] < len(res):
                res[:idx[0]] = 1
                res[idx[0]] = 1
        elif op == '>=':
            idx = np.digitize([val_unnorm], bins) - 1
            if 0 <= idx[0] < len(res):
                res[idx[0]:] = 1
                res[idx[0]] = 1
        elif op in ['!=', '<>']:
            idx = np.digitize([val_unnorm], bins) - 1
            if 0 <= idx[0] < len(res):
                res[:idx[0]] = 1
                res[idx[0]+1:] = 1
        else:
            # logging.warning(f"Unsupported operator '{op}' in filter condition.")
            res = np.zeros(hist_bins)

        ress.append(res)

    # Pad or truncate the ress list to have max_filters entries
    while len(ress) < max_filters:
        ress.append(np.zeros(hist_bins))

    ress = ress[:max_filters]  # Ensure only max_filters histograms are considered

    # Concatenate the histograms
    ress = np.concatenate(ress) if ress else np.zeros(hist_bins * max_filters)

    # Ensure the length is correct
    assert len(ress) == hist_bins * max_filters, f"Histograms concatenated length {len(ress)} does not match expected {hist_bins * max_filters}."

    return ress


def formatJoin(json_node):
    """
    Formats join conditions from JSON plan nodes.
    """
    join = None
    if 'Hash Cond' in json_node:
        join = json_node['Hash Cond']
    elif 'Join Filter' in json_node:
        join = json_node['Join Filter']
    elif 'Index Cond' in json_node and not json_node['Index Cond'][-2].isnumeric():
        join = json_node['Index Cond']
    
    if join is not None:
        twoCol = join[1:-1].split(' = ')
        twoCol = [json_node['Alias'] + '.' + col if len(col.split('.')) == 1 else col for col in twoCol ] 
        join = ' = '.join(sorted(twoCol))
    
    return join

def formatFilter(plan):
    """
    Extracts and formats filters from JSON plan nodes.
    """
    alias = None
    if 'Alias' in plan:
        alias = plan['Alias']
    else:
        pl = plan
        while 'parent' in pl:
            pl = pl['parent']
            if 'Alias' in pl:
                alias = pl['Alias']
                break
    
    filters = []
    if 'Filter' in plan:
        filters.append(plan['Filter'])
    if 'Index Cond' in plan and plan['Index Cond'][-2].isnumeric():
        filters.append(plan['Index Cond'])
    if 'Recheck Cond' in plan:
        filters.append(plan['Recheck Cond'])
        
    return filters, alias

class TreeNode:
    def __init__(self, nodeType, typeId, filt, card, join, join_str, filterDict):
        self.nodeType = nodeType
        self.typeId = typeId
        self.filter = filt
        
        self.table = 'NA'
        self.table_id = 0
        self.query_id = None  # So that sample bitmap can recognize
        
        self.join = join
        self.join_str = join_str
        self.card = card  # 'Actual Rows'
        self.children = []
        self.rounds = 0
        
        self.filterDict = filterDict
        
        self.parent = None
        
        self.feature = None
        
    def addChild(self, treeNode):
        self.children.append(treeNode)
    
    def __str__(self):
        return '{} with {}, {}, {} children'.format(self.nodeType, self.filter, self.join_str, len(self.children))
    
    def __repr__(self):
        return self.__str__()
    
    @staticmethod
    def print_nested(node, indent=0): 
        print('--' * indent + '{} with {} and {}, {} childs'.format(node.nodeType, node.filter, node.join_str, len(node.children)))
        for child in node.children: 
            TreeNode.print_nested(child, indent + 1)
