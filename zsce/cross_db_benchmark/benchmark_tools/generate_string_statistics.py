import collections
import json
import os

import pandas as pd

from cross_db_benchmark.benchmark_tools.column_types import Datatype
from cross_db_benchmark.benchmark_tools.utils import load_schema_json, load_column_statistics


def generate_string_stats(data_dir, dataset, force=True, max_sample_vals=100000, min_str_occ=0.01,
                          verbose=False):
    # read the schema file
    string_stats_path = os.path.join('cross_db_benchmark/datasets/', dataset, 'string_statistics.json')
    if os.path.exists(string_stats_path) and not force:
        print("String stats already created")
        return

    schema = load_schema_json(dataset)
    column_stats = load_column_statistics(dataset)

    column_type_file = f'cross_db_benchmark/datasets/{dataset}/column_type.json'
    if not os.path.exists(column_type_file):
        print(f"column types not extracted, {column_type_file} does not exist. See cross_db_benchmark/datasets/tpc_ds/scripts/script_to_get_column_type.py first.")
        exit()
    with open(column_type_file) as f:
        column_type = json.load(f)
    tables = {}
    for table, columns in column_type.items():
        tables[table] = columns.keys()

    cols_with_freq_words = 0
    string_stats = dict()
    for table, cols in vars(column_stats).items():

        string_stats[table] = dict()
        table_dir = os.path.join(data_dir, f'{table}.csv')
        assert os.path.exists(data_dir), f"Could not find table csv {table_dir}"
        if verbose:
            print(f"Generating string statistics for {table}")

        df_table = pd.read_csv(table_dir, nrows=max_sample_vals, **vars(schema.csv_kwargs), names=tables[table])

        for c, col_stats in vars(cols).items():
            if col_stats.datatype in {str(Datatype.CATEGORICAL), str(Datatype.MISC)}:
                col_vals = df_table[c]
                # do not consider too many values
                col_vals = col_vals[:max_sample_vals]
                len_strs = len(col_vals)

                # check how often a word occurs
                word_vals = collections.defaultdict(int)
                try:
                    split_col_vals = col_vals.str.split(' ')
                except:
                    continue

                for scol_vals in split_col_vals:
                    if not isinstance(scol_vals, list):
                        continue
                    for v in scol_vals:
                        if not isinstance(v, str):
                            continue
                        word_vals[v] += 1

                # how often should a word appear
                min_expected_occ = max(int(len_strs * min_str_occ), 1)

                freq_str_words = list()
                for val, occ in word_vals.items():
                    if occ > min_expected_occ:
                        freq_str_words.append(val)

                if len(freq_str_words) > 0:
                    if verbose:
                        print(f"Found {len(freq_str_words)} frequent words for {c} "
                              f"(expected {min_expected_occ}/{len_strs})")

                    cols_with_freq_words += 1
                    string_stats[table][c] = dict(freq_str_words=freq_str_words)

    # save to json
    with open(string_stats_path, 'w') as outfile:
        print(f"Found {cols_with_freq_words} string-queryable columns for dataset {dataset}")
        # workaround for numpy and other custom datatypes
        json.dump(string_stats, outfile)
