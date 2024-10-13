import json
import os

import numpy as np
import pandas as pd

from cross_db_benchmark.benchmark_tools.column_types import Datatype
from cross_db_benchmark.benchmark_tools.utils import load_schema_json


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Datatype):
            return str(obj)
        else:
            return super(CustomEncoder, self).default(obj)


def column_stats(column, columntype, categorical_threshold=10000):  
    """
    Default method for encoding the datasets

    Args:
    column is pandas.core.series.Seires
    """
    nan_ratio = sum(column.isna()) / len(column)  # DataFrame.isna returns a boolean same-sized object indicating if the values are NA. such as None or np.NaN. Characters such as empty strings '' or numpy.inf.
    stats = dict(nan_ratio=nan_ratio)
    if column.dtype == object:  # original column def type is string, date, time, etc.

        if len(column.unique()) > categorical_threshold:
            stats.update(dict(datatype=Datatype.MISC))

        else:
            vals_sorted_by_occurence = list(column.value_counts().index)
            stats.update(dict(
                datatype=Datatype.CATEGORICAL,
                unique_vals=vals_sorted_by_occurence,
                num_unique=len(column.unique())
            ))

    else: # integer, float64

        percentiles = list(column.quantile(q=[0.1 * i for i in range(11)]))

        stats.update(dict(
            max=column.max(),
            min=column.min(),
            mean=column.mean(),
            num_unique=len(column.unique()),
            percentiles=percentiles,
        ))

        if columntype == 'char':
            stats.update(dict(datatype=Datatype.STRING_FLOAT))
        else:
            if column.dtype == int:
                stats.update(dict(datatype=Datatype.INT))

            else:
                stats.update(dict(datatype=Datatype.FLOAT))

    return stats


def generate_stats(data_dir, dataset, force=True):
    # read the schema file
    column_stats_path = os.path.join('cross_db_benchmark/datasets/', dataset, 'column_statistics.json')
    if os.path.exists(column_stats_path) and not force:
        print("Column stats already created")
        return

    schema = load_schema_json(dataset)

    column_type_file = f'cross_db_benchmark/datasets/{dataset}/column_type.json'
    if not os.path.exists(column_type_file):
        print(f"column types not extracted, {column_type_file} does not exist. See cross_db_benchmark/datasets/tpc_ds/scripts/script_to_get_column_type.py first.")
        exit()
    with open(column_type_file) as f:
        column_type = json.load(f)

    # read individual table csvs and derive statistics
    joint_column_stats = dict()
    for t in schema.tables:

        column_stats_table = dict()
        table_dir = os.path.join(data_dir, f'{t}.csv')
        assert os.path.exists(data_dir), f"Could not find table csv {table_dir}"
        print(f"Generating statistics for {t}")

        df_table = pd.read_csv(table_dir, **vars(schema.csv_kwargs))

        for column in df_table.columns:
            # print(f"column {column}")
            # print(f"df_table:\n {df_table}")
            # print(f"column_type {column_type}")
            column_stats_table[column] = column_stats(df_table[column], columntype = column_type[t][column])

        joint_column_stats[t] = column_stats_table

    # save to json
    with open(column_stats_path, 'w') as outfile:
        # workaround for numpy and other custom datatypes
        json.dump(joint_column_stats, outfile, cls=CustomEncoder)
