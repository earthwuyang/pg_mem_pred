import json
import os
import sys
from enum import Enum
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
import numpy as np
import collections


def gather_feature_statistics(data_dir, dataset):
    def gather_values_recursively(json_dict, value_dict=None):
        if value_dict is None:
            value_dict = collections.defaultdict(list)

        if isinstance(json_dict, dict):
            for k, v in json_dict.items():

                if not (isinstance(v, list) or isinstance(v, tuple) or isinstance(v, dict)):
                    value_dict[k].append(v)
                elif (isinstance(v, list) or isinstance(v, tuple)) and len(v) > 0 and \
                        (isinstance(v[0], int) or isinstance(v[0], float) or isinstance(v[0], str)):
                    for v_e in v:
                        value_dict[k].append(v_e)
                else:
                    gather_values_recursively(v, value_dict=value_dict)
        elif isinstance(json_dict, tuple) or isinstance(json_dict, list):
            for e in json_dict:
                gather_values_recursively(e, value_dict=value_dict)

        return value_dict


    class FeatureType(Enum):
        numeric = 'numeric'
        categorical = 'categorical'

        def __str__(self):
            return self.value


    run_stats = []
    parsed_plan_file = os.path.join(data_dir, dataset, 'zsce', 'train_plans.json')
    # parsed_plan_file = os.path.join(data_dir, dataset, 'zsce', 'tiny_plans.json')
    with open(parsed_plan_file, 'r') as f:
        run_stats.append(json.load(f))
    value_dict = gather_values_recursively(run_stats)

    statistics_dict = dict()
    for k, values in tqdm(value_dict.items()):
        values = [v for v in values if v is not None] # filter those are None
        if len(values) == 0:
            continue
        # if value is numerical
        if all([isinstance(v, int) or isinstance(v, float) or v is None for v in values]): # but previously v has been confirmed not be None?
            scaler = RobustScaler()
            np_values = np.array(values, dtype=np.float32).reshape(-1, 1)
            scaler.fit(np_values)

            statistics_dict[k] = dict(max=float(np_values.max()),
                                        scale=scaler.scale_.item(),
                                        center=scaler.center_.item(),
                                        type=str(FeatureType.numeric))
        # if value is categorical
        else:
            unique_values = set(values)
            statistics_dict[k] = dict(value_dict={v: id for id, v in enumerate(unique_values)},
                                        no_vals=len(unique_values),
                                        type=str(FeatureType.categorical))

    # save as json
    target_json_file = os.path.join(data_dir, dataset, 'zsce', "statistics_workload_combined.json")
    # target = 'tpch_data/statistics_workload_combined.json'
    os.makedirs(os.path.dirname(target_json_file), exist_ok=True)
    with open(target_json_file, 'w') as outfile:
        json.dump(statistics_dict, outfile)


if __name__ == '__main__':
    data_dir = '/home/wuy/DB/pg_mem_data'
    dataset = 'tpch_sf1'
    gather_feature_statistics(data_dir, dataset)