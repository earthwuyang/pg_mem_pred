import os
import json
import pandas as pd
from tqdm import tqdm
def get_explain_json_plans(data_dir, dataset):
    mem_list_file = os.path.join(data_dir, dataset, 'raw_data', 'mem_info.csv')
    query_dir = os.path.join(data_dir, dataset, 'raw_data','query_dir')
    plan_dir = os.path.join(data_dir, dataset, 'raw_data','plan_dir')
    json_plan_file = os.path.join(data_dir, dataset, 'total_json_plans.json')

    json_plans = []
    mem_df = pd.read_csv(mem_list_file, sep=',')
 
    for i, row in tqdm(mem_df.iterrows(), total=len(mem_df), desc=f"{dataset}"):        
        plan_file = os.path.join(plan_dir, str(int(row['queryid']))+'.json')
        with open(plan_file, 'r') as f:
            plan_json = json.load(f)
        plan_json['peakmem'] = int(row['peakmem']) 
        plan_json['time'] = float(row['time'])
        json_plans.append(plan_json)
    with open(json_plan_file, 'w') as f:
        json.dump(json_plans, f)


data_dir = '/home/wuy/DB/pg_mem_data'
for dataset in ['tpch_sf1']:
    get_explain_json_plans(data_dir, dataset)