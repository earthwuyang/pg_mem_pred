import os
import json
import pandas as pd
from tqdm import tqdm
import argparse

def get_explain_json_plans(data_dir, dataset):
    print(f"Processing {dataset}:")
    mem_list_file = os.path.join(data_dir, dataset, 'raw_data', 'mem_info.csv')
    query_dir = os.path.join(data_dir, dataset, 'raw_data','query_dir')
    plan_dir = os.path.join(data_dir, dataset, 'raw_data','plan_dir')
    query_dir = os.path.join(data_dir, dataset, 'raw_data','query_dir')

    json_plans = []
    mem_df = pd.read_csv(mem_list_file, sep=',')
 
    for i, row in tqdm(mem_df.iterrows(), total=len(mem_df), desc=f"{dataset}"):        
        plan_file = os.path.join(plan_dir, str(int(row['queryid']))+'.json')
        with open(plan_file, 'r') as f:
            plan_json = json.load(f)
        plan_json['peakmem'] = int(row['peakmem']) 
        plan_json['time'] = float(row['time'])
        query_file = os.path.join(query_dir, str(int(row['queryid']))+'.sql')
        with open(query_file, 'r') as f:
            query_str = f.read()
        plan_json['sql'] = query_str
        json_plans.append(plan_json)
    # with open(json_plan_file, 'w') as f:
    #     json.dump(json_plans, f)

    plans = json_plans
    print(f"dumping total {len(plans)} plans")
    with open(os.path.join(data_dir, dataset, 'total_plans.json'), 'w') as f:
        json.dump(plans, f)
    print(f"{len(plans)} plans dumped to {os.path.join(data_dir, dataset, 'total_plans.json')}")

    train_size = int(0.8 * len(plans))
    val_size = int(0.1 * len(plans))
    test_size = len(plans) - train_size - val_size

    train_plans = plans[:train_size]
    val_plans = plans[train_size:train_size+val_size]
    test_plans = plans[train_size+val_size:]
    print(f"train size: {len(train_plans)}, val size: {len(val_plans)}, test size: {len(test_plans)}")

    # save data to files
    with open( os.path.join(data_dir, dataset, 'train_plans.json'), 'w') as f:
        json.dump(train_plans, f)
    print(f"train plans saved to {os.path.join(data_dir, dataset, 'train_plans.json')}")

    # save data to files
    with open( os.path.join(data_dir, dataset, 'val_plans.json'), 'w') as f:
        json.dump(val_plans, f)
    print(f"val plans saved to {os.path.join(data_dir, dataset, 'val_plans.json')}")

    # save data to files
    with open( os.path.join(data_dir, dataset, 'test_plans.json'), 'w') as f:
        json.dump(test_plans, f)
    print(f"test plans saved to {os.path.join(data_dir, dataset, 'test_plans.json')}")

if __name__ == '__main__':
    data_dir = '/home/wuy/DB/pg_mem_data'
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', nargs='+', help='dataset names', type=str, default=['tpch_sf1'])
    args = argparser.parse_args()
    for dataset in args.dataset:
        get_explain_json_plans(data_dir, dataset)