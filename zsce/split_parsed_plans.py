import os
import json

def split(data_dir, dataset):
    source = os.path.join(data_dir, dataset, 'zsce', 'parsed_plan.json')
    with open(source, 'r') as f:
        plans = json.load(f)
        ratio=0.8
    if dataset=='tpcds':
        plans = plans[:round(len(plans)* ratio)]
    train_size = int(0.8 * len(plans))
    val_size = int(0.1 * len(plans))
    test_size = len(plans) - train_size - val_size
    train_plans = plans[:train_size]
    val_plans = plans[train_size:train_size+val_size]
    test_plans = plans[train_size+val_size:]
    print(f"train size: {len(train_plans)}, val size: {len(val_plans)}, test size: {len(test_plans)}")
    target_dir = os.path.join(data_dir, dataset, 'zsce')
    with open(os.path.join(target_dir, 'train_plans.json'), 'w') as f:
        json.dump(train_plans, f)
    with open(os.path.join(target_dir, 'val_plans.json'), 'w') as f:
        json.dump(val_plans, f)
    with open(os.path.join(target_dir, 'test_plans.json'), 'w') as f:
        json.dump(test_plans, f)


if __name__ == '__main__':
    data_dir = '/home/wuy/DB/pg_mem_data'
    for dataset in ['tpch', 'tpcds']:
        split(data_dir, dataset)