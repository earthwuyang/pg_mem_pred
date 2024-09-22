import json
import os

def split(data_dir, dataset):
    # load json data
    json_file = os.path.join(data_dir, dataset, 'total_json_plans.json')
    with open(json_file, 'r') as f:
        plans = json.load(f)

    if dataset == 'tpcds':
        plans = plans[:round(len(plans)*0.711)]
    # split data into train, validation, and test sets
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

    # save data to files
    with open( os.path.join(data_dir, dataset, 'val_plans.json'), 'w') as f:
        json.dump(val_plans, f)

    # save data to files
    with open( os.path.join(data_dir, dataset, 'test_plans.json'), 'w') as f:
        json.dump(test_plans, f)


data_dir = '/home/wuy/DB/pg_mem_data'
# for dataset in ['tpch', 'tpcds']:
for dataset in ['tpcds']:
    print('Splitting', dataset)
    split(data_dir, dataset)