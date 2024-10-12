import os
import json

def split(data_dir, dataset):
    source = os.path.join(data_dir, dataset, 'zsce', 'parsed_plan.json')
    with open(source, 'r') as f:
        plans = json.load(f)
    print(f"""{len(plans['parsed_plans'])} plans loaded from {source}""")
    parsed_plans = plans['parsed_plans']

    train_size = int(0.8 * len(parsed_plans))
    val_size = int(0.1 * len(parsed_plans))
    test_size = len(parsed_plans) - train_size - val_size
    train_plans = parsed_plans[:train_size]
    val_plans = parsed_plans[train_size:train_size+val_size]
    test_plans = parsed_plans[train_size+val_size:]
    print(f"train size: {len(train_plans)}, val size: {len(val_plans)}, test size: {len(test_plans)}")


    target_dir = os.path.join(data_dir, dataset, 'zsce')
    plans['parsed_plans'] = train_plans
    with open(os.path.join(target_dir, 'train_plans.json'), 'w') as f:
        json.dump(plans, f)
    print(f"train plans saved to {os.path.join(target_dir, 'train_plans.json')}")

    plans['parsed_plans'] = val_plans
    with open(os.path.join(target_dir, 'val_plans.json'), 'w') as f:
        json.dump(plans, f)
    print(f"val plans saved to {os.path.join(target_dir, 'val_plans.json')}")

    plans['parsed_plans'] = test_plans
    with open(os.path.join(target_dir, 'test_plans.json'), 'w') as f:
        json.dump(plans, f)
    print(f"test plans saved to {os.path.join(target_dir, 'test_plans.json')}")

if __name__ == '__main__':
    data_dir = '/home/wuy/DB/pg_mem_data'
    for dataset in ['tpcds_sf1']:
        split(data_dir, dataset)