total_plan_file='/home/wuy/DB/pg_mem_pred/tpch_data/parsed_plan.json'
train_plan_file='/home/wuy/DB/pg_mem_pred/tpch_data/train_plans.json'
val_plan_file='/home/wuy/DB/pg_mem_pred/tpch_data/val_plans.json'

import json
import random
import copy
import numpy as np
random.seed(1)

with open(total_plan_file, 'r') as f:
    total_plans = json.load(f)
print(f"total plans loaded")
total_parsed_plans = total_plans['parsed_plans']

train_indexes = random.sample(range(len(total_parsed_plans)), int(len(total_parsed_plans)*0.8))
print(f"train indexes generated ")
train_parsed_plans = [total_parsed_plans[i] for i in train_indexes]
print(f"train plans generated ")
val_parsed_plans = [total_parsed_plans[i] for i in range(len(total_parsed_plans)) if i not in train_indexes]
print(f"val plans generated")

database_stats = total_plans['database_stats']
run_kwargs = total_plans['run_kwargs']

train_plans =  {'parsed_plans': train_parsed_plans, 'database_stats': database_stats, 'run_kwargs': run_kwargs}
val_plans =  {'parsed_plans': val_parsed_plans, 'database_stats': database_stats, 'run_kwargs': run_kwargs}



with open(train_plan_file, 'w') as f:
    json.dump(train_plans, f)
print(f"train plans saved to {train_plan_file}")
with open(val_plan_file, 'w') as f:
    json.dump(val_plans, f)
print(f"val plans saved to {val_plan_file}")