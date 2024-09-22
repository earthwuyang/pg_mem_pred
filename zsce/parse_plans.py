import os
import json
from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.postgres.parse_plan import parse_plans
from cross_db_benchmark.benchmark_tools.utils import load_json

def dumper(obj):
    try:
        return obj.toJSON()
    except:
        return obj.__dict__

def parse_raw(data_dir, dataset):
    source = os.path.join(data_dir, dataset, "zsce", 'raw_plan.json')
    target = os.path.join(data_dir, dataset, "zsce", 'parsed_plan.json')



    run_stats = load_json(source)
    print(f"json loaded")


    parsed_runs, stats = parse_plans(run_stats, min_runtime=0, max_runtime=100000000,
                                        parse_baseline=True, cap_queries=None,
                                        parse_join_conds=True,
                                        include_zero_card=True, explain_only=False)

    print(f"json dumping...")
    with open(target, 'w') as outfile:
        json.dump(parsed_runs, outfile, default=dumper)

if __name__ == '__main__':
    data_dir = '/home/wuy/DB/pg_mem_data'
    dataset = 'tpch'
    parse_raw(data_dir, dataset)