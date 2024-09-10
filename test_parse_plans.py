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
    
source = 'tpch_data_bk2/raw_plan.json'

target = 'tpch_data_bk2/parsed_plan.json'


run_stats = load_json(source)


parsed_runs, stats = parse_plans(run_stats, min_runtime=1, max_runtime=100000,
                                    parse_baseline=True, cap_queries=None,
                                    parse_join_conds=True,
                                    include_zero_card=False, explain_only=False)

with open(target, 'w') as outfile:
    json.dump(parsed_runs, outfile, default=dumper)