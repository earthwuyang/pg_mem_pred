import json
from functools import reduce
from operator import mul
import collections
from types import SimpleNamespace

# 加载数据库统计信息
database_stats_file = '/home/wuy/DB/pg_mem_data/tpch_sf1/database_stats.json'
with open(database_stats_file, 'r') as f:
    database_stats = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

column_id_mapping = dict()
table_id_mapping = dict()
partial_column_name_mapping = collections.defaultdict(set)

table_sizes=dict()
for table_stat in database_stats.table_stats:
        table_sizes[table_stat.relname] = table_stat.reltuples

for i, column_stat in enumerate(database_stats.column_stats):
    table = column_stat.tablename
    column = column_stat.attname
    column_stat.table_size = table_sizes[table]
    column_id_mapping[(table, column)] = i
    partial_column_name_mapping[column].add(table)

# similar for table statistics
for i, table_stat in enumerate(database_stats.table_stats):
    table = table_stat.relname
    table_id_mapping[table] = i


def load_explain_plan(json_str):
    """加载并解析 PostgreSQL 的 EXPLAIN JSON 计划"""
    return json.loads(json_str)

def parse_output_columns(output_list):
    """
    解析 Output 字段，将聚合函数和列索引分离。
    示例输入： "sum((nation.n_nationkey + partsupp.ps_partkey))"
    输出： [{'aggregation': 'SUM', 'columns': [0, 23]}, ...]
    """
    parsed_columns = []
    for item in output_list:
        if item.startswith('sum(') or item.startswith('SUM('):
            agg = 'SUM'
        elif item.startswith('avg(') or item.startswith('AVG('):
            agg = 'AVG'
        elif item.startswith('count(') or item.startswith('COUNT('):
            agg = 'COUNT'
        else:
            agg = 'None'
        
        # 假设列索引是固定的，这里需要根据实际情况调整
        # 这里简单提取数字作为列索引
        columns = [int(s) for s in item.replace(')', '').replace('(', ' ').replace(',', ' ').split() if s.isdigit()]
        parsed_columns.append({'aggregation': agg, 'columns': columns})
    return parsed_columns

def transform_node(node):
    """
    转换单个节点到目标格式
    """
    transformed = {
        'plain_content': [],  # 保持为空
        'plan_parameters': {
            'op_name': node.get('Node Type', 'Unknown'),
            'est_startup_cost': node.get('Startup Cost', 0.0),
            'est_cost': node.get('Total Cost', 0.0),
            'est_card': float(node.get('Plan Rows', 1)),
            'est_width': float(node.get('Plan Width', 0)),
            # 'act_startup_cost': float(node.get('Actual Startup Time', 0.0)),
            # 'act_time': float(node.get('Actual Total Time', 0.0)),
            # 'act_card': float(node.get('Actual Rows', 0.0)),
            'output_columns': parse_output_columns(node.get('Output', [])),
            'act_children_card': 0.0,  # 将在处理子节点后计算
            'est_children_card': 0.0,  # 将在处理子节点后计算
            'workers_planned': node.get('Workers Planned', 0)
        },
        'children': []
    }
    
    # 处理子节点
    children = node.get('Plans', [])
    for child in children:
        transformed_child = transform_node(child)
        transformed['children'].append(transformed_child)
    
    # 计算 act_children_card 和 est_children_card
    if children:
        est_card_product = reduce(mul, [float(child.get('Plan Rows', 1)) for child in children], 1.0)
        act_card_product = reduce(mul, [child['plan_parameters'].get('act_card', 1.0) for child in transformed['children']], 1.0)
        transformed['plan_parameters']['est_children_card'] = est_card_product
        transformed['plan_parameters']['act_children_card'] = act_card_product
    else:
        transformed['plan_parameters']['est_children_card'] = 1.0
        transformed['plan_parameters']['act_children_card'] = 1.0
    
    return transformed

def transform_plan(explain_json):
    """
    转换整个查询计划
    """
    plan = explain_json.get('Plan', {})
    transformed_plan = transform_node(plan)
    return transformed_plan

def main():
    # 将您的 EXPLAIN JSON 查询计划粘贴到此处
    json_file_path = '/home/wuy/DB/pg_mem_data/tpch_sf1/tiny_plans.json'
    with open(json_file_path, 'r') as f:
        plans = json.load(f)
    plan = plans[0]
    print(f"original plan: {json.dumps(plan)}")
    transformed_plan = transform_plan(plan)
    print(json.dumps(transformed_plan, indent=4))

if __name__ == '__main__':
    main()
