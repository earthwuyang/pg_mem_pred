import json
import collections
from types import SimpleNamespace
explain_plan ={'Plan': {'Node Type': 'Aggregate', 'Strategy': 'Plain', 'Partial Mode': 'Finalize', 'Parallel Aware': False, 'Async Capable': False, 'Startup Cost': 183595.06, 'Total Cost': 183595.07, 'Plan Rows': 1, 'Plan Width': 48, 'Plans': [{'Node Type': 'Gather', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Async Capable': False, 'Startup Cost': 183594.83, 'Total Cost': 183595.04, 'Plan Rows': 2, 'Plan Width': 48, 'Workers Planned': 2, 'Single Copy': False, 'Plans': [{'Node Type': 'Aggregate', 'Strategy': 'Plain', 'Partial Mode': 'Partial', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Async Capable': False, 'Startup Cost': 182594.83, 'Total Cost': 182594.84, 'Plan Rows': 1, 'Plan Width': 48, 'Plans': [{'Node Type': 'Hash Join', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Async Capable': False, 'Join Type': 'Inner', 'Startup Cost': 27415.02, 'Total Cost': 182594.61, 'Plan Rows': 22, 'Plan Width': 16, 'Inner Unique': False, 'Hash Cond': '(supplier.s_nationkey = nation.n_nationkey)', 'Plans': [{'Node Type': 'Nested Loop', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Async Capable': False, 'Join Type': 'Inner', 'Startup Cost': 27413.46, 'Total Cost': 182592.75, 'Plan Rows': 22, 'Plan Width': 16, 'Inner Unique': False, 'Plans': [{'Node Type': 'Nested Loop', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Async Capable': False, 'Join Type': 'Inner', 'Startup Cost': 27413.04, 'Total Cost': 182582.86, 'Plan Rows': 22, 'Plan Width': 16, 'Inner Unique': False, 'Plans': [{'Node Type': 'Nested Loop', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Async Capable': False, 'Join Type': 'Inner', 'Startup Cost': 27412.61, 'Total Cost': 182550.13, 'Plan Rows': 44, 'Plan Width': 16, 'Inner Unique': False, 'Plans': [{'Node Type': 'Hash Join', 'Parent Relationship': 'Outer', 'Parallel Aware': True, 'Async Capable': False, 'Join Type': 'Inner', 'Startup Cost': 27412.33, 'Total Cost': 182527.59, 'Plan Rows': 71, 'Plan Width': 16, 'Inner Unique': False, 'Hash Cond': '((lineitem.l_partkey = partsupp.ps_partkey) AND (lineitem.l_suppkey = partsupp.ps_suppkey))', 'Plans': [{'Node Type': 'Seq Scan', 'Parent Relationship': 'Outer', 'Parallel Aware': True, 'Async Capable': False, 'Relation Name': 'lineitem', 'Alias': 'lineitem', 'Startup Cost': 0.0, 'Total Cost': 150010.59, 'Plan Rows': 175837, 'Plan Width': 12, 'Filter': "((l_linestatus <> 'F'::bpchar) AND (l_shipmode = 'MAIL'::bpchar))"}, {'Node Type': 'Hash', 'Parent Relationship': 'Inner', 'Parallel Aware': True, 'Async Capable': False, 'Startup Cost': 20784.33, 'Total Cost': 20784.33, 'Plan Rows': 333333, 'Plan Width': 12, 'Plans': [{'Node Type': 'Seq Scan', 'Parent Relationship': 'Outer', 'Parallel Aware': True, 'Async Capable': False, 'Relation Name': 'partsupp', 'Alias': 'partsupp', 'Startup Cost': 0.0, 'Total Cost': 20784.33, 'Plan Rows': 333333, 'Plan Width': 12}]}]}, {'Node Type': 'Index Scan', 'Parent Relationship': 'Inner', 'Parallel Aware': False, 'Async Capable': False, 'Scan Direction': 'Forward', 'Index Name': 'zero_shot_supplier_s_suppkey', 'Relation Name': 'supplier', 'Alias': 'supplier', 'Startup Cost': 0.29, 'Total Cost': 0.31, 'Plan Rows': 1, 'Plan Width': 8, 'Index Cond': '(s_suppkey = lineitem.l_suppkey)', 'Filter': '(s_acctbal >= 3174.392856419646)'}]}, {'Node Type': 'Index Scan', 'Parent Relationship': 'Inner', 'Parallel Aware': False, 'Async Capable': False, 'Scan Direction': 'Forward', 'Index Name': 'zero_shot_orders_o_orderkey', 'Relation Name': 'orders', 'Alias': 'orders', 'Startup Cost': 0.43, 'Total Cost': 0.73, 'Plan Rows': 1, 'Plan Width': 8, 'Index Cond': '(o_orderkey = lineitem.l_orderkey)', 'Filter': "(o_orderstatus <> 'O'::bpchar)"}]}, {'Node Type': 'Index Only Scan', 'Parent Relationship': 'Inner', 'Parallel Aware': False, 'Async Capable': False, 'Scan Direction': 'Forward', 'Index Name': 'zero_shot_customer_c_custkey', 'Relation Name': 'customer', 'Alias': 'customer', 'Startup Cost': 0.42, 'Total Cost': 0.44, 'Plan Rows': 1, 'Plan Width': 4, 'Index Cond': '(c_custkey = orders.o_custkey)'}]}, {'Node Type': 'Hash', 'Parent Relationship': 'Inner', 'Parallel Aware': False, 'Async Capable': False, 'Startup Cost': 1.25, 'Total Cost': 1.25, 'Plan Rows': 25, 'Plan Width': 4, 'Plans': [{'Node Type': 'Seq Scan', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Async Capable': False, 'Relation Name': 'nation', 'Alias': 'nation', 'Startup Cost': 0.0, 'Total Cost': 1.25, 'Plan Rows': 25, 'Plan Width': 4}]}]}]}]}]}, 'peakmem': 159572}

# load database statistics
with open('/home/wuy/DB/pg_mem_data/tpch/database_stats.json') as f:
    database_stats = json.load(f, object_hook=lambda d: SimpleNamespace(**d))


column_id_mapping = dict() # map (table, column) to a number
table_id_mapping = dict()
partial_column_name_mapping = collections.defaultdict(set)

# enrich column stats with table sizes
table_sizes = dict()
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



# Mapping of Node Types to op_name
NODE_TYPE_MAPPING = {
    "Aggregate": {
        "Finalize": "Finalize Aggregate",
        "Partial": "Partial Aggregate"
    },
    "Gather": "Gather",
    "Hash Join": "Hash Join",
    "Nested Loop": "Nested Loop",
    "Seq Scan": "Seq Scan",
    "Index Scan": "Index Scan",
    "Index Only Scan": "Index Only Scan",
    "Hash": "Hash"
}

def map_node_type(node):
    node_type = node.get("Node Type")
    if node_type == "Aggregate":
        partial_mode = node.get("Partial Mode", "Finalize")
        return NODE_TYPE_MAPPING.get(node_type, node_type) \
            .get(partial_mode, "Aggregate")
    return NODE_TYPE_MAPPING.get(node_type, node_type)

def map_filter_conditions(node):
    filters = []
    if "Hash Cond" in node:
        # Example: 'Hash Cond': '(supplier.s_nationkey = nation.n_nationkey)'
        condition = node["Hash Cond"]
        # Parsing the condition string can be complex; for simplicity, store as string
        filters.append({
            "operator": "=",
            "columns": condition.split("=")
        })
    if "Filter" in node:
        # Example: "Filter": "((l_linestatus <> 'F'::bpchar) AND (l_shipmode = 'MAIL'::bpchar))"
        condition = node["Filter"]
        # Parsing can be enhanced based on specific needs
        filters.append({
            "operator": "FILTER",
            "condition": condition
        })
    return filters if filters else None

def map_output_columns(node):
    # This function needs to be tailored based on how output columns are represented
    # For demonstration, returning an empty list
    return []

def transform_plan(node):
    target_node = {
        "plain_content": [],
        "plan_parameters": {
            "op_name": map_node_type(node),
            "est_startup_cost": node.get("Startup Cost", 0.0),
            "est_cost": node.get("Total Cost", 0.0),
            "est_card": float(node.get("Plan Rows", 0)),
            "est_width": float(node.get("Plan Width", 0)),
            "act_startup_cost": 0.0,  # Placeholder; actual values may come from elsewhere
            "act_time": 0.0,          # Placeholder
            "act_card": 0.0,          # Placeholder
            "output_columns": map_output_columns(node),
            "act_children_card": 0.0, # Placeholder
            "est_children_card": float(len(node.get("Plans", []))),
            "workers_planned": node.get("Workers Planned", 0)
        },
        "children": []
    }

    # Map additional fields if present
    if "Hash Cond" in node or "Filter" in node:
        target_node["plan_parameters"]["filter_columns"] = map_filter_conditions(node)
    
    # Recursively transform child plans
    for child in node.get("Plans", []):
        transformed_child = transform_plan(child)
        target_node["children"].append(transformed_child)
    
    return target_node

def transform_explain_plan(source_plan):
    root_plan = source_plan.get("Plan", {})
    transformed = transform_plan(root_plan)
    # Add top-level fields
    transformed["plan_runtime"] = source_plan.get("plan_runtime", 0)
    transformed["join_conds"] = source_plan.get("join_conds", [])
    transformed["peakmem"] = source_plan.get("peakmem", 0)
    return transformed

# Transform the plan
target_plan = transform_explain_plan(explain_plan)

# For demonstration, print the transformed plan as JSON
print(json.dumps(target_plan))