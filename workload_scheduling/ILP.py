import logging
import json
from typing import List, Dict
import pulp
import math

class Query:
    def __init__(self, qid, sql, explain_json_plan, pred_peakmem, pred_duration, qtype='large'):
        self.id = qid
        self.sql = sql
        self.explain_json_plan = explain_json_plan
        self.pred_peakmem = pred_peakmem  # M_i
        self.pred_duration = pred_duration  # T_i
        self.type = qtype
        # Will be filled by the ILP solution:
        self.x = None
        self.y = None

def load_queries(plan_file: str, total_query_memory_limit_kb: int) -> List[Query]:
    """
    Loads queries from a JSON file, skipping those
    that exceed the memory limit.
    """
    try:
        with open(plan_file, 'r') as f:
            plans = json.load(f)
    except FileNotFoundError:
        logging.error(f"Plan file not found: {plan_file}")
        return []
    except json.JSONDecodeError as jde:
        logging.error(f"JSON decode error in plan file: {jde}")
        return []
    
    queries = []
    for idx, plan in enumerate(plans):
        mem_use = plan.get('peakmem', 0)
        if mem_use >= total_query_memory_limit_kb:
            continue
        q = Query(
            qid=(idx + 1),
            sql=plan.get('sql', ''),
            explain_json_plan=plan,
            pred_peakmem=mem_use,
            pred_duration=plan.get('time', 0.0),
            qtype='large'
        )
        queries.append(q)
    print(f"Total queries loaded: {len(queries)}")
    return queries

def solve_box_placement_ilp(queries: List[Query], memory_limit_kb: int):
    """
    Solve the box-placement problem exactly:
      - We have rectangles for each query q_i:
         width = T_i (duration), height = M_i (memory).
      - (x_i, y_i) = bottom-left coordinate for the rectangle i.
      - No overlap constraints: for each pair (i,j), exactly
        one of:
          x_i + T_i <= x_j, OR
          x_j + T_j <= x_i, OR
          y_i + M_i <= y_j, OR
          y_j + M_j <= y_i.
      - Also, y_i + M_i <= memory_limit_kb and x_i >= 0, y_i >= 0.
      - We define C >= x_i + T_i for each i, and minimize C.
    """
    n = len(queries)
    if n == 0:
        return 0.0

    # Create the ILP model
    prob = pulp.LpProblem("ExactBoxPlacement", pulp.LpMinimize)

    # Big Ms for time and memory
    # M_X should be large enough to exceed any feasible difference in x positions
    # M_Y should be large enough to exceed any feasible difference in y positions
    sum_of_durations = sum(q.pred_duration for q in queries)
    M_X = sum_of_durations + 10.0
    M_Y = memory_limit_kb + 10.0  # just a bit larger than the memory limit

    # Decision variables: x_i, y_i (continuous), and C (makespan)
    x_vars = [pulp.LpVariable(f"x_{i}", lowBound=0, cat=pulp.LpContinuous) for i in range(n)]
    y_vars = [pulp.LpVariable(f"y_{i}", lowBound=0, cat=pulp.LpContinuous) for i in range(n)]
    C = pulp.LpVariable("C", lowBound=0, cat=pulp.LpContinuous)

    # Objective: minimize C
    prob += C, "MinimizeMakespan"

    # 1) Each query must finish by C
    for i, q in enumerate(queries):
        prob += x_vars[i] + q.pred_duration <= C, f"finish_time_{i}"

    # 2) Memory boundary: y_i + M_i <= memory_limit
    for i, q in enumerate(queries):
        prob += y_vars[i] + q.pred_peakmem <= memory_limit_kb, f"mem_bound_{i}"

    # 3) Non-overlapping constraints
    # For each pair (i,j), we have 4 binary variables:
    # a1_{ij}, a2_{ij}, a3_{ij}, a4_{ij} in {0,1}, summing to exactly 1
    # They correspond to:
    #   a1: x_i + T_i <= x_j
    #   a2: x_j + T_j <= x_i
    #   a3: y_i + M_i <= y_j
    #   a4: y_j + M_j <= y_i
    # We'll only define them for i < j to avoid duplication
    a1 = {}
    a2 = {}
    a3 = {}
    a4 = {}

    for i in range(n):
        for j in range(i+1, n):
            # Create 4 binary variables
            a1[(i,j)] = pulp.LpVariable(f"a1_{i}_{j}", cat=pulp.LpBinary)
            a2[(i,j)] = pulp.LpVariable(f"a2_{i}_{j}", cat=pulp.LpBinary)
            a3[(i,j)] = pulp.LpVariable(f"a3_{i}_{j}", cat=pulp.LpBinary)
            a4[(i,j)] = pulp.LpVariable(f"a4_{i}_{j}", cat=pulp.LpBinary)

            # Exactly one of the four is 1
            prob += (a1[(i,j)] + a2[(i,j)] + a3[(i,j)] + a4[(i,j)] == 1), f"exactly_one_{i}_{j}"

            # If a1_{ij} = 1 => x_i + T_i <= x_j
            prob += (x_vars[i] + queries[i].pred_duration
                     <= x_vars[j] + M_X * (1 - a1[(i,j)])), f"no_overlap_a1_{i}_{j}"

            # If a2_{ij} = 1 => x_j + T_j <= x_vars[i]
            prob += (x_vars[j] + queries[j].pred_duration
                     <= x_vars[i] + M_X * (1 - a2[(i,j)])), f"no_overlap_a2_{i}_{j}"

            # If a3_{ij} = 1 => y_i + M_i <= y_j
            prob += (y_vars[i] + queries[i].pred_peakmem
                     <= y_vars[j] + M_Y * (1 - a3[(i,j)])), f"no_overlap_a3_{i}_{j}"

            # If a4_{ij} = 1 => y_j + M_j <= y_i
            prob += (y_vars[j] + queries[j].pred_peakmem
                     <= y_vars[i] + M_Y * (1 - a4[(i,j)])), f"no_overlap_a4_{i}_{j}"

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=1)
    prob.setSolver(solver)
    result_status = prob.solve()

    if pulp.LpStatus[result_status] != 'Optimal':
        print(f"Solve ended with status: {pulp.LpStatus[result_status]}")
    else:
        print("Solve status: Optimal")

    # Retrieve results
    makespan = pulp.value(C)
    for i, q in enumerate(queries):
        q.x = pulp.value(x_vars[i])
        q.y = pulp.value(y_vars[i])

    return makespan

def main():
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num_queries', type=int, default=8, help='Number of queries to execute.')
    argparser.add_argument('--dataset', type=str, default='tpcds_sf1', help='Dataset to use.')
    argparser.add_argument('--debug', action='store_true', help='Enable debug logging.')
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d]: %(message)s",
    )

    # For demonstration, set memory limit to 3GB
    total_memory_kb = 3 * 1024**2

    # 1. Load queries
    plan_file = f'/home/wuy/DB/pg_mem_data/{args.dataset}/total_plans.json'
    queries = load_queries(plan_file, total_memory_kb)

    # 2. Optionally filter or choose top queries
    # queries = queries[:args.num_queries]
    queries = sorted(queries, key=lambda x: x.pred_peakmem, reverse=True)

    final_queries = []
    large_num = args.num_queries // 2
    small_num = args.num_queries - large_num
    final_queries.extend(queries[:large_num])
    final_queries.extend(queries[-small_num:])
    queries = final_queries

    # 3. Solve via exact box placement ILP
    makespan = solve_box_placement_ilp(queries, total_memory_kb)
    print(f"\nILP Found Minimal Makespan = {makespan:.6f}")

    # Sort queries by x for output
    queries_sorted = sorted(queries, key=lambda q: q.x if q.x is not None else 0)
    for q in queries_sorted:
        finish_time = (q.x + q.pred_duration) if q.x is not None else None
        top_edge = (q.y + q.pred_peakmem) if q.y is not None else None
        print(f"Q{q.id}: x={q.x:.3f}, y={q.y:.3f}, finish={finish_time:.3f}, top={top_edge:.3f}, "
              f"mem={q.pred_peakmem}, dur={q.pred_duration}")

if __name__ == "__main__":
    main()
