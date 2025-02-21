import numpy as np

# DP-based solution to the Box Placement Problem
def dp_box_placement(queries, M_total):
    """
    Solves the query box placement problem using dynamic programming.
    
    queries: A list of tuples (memory, time) representing the predicted memory usage and execution time of each query.
    M_total: The total memory available at any time.
    
    Returns the minimum makespan (the time when the last query finishes).
    """
    n = len(queries)
    
    # dp[i][j] represents the minimum makespan after scheduling the first i queries with j memory units
    dp = np.inf * np.ones((n + 1, M_total + 1))
    dp[0][0] = 0  # No queries scheduled and no memory used
    
    for i in range(1, n + 1):
        memory_needed = queries[i - 1][0]  # memory for query i
        time_needed = queries[i - 1][1]  # execution time for query i
        for j in range(M_total + 1):
            # Case 1: Do not schedule query i
            dp[i][j] = min(dp[i][j], dp[i - 1][j])
            
            # Case 2: Schedule query i if memory allows
            if j >= memory_needed:
                dp[i][j] = min(dp[i][j], max(dp[i - 1][j - memory_needed], time_needed) + time_needed)
    
    # The optimal makespan will be the minimum value for the last row in dp.
    optimal_makespan = np.min(dp[n])
    return optimal_makespan

# Example query list [(memory, time)]
queries = [(3, 2), (2, 3), (4, 4), (5, 1), (1, 3)]  # example queries with (memory, time)
M_total = 10  # total available memory

optimal_makespan = dp_box_placement(queries, M_total)
print(f"Optimal Makespan: {optimal_makespan}")
