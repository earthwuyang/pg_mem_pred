import os
import random
import subprocess

# Path to qgen executable
QGEN_PATH = "./qgen"

# Directory to store generated queries
OUTPUT_DIR = "./random_queries"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of queries to generate
NUM_QUERIES = 10000

# Range of query templates (1 to 22 for TPC-H)
QUERY_TEMPLATE_RANGE = range(1, 23)

# Generate 10,000 random queries
for i in range(1, NUM_QUERIES + 1):
    # Randomly select a query template between 1 and 22
    query_template = random.choice(QUERY_TEMPLATE_RANGE)

    # Random seed for generating different versions of the query
    random_seed = random.randint(1, 32767)

    # Output file for the generated query
    output_file = os.path.join(OUTPUT_DIR, f"query_{i}.sql")

    # Run qgen to generate the query
    cmd = [QGEN_PATH, str(query_template), "-s", str(100), "-r", str(random_seed)]
    
    with open(output_file, "w") as outfile:
        subprocess.run(cmd, stdout=outfile)

    print(f"Generated query {i} with template {query_template} and seed {random_seed}")

print(f"Finished generating {NUM_QUERIES} queries.")
