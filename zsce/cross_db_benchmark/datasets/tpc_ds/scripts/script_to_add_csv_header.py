import os
import re


def extract_column_names(table_defs):
    column_names = dict()

    single_table_defs = table_defs.split("create table")
    for single_table in single_table_defs:
        alphanumeric_sequences = re.findall('\w+', single_table)
        if len(alphanumeric_sequences) > 0:
            table_name = alphanumeric_sequences[0]
            cols = [col.strip() for col in re.findall('\n\s+\w+', single_table)]
            if 'drop' in cols:
                cols.remove('drop')
            if 'primary' in cols:  # added by wuy
                cols.remove('primary')
            column_names[table_name] = cols

    return column_names


source_path = '/home/wuy/DB/tpcds-data-1'
target = '/home/wuy/DB/performance_estimation/zero-shot-data/datasets/tpc_ds'
os.makedirs(target, exist_ok=True)
sql_ddl_path = '../schema_sql/postgres.sql'
assert os.path.exists(sql_ddl_path)
assert os.path.exists(source_path)

with open(sql_ddl_path, 'r') as file:
    table_defs = file.read()
    # This is a rather improvised function. It does not properly parse the sql but instead assumes that columns
    # start with a newline followed by whitespaces and table definitions start with CREATE TABLE ...
    column_names = extract_column_names(table_defs)

print(column_names)
# for table in list(column_names.keys()):
for table in ["call_center","catalog_page","catalog_returns","catalog_sales","customer","customer_address","customer_demographics","date_dim","household_demographics","income_band","inventory","item","promotion","reason","ship_mode","store","store_returns","store_sales","time_dim","warehouse","web_page","web_returns","web_sales","web_site","dbgen_version"]:
    print(f"Creating headers for {table}")
    with open(os.path.join(target, f'{table}.csv'), 'w') as outfile:
        with open(os.path.join(source_path, f'{table}.dat')) as infile:
            outfile.write('|'.join(column_names[table]) + '\n')
            for line in infile:
                outfile.write(line)
