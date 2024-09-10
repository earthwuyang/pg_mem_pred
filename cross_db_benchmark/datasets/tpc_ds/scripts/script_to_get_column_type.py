import os
import re
import json

def get_column_type(table_defs):
    columns_type = {}

    single_table_defs = table_defs.split("create table")
    for single_table in single_table_defs:
        table_name = single_table.split('\n')[0].strip()
        if table_name == "":
            continue
        columns_type[table_name]={}
        columns = single_table[single_table.find('(')+1:].split(');')[0].split(',\n')
        columns = [column.strip() for column in columns]
        for column in columns:
            cs=column.split()
            column_name = cs[0]
            if column_name == 'primary':
                continue
            column_type = cs[1]
            combined_column_type = None
            if column_type == 'date':
                combined_column_type = 'date'
            elif column_type == 'time':
                combined_column_type = 'time'
            elif column_type == 'integer':
                combined_column_type = 'int'
            elif 'char' in column_type:
                combined_column_type = 'char'
            elif 'decimal' in column_type:
                combined_column_type = 'float'
            columns_type[table_name][column_name]=combined_column_type

    return columns_type


sql_ddl_path = '../schema_sql/postgres.sql'
assert os.path.exists(sql_ddl_path)

with open(sql_ddl_path, 'r') as file:
    table_defs = file.read()
    # This is a rather improvised function. It does not properly parse the sql but instead assumes that columns
    # start with a newline followed by whitespaces and table definitions start with CREATE TABLE ...
    columns_type = get_column_type(table_defs)  # {"database":{"column":type, "column2":type}, "database2":{...}, ...}

filepath = os.path.join(os.path.dirname(__file__), '../column_type.json')
with open(filepath, 'w') as f:
    json.dump(columns_type, f)
print(columns_type)