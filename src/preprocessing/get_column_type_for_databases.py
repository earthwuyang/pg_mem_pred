import os
import json
import re

import re

def get_column_type(table_defs):
    columns_type = {}
    # Split each table definition by "CREATE TABLE"
    single_table_defs = table_defs.split("CREATE TABLE")

    for single_table in single_table_defs:
        # Extract the table name correctly (match between CREATE TABLE and '(')
        # This regex matches table names with or without double quotes
        table_match = re.search(r'\s*(?:"(.*?)"|(\w+))\s*\(', single_table)
        if not table_match:
            continue
        # Extract the table name, giving priority to the quoted group
        table_name = table_match.group(1) if table_match.group(1) else table_match.group(2)

        columns_type[table_name] = {}
        # Extract the columns part of the table definition
        columns = single_table[single_table.find('(')+1:].split(');')[0].split(',\n')
        columns = [column.strip() for column in columns]
        
        for column in columns:
            # Skip primary key or constraints definitions
            if 'PRIMARY KEY' in column or 'FOREIGN KEY' in column:
                continue

            cs = column.split()
            column_name = cs[0].strip('"')  # Handle quoted column names
            column_type = cs[1]

            # Simplify column type detection
            combined_column_type = None
            if column_type == 'date':
                combined_column_type = 'date'
            elif column_type == 'time':
                combined_column_type = 'time'
            elif column_type == 'integer':
                combined_column_type = 'int'
            elif 'char' in column_type or 'varchar' in column_type:
                combined_column_type = 'char'
            elif 'decimal' in column_type or 'double' in column_type:
                combined_column_type = 'float'

            # Store the column name and its type
            columns_type[table_name][column_name] = combined_column_type

    return columns_type


def main(dataset):
    print(f"Getting column_type for {dataset}")
    # File paths
    sql_ddl_path = os.path.join(os.path.dirname(__file__), f'../../zsce/cross_db_benchmark/datasets/{dataset}/schema_sql/postgres.sql')
    assert os.path.exists(sql_ddl_path)

    # Read and parse the SQL DDL
    with open(sql_ddl_path, 'r') as file:
        table_defs = file.read()
        columns_type = get_column_type(table_defs)

    # Save the parsed column definitions as JSON
    filepath = os.path.join(os.path.dirname(__file__), f'../../zsce/cross_db_benchmark/datasets/{dataset}/column_type.json')
    with open(filepath, 'w') as f:
        json.dump(columns_type, f, indent=2)

    # print(f"column_type for {dataset} saved")

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from database_list import full_database_list

    for dataset in full_database_list:
        main(dataset)
