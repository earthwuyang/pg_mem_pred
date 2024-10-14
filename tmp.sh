#!/bin/bash

# List of databases to drop
databases=('accidents' 'airline' 'carcinogenesis' 'credit' \
           'employee' 'financial' 'geneea' 'hepatitis' \
           'seznam' 'walmart'  'tpch_sf1' 'tpcds_sf1')

# PostgreSQL credentials
PGUSER="wuy"
PGPASSWORD="wuy"
PGHOST="localhost"
PGPORT="5432"

# Export PostgreSQL password (optional, for password authentication)
export PGPASSWORD=$PGPASSWORD

# Loop through each database and drop it
for db in "${databases[@]}"
do
    echo "Dropping database: $db"
    psql -h $PGHOST -p $PGPORT -U $PGUSER -c "DROP DATABASE IF EXISTS $db;"
done

# Unset the password variable for security
unset PGPASSWORD