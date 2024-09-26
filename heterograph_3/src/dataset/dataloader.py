import os
import psycopg2
from torch_geometric.loader import DataLoader
from .plan_to_graph import get_unique_data_types, connect_to_db
from .dataset import QueryPlanDataset

    
def get_data_type_mapping(logger, dataset):

    # Database connection details
    DB_CONFIG = {
        'dbname': dataset,
        'user': 'wuy',
        'password': '',
        'host': 'localhost',  # e.g., 'localhost'
    }
    # Connect to PostgreSQL
    conn = connect_to_db(DB_CONFIG)
    if not conn:
        raise Exception("Failed to connect to the database")
        exit(1)  # Exit if connection failed
        
    # Fetch unique data types and create a mapping
    unique_data_types = get_unique_data_types(conn)
    conn.close()

    # To ensure consistent one-hot encoding, create a sorted list
    unique_data_types = sorted(unique_data_types)  # ['character', 'date', 'character varying', 'integer', 'numeric']
    # Add 'unknown' category at the end
    data_type_mapping = {dtype: idx for idx, dtype in enumerate(unique_data_types)}
    # logger.info(f"Unique data types: {data_type_mapping}")
    return data_type_mapping
    
def get_loaders(logger, dataset_dir, train_dataset, test_dataset, batch_size=1, num_workers=0):

    data_type_mapping = get_data_type_mapping(logger, train_dataset)

    traindataset = QueryPlanDataset(logger, dataset_dir, train_dataset, 'train', data_type_mapping)  
    valdataset = QueryPlanDataset(logger, dataset_dir, train_dataset, 'val', data_type_mapping)
    testdataset = QueryPlanDataset(logger, dataset_dir, test_dataset, 'test', data_type_mapping)  

    logger.info('Train dataset size: {}'.format(len(traindataset)))
    logger.info('Val dataset size: {}'.format(len(valdataset)))
    logger.info('Test dataset size: {}'.format(len(testdataset)))

    train_loader = DataLoader(traindataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(valdataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(testdataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    import logging
    # define a logger that outputs to stdout
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    # create a logger object

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    train_loader, val_loader, test_loader = get_loaders(logger, '/home/wuy/DB/pg_mem_data', 'tpch_sf1', 'tpch_sf1', 1, 0)
    print(train_loader.dataset[0])
    # for i,batch in enumerate(train_loader):
    #     print(batch['column'].x.shape)
    #     assert batch['column'].x.shape[1] == 10
