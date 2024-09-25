import os
import psycopg2
from torch_geometric.loader import DataLoader
from .plan_to_graph import get_unique_data_types, connect_to_db
from .dataset import QueryPlanDataset

    
def get_loaders(logger, dataset_dir, train_dataset, test_dataset, batch_size=1, num_workers=0):


    traindataset = QueryPlanDataset(logger, dataset_dir, train_dataset, 'train')  
    valdataset = QueryPlanDataset(logger, dataset_dir, train_dataset, 'val')
    testdataset = QueryPlanDataset(logger, dataset_dir, test_dataset, 'test')  

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
