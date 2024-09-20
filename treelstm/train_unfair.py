from dataloader import PlanDataset, plan2graph
from TreeLSTM import PlanNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm
import random
from metrics import compute_metrics
import argparse
import os
import numpy as np
import logging

def train_epoch(model, trainset, optimizer, loss_fn, h_size, device):
    model.train()
    epoch_loss = 0.0
    for i, batch in enumerate(trainset):
        # logging.info(f"epoch {epoch}, iter {i}")
        g, memory, root_node_indexes = batch
        
        n=g.num_nodes()
        h = torch.zeros((n,h_size))
        c = torch.zeros(n,h_size)
        # cost = torch.FloatTensor([cost])
        # memory = torch.FloatTensor([memory])
        g = g.to(device)
        memory = memory.to(device)

        h = h.to(device)
        c = c.to(device)

        output = model(g, g.ndata['feat'], h, c, root_node_indexes)
        
        optimizer.zero_grad()
        loss = loss_fn(output, memory.unsqueeze(1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        # if i % 100 == 0:
        #     logging.info('Epoch: ', epoch, 'Iter: ', i, 'Loss: ', loss.item())

    # logging.info('Epoch ', epoch, 'Mean Loss: ', epoch_loss/len(trainset))
    return epoch_loss/len(trainset)

def validate_model(logger, model, testset, loss_fn, h_size, device ):
    with torch.no_grad():
        model.eval()

        val_epoch_loss = 0.0
        for i, batch in enumerate(testset):  
            g, memory, root_node_indexes = batch
            n=g.num_nodes()
            h = torch.zeros((n,h_size))  
            c = torch.zeros(n,h_size)
            # cost = torch.FloatTensor([cost])
            # memory = torch.FloatTensor([memory])
            g = g.to(device)
            memory = memory.to(device)
            h = h.to(device)
            c = c.to(device)

            output = model(g, g.ndata['feat'], h, c, root_node_indexes)
            loss = loss_fn(output, memory.unsqueeze(1))
            
            # qerror = Qerror(test_dataset, output, memory)
            val_epoch_loss += loss.item()
        avg_val_loss = val_epoch_loss/len(testset)
        # qerror_0, qerror_50, qerror_95, qerror_max = MedianQerror(test_dataset, output, memory)
        # logging.info('Epoch ', epoch, 'Train Loss: ', epoch_loss/len(trainset), ' Validation Loss: ', val_epoch_loss/len(testset), 'MeanQError: ', qerror.item(), 'Mean relative error: ', MRE(test_dataset, output, memory).item(), 'Qerror_0: ', qerror_0, 'QError_50: ', qerror_50, 'QError_95: ', qerror_95, 'QError_max: ', qerror_max)
        original_memory = testset.dataset.memory_scaler.inverse_transform(memory.cpu().reshape(-1,1)).reshape(-1)
        output = testset.dataset.memory_scaler.inverse_transform(output.cpu().reshape(-1,1)).reshape(-1)
        metrics = compute_metrics(original_memory, output)
        logger.info(f"Metrics: {metrics}")
        return avg_val_loss, metrics


def get_logger(logfile):

    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    fmt = f"[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d]:%(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    fh=logging.FileHandler(logfile)
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    log.addHandler(ch)
    log.addHandler(fh)
    return log

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train_dataset', type=str, default='tpch', help='dataset to use for training')
    argparser.add_argument('--test_dataset', type=str, default='tpch', help='dataset to use for testing')
    args = argparser.parse_args()

    logfile = f"train_{args.train_dataset}_test_{args.test_dataset}.log"
    logger = get_logger(logfile)

    random.seed(1)
    torch.manual_seed(1)
    train_dataset = PlanDataset(os.path.join(args.train_dataset, 'train'))
    test_dataset = PlanDataset(os.path.join(args.test_dataset, 'val'))
    # dataset = PlanDataset()

    # train_size = int(0.9 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(
    #     dataset=dataset,
    #     lengths=[train_size, test_size],
    #     generator=torch.Generator().manual_seed(1)
    # )
    logger.info(f"train_dataset len {len(train_dataset)}")
    logger.info(f"test_dataset len {len(test_dataset)}")

    batch_size = 60000
    num_epochs = 100000
    train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=plan2graph)
    test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=plan2graph)
    trainset = train_dataset
    testset = test_dataset

    x_size = 6  # this depends on how many features each node has
    h_size = 6  # this is the hidden state size of the LSTM, this is hyper-parameter 
    dropout = 0.5


    model = PlanNet(x_size, h_size, dropout)

    # model.load_state_dict(torch.load('plan_net.pth'))

    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    loss_fn = nn.MSELoss()

    logger.info(f"training begins...")

    # Early Stopping Parameters
    patience = 20
    min_delta = 0.0
    best_val_loss = float('inf')
    best_qerror = float('inf')
    epochs_no_improve = 0
    early_stop = False

    # Path to save the best model
    best_model_path = f'best_plan_net_train_{args.train_dataset}_test_{args.test_dataset}.pth'

    for epoch in range(num_epochs):
        avg_epoch_loss = train_epoch(model, trainset, optimizer, loss_fn, h_size, device)
        
        logger.info(f"Epoch {epoch}:")
        avg_val_loss, metrics = validate_model(logger, model, testset, loss_fn, h_size, device)
        logger.info(f"Epoch {epoch} Train Loss: {avg_epoch_loss} Validation Loss: {avg_val_loss}")
        
        median_qerror = metrics['qerror_50 (Median)']
        if best_qerror > median_qerror:
            best_qerror = median_qerror
            logger.info(f"Epoch {epoch} Best Qerror: {best_qerror}")
            epochs_no_improve = 0
            logger.info(f"epochs_no_improve is {epochs_no_improve}")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Epoch {epoch} state dict saved to {best_model_path}")
        else:
            epochs_no_improve += 1
            logger.info(f"epochs_no_improve is {epochs_no_improve}")
            if epochs_no_improve >= patience:
                early_stop = True
                logger.info(f"Early stopping at epoch {epoch}")
                break
        # if avg_val_loss < best_val_loss - min_delta:
        #     best_val_loss = avg_val_loss
        #     epochs_no_improve = 0
        #     logging.info(f"epochs_no_improve is {epochs_no_improve}")
        #     torch.save(model.state_dict(), best_model_path)
        #     logging.info(f"Epoch {epoch} state dict saved to {best_model_path}")
        # else:
        #     epochs_no_improve += 1
        #     logging.info(f"epochs_no_improve is {epochs_no_improve}")
        #     if epochs_no_improve >= patience:
        #         early_stop = True
        #         logging.info(f"Early stopping at epoch {epoch}")
        #         break
                
    model.load_state_dict(torch.load(best_model_path))

    validate_model(logger, model, testset, loss_fn, h_size, device)


if __name__ == '__main__':
    main()


