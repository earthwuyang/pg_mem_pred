from train import validate_model
import torch
import argparse
from dataloader import PlanDataset, plan2graph
import os
import torch
import torch.nn as nn
import torch.optim as optim
from TreeLSTM import PlanNet
import logging

def get_logger():

    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    fmt = f"[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d]:%(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")


    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log

if __name__ == '__main__':
    logger = get_logger()
    parser = argparse.ArgumentParser(description='Test a TreeLSTM model')
    parser.add_argument('--dataset', type=str, default='tpch', help='dataset name')
    parser.add_argument('--model_path', type=str, default='best_plan_net_tpch.pth', help='directory of saved models')
    args = parser.parse_args()

    dataset = args.dataset
    test_dataset = PlanDataset(os.path.join(dataset, 'val'))
    batch_size = 50000
    x_size = 6  # this depends on how many features each node has
    h_size = 6  # this is the hidden state size of the LSTM, this is hyper-parameter 
    dropout = 0.5
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PlanNet(x_size, h_size, dropout)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1)
    loss_fn = nn.MSELoss()
    test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=plan2graph)
    _,metrics = validate_model(logger, model, test_dataset, loss_fn, h_size, device)
