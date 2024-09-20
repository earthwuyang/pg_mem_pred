from dataloader import PlanDataset, plan2graph
from TreeLSTM import PlanNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm
import random
from metrics import Qerror, MRE, MedianQerror
# from new_metrics import compute_metrics


random.seed(1)
torch.manual_seed(1)
dataset = PlanDataset()

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(
    dataset=dataset,
    lengths=[train_size, test_size],
    generator=torch.Generator().manual_seed(1)
)
print(f"train_dataset len {len(train_dataset)}")
print(f"test_dataset len {len(test_dataset)}")

batch_size = 60000
num_epochs = 10000
train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=plan2graph)
test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=plan2graph)
trainset = train_dataset
testset = test_dataset

x_size = 6  # this depends on how many features each node has
h_size = 6  # this is the hidden state size of the LSTM, this is hyper-parameter 
dropout = 0.5


model = PlanNet(x_size, h_size, dropout)

model.load_state_dict(torch.load('plan_net_save.pth'))

# optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

loss_fn = nn.MSELoss()


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
        qerror = Qerror(dataset, output, memory)
        val_epoch_loss += loss.item()
    qerror_0, qerror_50, qerror_95, qerror_max = MedianQerror(dataset, output, memory)
    print('Epoch ',  ' Validation Loss: ', val_epoch_loss/len(testset), 'MeanQError: ', qerror.item(), 'Mean relative error: ', MRE(dataset, output, memory).item(), 'Qerror_0: ', qerror_0, 'QError_50: ', qerror_50, 'QError_95: ', qerror_95, 'QError_max: ', qerror_max)
        # print('Epoch ', epoch, 'Train Loss: ', epoch_loss/len(trainset), ' Validation Loss: ', val_epoch_loss/len(testset))
        # original_memory = dataset.memory_scaler.inverse_transform(memory.cpu().reshape(-1,1)).reshape(-1)
        # metrics = compute_metrics(original_memory, output.cpu())
        # print(f"Metrics: {metrics}")





