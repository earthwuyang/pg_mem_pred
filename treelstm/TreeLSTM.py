import torch
import torch.nn as nn
import dgl

# child-sum Tree LSTM

class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    # def reduce_func(self, nodes):
    #     # concatenate h_jl for equation (1), (2), (3), (4)
    #     h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
    #     # equation (2)
    #     f = torch.sigmoid(self.U_f(h_cat)).view(*nodes.mailbox['h'].size())
    #     # second term of equation (5)
    #     c = torch.sum(f * nodes.mailbox['c'], 1)
    #     return {'iou': self.U_iou(h_cat), 'c': c}
    def reduce_func(self, nodes):
        h_tild = torch.sum(nodes.mailbox['h'], 1)
        f = torch.sigmoid(self.U_f(nodes.mailbox['h']))
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_tild), 'c': c}


    def apply_node_func(self, nodes):
        # equation (1), (3), (4)
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        # equation (5)
        c = i * u + nodes.data['c']
        # equation (6)
        h = o * torch.tanh(c)
        return {'h' : h, 'c' : c}
    

class TreeLSTM(nn.Module):
    def __init__(self,
                 x_size,
                 h_size,
                 dropout):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, h_size)
        self.relu = nn.ReLU()
        self.cell = TreeLSTMCell(x_size, h_size)

    def forward(self, graph, features, h, c):
        """Compute tree-lstm prediction given a batch.

        Parameters
        ----------
        batch : dgl.data.SSTBatch
            The data batch.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.

        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        # print(f"graph: {graph}, features: {features}, h: {h}, c: {c}")
        g = graph
        # to heterogenous graph
        # g = dgl.graph(g.edges())
        # print(f"g.number_of_nodes(): {g.number_of_nodes()}, g.number_of_edges(): {g.number_of_edges()}")
        # print(f"features.shape {features.shape}")
        # feed embedding
        embeds = features
        g.ndata['iou'] = self.cell.W_iou(self.dropout(embeds)) 
        g.ndata['h'] = h
        g.ndata['c'] = c
        # propagate
        dgl.prop_nodes_topo(g,
                            message_func=self.cell.message_func,
                            reduce_func=self.cell.reduce_func,
                            apply_node_func=self.cell.apply_node_func)
        # compute logits
        h = self.dropout(g.ndata.pop('h'))

        logits = self.linear(h)
        logits = self.relu(logits)
        return logits
    

class PlanNet(nn.Module):
    def __init__(self,
                 x_size,
                 h_size,
                 dropout):
        super(PlanNet, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.treelstm = TreeLSTM(x_size, h_size, dropout)
        self.fc = nn.Linear(self.h_size, 1)
        ## TODO: possible more layers here.
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, graph, features, h, c, root_node_indexes):
        # print(f"graph: {graph}, features: {features.shape}, h: {h.shape}, c: {c.shape}, cost: {cost.shape}")
        output = self.treelstm(graph, features, h, c)
        # print(f"output: {output.shape}") # [358,h_size]
        root_node_features = output[root_node_indexes]
        # print(f"root_node_features {root_node_features.shape}") # [batch_size, h_size]
        # print(f"cost: {cost.shape}, features: {features.shape}")  # cost: torch.Size([16]), features: torch.Size([385, 4])
        
        output = self.fc(root_node_features)

        # output = self.softplus(output) 
        # print(f"final output: {output}")
  
        return output