import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch_geometric.nn import ChebConv

class graph_ChebNet(torch.nn.Module):
    def __init__(self, hdim = 256, dropout = 0.5):
        super(graph_ChebNet, self).__init__()
        self.conv1 = ChebConv(48, hdim, K=2)
        self.conv2 = ChebConv(hdim, hdim, K=2)
        self.conv3 = ChebConv(hdim, hdim, K=2)
        self.dropout = dropout
    def forward(self, x, edge):
        edge_index = edge

        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.relu(self.conv2(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv3(x, edge_index)

        return x
    
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

    
class hypergrph_HGNN(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5, n_class=2):
        super(hypergrph_HGNN, self).__init__()
        self.dropout = dropout
        self.fc = nn.Linear(in_ch,n_hid)
        self.hgc1 = HGNN_conv(n_hid, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)
        self.hgc3 = HGNN_conv(n_hid, n_hid)
        self.outLayer = nn.Linear(n_hid, n_class)
    def forward(self, x, G):
        x1 = F.relu(self.fc(x))
        x1= F.dropout(x1, self.dropout, training=self.training)
        
        x2 = F.relu(self.hgc1(x1, G)+x1) 
        x2 = F.dropout(x2, self.dropout, training=self.training) 
        
        x3 = F.relu(self.hgc2(x2, G)+x2) 
        x3 = F.dropout(x3, self.dropout, training=self.training) 
        
        x4 = F.relu(self.hgc3(x3, G)+x3)
        return x4
    
class DISFusion(nn.Module):
    def __init__(self, input_dim, length, lambdinter, attention, nb_classes, dropout = 0.5):
        super().__init__()
        self.lambdinter = lambdinter
        self.attention = attention
        self.dropout = dropout
        self.w_list = nn.ModuleList([nn.Linear(input_dim, input_dim, bias=True) for _ in range(length)])
        self.y_list = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(length)])
        self.att_act1 = nn.Tanh()
        self.att_act2 = nn.Softmax(dim=-1)
        self.logistic = LogReg(input_dim, nb_classes, self.dropout)
        self.concatFC=nn.Linear(input_dim * 2, input_dim)
        
    def combine_att(self, input1, input2):
        h_list = []
        h_list.append(input1)
        h_list.append(input2)
        h_combine_list = []
        for i, h in enumerate(h_list):
            h = self.w_list[i](h)
            h = self.y_list[i](h)
            h_combine_list.append(h)
        score = torch.cat(h_combine_list, -1)
        score = self.att_act1(score)
        score = self.att_act2(score)
        score = torch.unsqueeze(score, -1)
        h = torch.stack(h_list, dim=1)
        h = score * h
        h = torch.sum(h, dim=1)
        return h
    
    def combine_concat(self, input1, input2):
        x = torch.cat([input1,input2],1) 
        x = F.relu(self.concatFC(x))
        return x #x1= F.dropout(x1, self.dropout, training=self.training)
        
    def forward(self, input1, input2):
        if self.attention:
            h_fusion = self.combine_att(input1,input2)
        else:
            h_fusion = self.combine_concat(input1,input2)
        semi = self.logistic(h_fusion).squeeze(0)
        EPS = 1e-15
        batch_size = input1.size(0)
        feature_dim = input1.size(1)
        input1 = (input1 - input1.mean(dim=0)) / (input1.std(dim=0) + EPS)
        input2 = (input2 - input2.mean(dim=0)) / (input2.std(dim=0) + EPS)
        inter_c = input1.T @ input2 / batch_size
        on_diag_inter = torch.diagonal(inter_c).add_(-1).pow_(2).sum()
        off_diag_inter = off_diagonal(inter_c).pow_(2).sum()
        loss_inter = (on_diag_inter + self.lambdinter * off_diag_inter)
        
        return loss_inter, semi

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes, dropout):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.dropout = dropout
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        seq = F.dropout(seq, self.dropout, training=self.training)
        ret = self.fc(seq)
        return F.log_softmax(ret, dim=1)
    
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()