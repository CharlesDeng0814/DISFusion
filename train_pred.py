import pandas as pd
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import random
from utils import cal_auc, _generate_G_from_H_weight, getData
from models import DISFusion,hypergrph_HGNN,graph_ChebNet

def train_test(trainIndex, testIndex, labelFrame, multi_feature, incidenceMatrix, PPI_grpah, geneList, lr, epochs, dropout, n_hid, weight_decay, lambdinter, w_self):
    trainFrame = labelFrame.iloc[trainIndex]
    trainPositiveGene = list(trainFrame.where(trainFrame==1).dropna().index)
    positiveMatrixSum = incidenceMatrix.loc[trainPositiveGene].sum()
        
    # disease-specific hyperedge weight
    selHyperedgeIndex = np.where(positiveMatrixSum>=3)[0]
    selHyperedge = incidenceMatrix.iloc[:, selHyperedgeIndex]
    hyperedgeWeight = positiveMatrixSum[selHyperedgeIndex].values
    selHyperedgeWeightSum = incidenceMatrix.iloc[:, selHyperedgeIndex].values.sum(0)
    hyperedgeWeight = hyperedgeWeight/selHyperedgeWeightSum
        
    H = np.array(selHyperedge).astype('float')
    DV = np.sum(H * hyperedgeWeight, axis=1)
    for i in range(DV.shape[0]):
        if(DV[i] == 0):
            t = random.randint(0, H.shape[1]-1)
            H[i][t] = 0.0001
    G = _generate_G_from_H_weight(H, hyperedgeWeight)
    N = H.shape[0]
    adj_hyperGraph = torch.Tensor(G).float()
    fh = torch.eye(N).float()
    labels = torch.from_numpy(labelFrame.values.reshape(-1,))
        
    model_hypergrph = hypergrph_HGNN(in_ch = N, n_hid = n_hid, dropout=0.2)
    model_graph = graph_ChebNet(hdim = n_hid, dropout = 0.5)
    optimizer_hypergrph = optim.Adam(model_hypergrph.parameters(), lr= 0.005, weight_decay=0.000005)
    optimizer_graph = optim.Adam(model_graph.parameters(), lr=0.001, weight_decay=0)
    schedular_hypergrph = optim.lr_scheduler.MultiStepLR(optimizer_hypergrph,milestones=[100,200,300,400],gamma=0.5)

    model_fusion = DISFusion(n_hid, 2, lambdinter, 0, 2, dropout = dropout)
    optimizer_fusion = optim.Adam(model_fusion.parameters(), lr=lr, weight_decay=weight_decay)
        
    if torch.cuda.is_available():
        model_hypergrph.cuda()
        model_graph.cuda()
        model_fusion.cuda()
        fh = fh.cuda()
        adj_hyperGraph = adj_hyperGraph.cuda()
        labels = labels.cuda()
    for epoch in range(epochs):
        model_hypergrph.train() # 先将model置为训练状态
        model_graph.train()
        optimizer_hypergrph.zero_grad() # 梯度置0
        optimizer_graph.zero_grad()
        model_fusion.train()
        optimizer_fusion.zero_grad()
        loss_self, output_fusion = model_fusion(model_hypergrph(fh, adj_hyperGraph),model_graph(multi_feature,PPI_grpah))
        loss = F.nll_loss(output_fusion[trainIndex], labels[trainIndex].long())
        loss += w_self * loss_self
        loss.backward()
        optimizer_fusion.step()
        optimizer_hypergrph.step() # 更新参数
        optimizer_graph.step()
        schedular_hypergrph.step()
    model_hypergrph.eval() # 先将model置为训练状态
    model_graph.eval()
    model_fusion.eval()
    with torch.no_grad():
        loss_self, output = model_fusion(model_hypergrph(fh, adj_hyperGraph),model_graph(multi_feature,PPI_grpah))
        loss_test = F.nll_loss(output[testIndex], labels[testIndex])
        AUROC_val, AUPRC_val = cal_auc(output[testIndex], labels[testIndex])
        outputFrame = pd.DataFrame(data = output.exp().cpu().detach().numpy(), index = geneList)
    return AUROC_val, AUPRC_val, outputFrame


def trainPred(geneList, multi_feature, Function_hypergraph, PPI_grpah, positiveGenePath,
              negativeGenePath, lr, epochs, dropout, n_hid, weight_decay, lambdinter, w_self):
    aurocList = list()
    auprcList = list()
    evaluationRes = pd.DataFrame(index = geneList)
    for i in range(5):
        sampleIndex,label,labelFrame = getData(positiveGenePath, negativeGenePath, geneList)
        sk_X = sampleIndex.reshape([-1,1])
        sfolder = StratifiedKFold(n_splits = 5, random_state = i, shuffle = True)
        for train_index,test_index in sfolder.split(sk_X, label):
            trainIndex, testIndex, _, __ = sampleIndex[train_index], sampleIndex[test_index], label[train_index], label[test_index]
            
            AUROC_val, AUPRC_val, outputFrame = train_test(trainIndex, testIndex, labelFrame, multi_feature, Function_hypergraph, PPI_grpah, geneList, lr, epochs, dropout, n_hid, weight_decay, lambdinter, w_self)
            aurocList.append(AUROC_val.item())
            auprcList.append(AUPRC_val.item())
            evaluationRes = pd.concat([evaluationRes,outputFrame[1]], axis = 1)
    return aurocList, auprcList, evaluationRes