import pandas as pd
import sys, os, random
import numpy as np
import scipy.sparse as sp
from train_pred import trainPred
from utils import processingIncidenceMatrix
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
 

if __name__ == "__main__":    
    _, outputPath = sys.argv
    lr = 1e-5
    weight_decay = 5e-5
    epochs = 200
    n_hid = 256
    lambdinter = 1e-4
    w_self = 0.005
    dropout = 0.5
    
    positiveGenePath = r'./Data/796true.txt'
    negativeGenePath = r'./Data/2187false.txt'
    geneList = pd.read_csv(r'./Data/geneList.txt', header=None)
    geneList = list(geneList.iloc[:,1].values)
    Function_hypergraph = processingIncidenceMatrix(geneList)
    edge = np.array(np.loadtxt("./Data/PPI_edge_index.txt").transpose())
    PPI_graph = torch.from_numpy(edge).long().cuda()

    feature = pd.read_csv("./Data/biological features.csv",sep=",")
    feature = torch.Tensor(feature.values).cuda()
    aurocList, auprcList, evaluationRes = trainPred(geneList, feature, Function_hypergraph, PPI_graph, positiveGenePath,
                                          negativeGenePath, lr, epochs, dropout, n_hid, weight_decay, lambdinter, w_self) 
    predRes = evaluationRes.sum(1).sort_values(ascending = False) / 25
    predRes.to_csv(outputPath,sep='\t', header = False)
    print(np.mean(aurocList)) # 0.947
    print(np.mean(auprcList)) # 0.897