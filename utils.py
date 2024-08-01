from sklearn.metrics import auc, roc_auc_score
from sklearn.metrics import precision_recall_curve
import pandas as pd
import numpy as np
import torch
import random
import scipy.sparse as sp

def cal_auc(output, labels):
    outputTest = output.cpu().detach().numpy()
    outputTest = np.exp(outputTest)
    outputTest = outputTest[:,1]
    labelsTest = labels.cpu().numpy()
    AUROC = roc_auc_score(labelsTest, outputTest)
    precision, recall, _thresholds = precision_recall_curve(labelsTest, outputTest)
    AUPRC = auc(recall, precision)
    return AUROC,AUPRC

def _generate_G_from_H_weight(H, W):
    n_edge = H.shape[1]
    DV = np.sum(H * W, axis=1)  # the degree of the node
    DE = np.sum(H, axis=0)  # the degree of the hyperedge
    invDE = np.mat(np.diag(1/DE))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T
    G = DV2 * H * W * invDE * HT * DV2
    return G

def getData(positiveGenePath, negativeGenePath, geneList):
    positiveGene = pd.read_csv(positiveGenePath, header = None)
    positiveGene = list(positiveGene[0].values)
    positiveGene = list(set(geneList)&set(positiveGene))
    positiveGene.sort()
    negativeGene = pd.read_csv(negativeGenePath, header = None)     
    negativeGene = negativeGene[0]
    negativeGene = list(set(negativeGene)&set(geneList))
    negativeGene.sort()
    
    #print("positiveGene = ",len(positiveGene))
    labelFrame = pd.DataFrame(data = [0]*len(geneList), index = geneList)
    labelFrame.loc[positiveGene,:] = 1
    positiveIndex = np.where(labelFrame == 1)[0]
    labelFrame.loc[negativeGene,:] = -1
    negativeIndex = np.where(labelFrame == -1)[0]
    labelFrame = pd.DataFrame(data = [0]*len(geneList), index = geneList)
    labelFrame.loc[positiveGene,:] = 1
    
    positiveIndex = list(positiveIndex)
    negativeIndex = list(negativeIndex)
    sampleIndex = positiveIndex + negativeIndex
    sampleIndex = np.array(sampleIndex)
    label = pd.DataFrame(data = [1]*len(positiveIndex) + [0]*len(negativeIndex))
    label = label.values.ravel()
    return  sampleIndex, label, labelFrame

def processingIncidenceMatrix(geneList):
    ids = ['c2','c5']
    incidenceMatrix = pd.DataFrame(index= geneList)
    for id in ids:
        geneSetNameList = pd.read_csv('./Data/'+id+'Name.txt',sep='\t',header=None)
        geneSetNameList = list(geneSetNameList[0].values)
        z=0
        idList = list()
        for name in geneSetNameList:
            if(id=='c2'):
                q = name.split('_')
                if('CANCER' in q or 'TUMOR' in q or 'NEOPLASM' in q):
                    print(name)
                else:
                    idList.append(z)
            elif(name[:2]=='HP'):
                q = name.split('_')
                if('CANCER' in q or 'TUMOR' in q or 'NEOPLASM' in q):
                    print(name)
                else:
                    idList.append(z)
            else:
                idList.append(z)
            z=z+1
        genesetData = sp.load_npz('./Data/'+id+'_GenesetsMatrix.npz')
        incidenceMatrixTemp = pd.DataFrame(data = genesetData.A,index= geneList)
        incidenceMatrixTemp = incidenceMatrixTemp.iloc[:,idList]

        incidenceMatrix = pd.concat([incidenceMatrix,incidenceMatrixTemp],axis=1)

    incidenceMatrix.columns = np.arange(incidenceMatrix.shape[1])
    return incidenceMatrix