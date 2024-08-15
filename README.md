# DISFusion: Improving cancer gene prediction by enhancing common information between the PPI network and gene functional association
DISFusion is a cancer gene identification method that integrates cancer multi-omics data, PPI networks, and gene functional associations.
This repo is for the source code of "Improving cancer gene prediction by enhancing common information between the PPI network and gene functional association". \
Paper Link: 

Setup
------------------------
The setup process for DISFusion requires the following steps:
### Download
Download DISFusion.  The following command clones the current DISFusion repository from GitHub:

    git clone https://github.com/Spiritkkk/DISFusion.git
    
### Environment Settings
> python==3.7.0 \
> scipy==1.1.0 \
> torch==1.13.0+cu117 \
> numpy==1.15.2 \
> pandas==0.23.4 \
> scikit_learn==0.19.2

GPU: GeForce RTX 2080 11G \
CPU: Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz

### Usage
(1) After downloading and unzipping this repository, go into the folder. 

(2) We have created examples of DISFusion for predicting pan-cancer genes, namely 'main.py'.

Assuming that you are currently in the downloaded folder, just run the following command and you will be able to build a model and make predictions:

predicting pan-cancer genes
```bash
 
python main.py ./outputFile
 
 ```
 ### Input
The input of DISFusion mainly consists of two parts, one of which is the incidence matrix of the hypergraph and the other is the labeled genes. We used two annotated gene sets in our example to predict cancer genes, but this can be easily extended to other diseases.

 ### Output
The output of DISFusion is the ranking results and prediction scores of all genes.

### Files
*main.py*: Examples of DISFusion for cancer gene identification \
*models.py*: DISFusion model \
*train_pred.py*: Training and testing functions \
*utils.py*: Supporting functions

### Cite
```

```



