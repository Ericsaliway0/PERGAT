# PERGAT: Pretrained Embeddings of Graph Neural Networks for miRNA-Cancer Association Predictions

This repository contains the code for our paper, "Adaptive Chebyshev Graph Neural Network for Cancer Gene Prediction with Multi-Omics Integration," accepted for presentation at the IEEE International Conference on Bioinformatics & Biomedicine (BIBM) 2024, held from December 3-6, 2024, in Lisbon, Portugal.


## Data resources
The different dataset and KG used in this project are located in data directory. These files include:

-) dbDEMC: A Database of Differentially Expressed miRNAs in Human Cancers (https://www.biosino.org/dbDEMC/index)

-) HMDD: the Human microRNA Disease Database (http://www.cuilab.cn/hmdd)

-) miR2Disease: (http://www.mir2disease.org/)

## Setup

-) conda create -n gnn python=3.11 -y

-) conda activate gnn 

-) conda install pytorch::pytorch torchvision torchaudio -c pytorch

-) pip install pandas

-) pip install py2neo pandas matplotlib scikit-learn

-) pip install tqdm

-) conda install -c dglteam dgl

-) pip install seaborn

##
pip install -r requirements.txt

-) conda activate gnn 

-) conda install pytorch::pytorch torchvision torchaudio -c pytorch

-) pip install pandas

-) pip install py2neo pandas matplotlib scikit-learn

-) pip install tqdm

-) conda install -c dglteam dgl

-) pip install seaborn

## Get start
## get embedding
python PERGAT_embedding/gat_embedding.py --in_feats 256 --out_feats 256 --num_layers 2 --num_heads 2 --batch_size 1 --lr 0.0001 --num_epochs 105

PERGAT_embbedding % python gat_embedding.py --in_feats 256 --out_feats 256 --num_layers 2 --num_heads 2 --batch_size 1 --lr 0.0001 --num_epochs 107

## prediction
python main.py --in-feats 256 --out-feats 256 --num-heads 8 --num-layers 2 --lr 0.001 --input-size 2 --hidden-size 16 --feat-drop 0.5 --attn-drop 0.5 --epochs 1000    

