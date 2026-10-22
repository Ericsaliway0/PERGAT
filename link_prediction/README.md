# Pretrained Embeddings of Graph Neural Networks for miRNA-Cancer Association Prediction

## Data resources
The different dataset and KG used in this project are located in data directory. These files include:

-) The data about pathways from https://reactome.org/download/current/ReactomePathways.txt, relationships between pathways from https://reactome.org/download/current/ReactomePathwaysRelation.txt and pathway-protein relations from https://reactome.org/download/current/NCBI2Reactome.txt on 24 March 2024.

-) The built knowledge graph including pathway-pathway and pathway-protein relationships.


## Scripts
The code directory contains the following scripts:

-)The script for processing data download from Reactome

-)The script for building KG and save to Neo4j Aura.


## Setup
-)conda create -n kg python=3.10 -y

-)conda activate kg

-)pip install -r requirements.txt


## Get start
python main.py --in-feats 256 --out-feats 256 --num-heads 4 --num-layers 2 --lr 0.001 --input-size 2 --hidden-size 16 --feat-drop 0.1 --attn-drop 0.1 --epochs 200

