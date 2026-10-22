# Pretrained Embeddings of Graph Neural Networks for miRNA-Cancer Association Prediction

## Data resources
The different dataset and KG used in this project are located in data directory. These files include:

-) The data about pathways from https://reactome.org/download/current/ReactomePathways.txt, relationships between pathways from https://reactome.org/download/current/ReactomePathwaysRelation.txt and pathway-protein relations from https://reactome.org/download/current/NCBI2Reactome.txt on 24 March 2024.

## Setup
-)conda create -n kg python=3.10 -y

-)conda activate kg

-)pip install -r requirements.txt

## Predict miRNA-disease associations

Download the data and place in `link_prediction/data/`:

- [Download data](https://drive.google.com/drive/folders/1pOUfwxnpu-0GFdcxJ8UZ_9xT-sNBEzb9?usp=drive_link)

Run the prediction from the `link_prediction` directory:

python main.py --in-feats 256 --out-feats 256 --num-heads 4 --num-layers 2 --lr 0.001 --input-size 2 --hidden-size 16 --feat-drop 0.1 --attn-drop 0.1 --epochs 200

