# PERGAT: Pretrained Embeddings of Graph Neural Networks for miRNA-Cancer Association Predictions

This repository contains the code for our paper,  
**"[PERGAT: Pretrained Embeddings of Graph Neural Networks for miRNA-Cancer Association Prediction](https://ieeexplore.ieee.org/document/10822135),"**  
published at the **IEEE International Conference on Bioinformatics & Biomedicine (BIBM) 2024**,  
held from **December 3-6, 2024, in Lisbon, Portugal**.


![Alt text](images/_miRNA_disease_prediction.png)

## Data resources
The different dataset and KG used in this project are located in data directory. These files include:

-) dbDEMC: A Database of Differentially Expressed miRNAs in Human Cancers (https://www.biosino.org/dbDEMC/index)

-) HMDD: the Human microRNA Disease Database (http://www.cuilab.cn/hmdd)

-) miR2Disease: (http://www.mir2disease.org/)

## Setup and Get Started

1. Create Conda environment:
   - `conda create --name gnn python=3.11.3`

2. Activate the Conda environment:
   - `conda activate gnn`

3. Install PyTorch:
   - `conda install pytorch torchvision torchaudio -c pytorch`

4. Install DGL:
   - `conda install -c dglteam dgl`


## get embedding
gcn_embedding % python gcn_embedding.py --in_feats 256 --out_feats 256 --num_layers 2 --num_heads 2 --batch_size 1 --lr 0.0001 --num_epochs 105

## prediction
python main.py --in-feats 256 --out-feats 256 --num-heads 8 --num-layers 2 --lr 0.001 --input-size 2 --hidden-size 16 --feat-drop 0.5 --attn-drop 0.5 --epochs 1000    

## Citation

If you find this project useful for your research, please cite it using the following BibTeX entry:

```bibtex
\bibitem{DBLP:conf/bibm/LiSM24}
Sa Li, Jonah Shader, and Tianle Ma.  
\newblock {PERGAT:} Pretrained Embeddings of Graph Neural Networks for miRNA-Cancer Association Prediction.  
\newblock In *Proceedings of the IEEE International Conference on Bioinformatics and Biomedicine (BIBM) 2024*,  
\newblock pages 5776--5785, Lisbon, Portugal, December 3-6, 2024.  
\newblock IEEE.  
\newblock DOI: \href{https://doi.org/10.1109/BIBM62325.2024.10822135}{10.1109/BIBM62325.2024.10822135}.
