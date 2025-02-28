import copy
import json
import os
import csv
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from src.dataset import Dataset
from src.model import GCNModel
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from py2neo import Graph, Node, Relationship
from neo4j import GraphDatabase
import torch
import torch.nn as nn
from src.utils import create_graphs
from src.plot import (populate_miRNA_dic, 
                    populate_disease_dic, 
                    create_heatmap, 
                    miRNA_plot_cosine_similarity_matrix_for_clusters_with_values, 
                    disease_plot_cosine_similarity_matrix_for_clusters_with_values,
                    visualize_embeddings_tsne,visualize_embeddings_pca,
                    calculate_cluster_labels,draw_loss_plot,
                    draw_accuracy_plot,draw_f1_plot)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure the input and target have the same shape
        if inputs.dim() > targets.dim():
            inputs = inputs.squeeze(dim=-1)
        elif targets.dim() > inputs.dim():
            targets = targets.squeeze(dim=-1)

        # Check if the shapes match after squeezing
        if inputs.size() != targets.size():
            raise ValueError(f"Target size ({targets.size()}) must be the same as input size ({inputs.size()})")

        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

def save_embeddings_to_json(file_path, miRNA_data, disease_data):
    data = []
    for name, embedding in miRNA_data:
        data.append({
            "miRNA": {
                "properties": {
                    "name": name,
                    "embedding": embedding.tolist()
                }
            },
            "relation": {
                "type": "ASSOCIATED_WITH"
            },
            "disease": {
                "properties": {
                    "name": "",
                    "embedding": []
                }
            }
        })

    for name, embedding in disease_data:
        data.append({
            "miRNA": {
                "properties": {
                    "name": "",
                    "embedding": []
                }
            },
            "relation": {
                "type": "ASSOCIATED_WITH"
            },
            "disease": {
                "properties": {
                    "name": name,
                    "embedding": embedding.tolist()
                }
            }
        })

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def train(hyperparams=None, data_path='gcn/data/emb', plot=True):
    num_epochs = hyperparams['num_epochs']
    feat_drop = hyperparams['feat_drop']
    in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']
    learning_rate = hyperparams['lr']
    batch_size = hyperparams['batch_size']
    device = hyperparams['device']
    model_path = os.path.join(data_path, 'models')
    model_path = os.path.join(model_path, f'model_dim{out_feats}_lay{num_layers}_epo{num_epochs}.pth')
    
    ds = Dataset(data_path)

    ds_train = [ds[1]]
    ds_valid = [ds[0]]

    ## convert to dgl graph
    dl_train = GraphDataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = GraphDataLoader(ds_valid, batch_size=batch_size, shuffle=False)

    net = GCNModel(out_feats, num_layers, do_train=True).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    best_model = GCNModel(out_feats, num_layers, do_train=True)
    best_model.load_state_dict(copy.deepcopy(net.state_dict()))

    loss_per_epoch_train, loss_per_epoch_valid = [], []
    f1_per_epoch_train, f1_per_epoch_valid = [], []

    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    ##criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    weight = torch.tensor([0.00001, 0.99999]).to(device)

    best_train_loss, best_valid_loss = float('inf'), float('inf')
    best_f1_score = 0.0

    max_f1_scores_train = []
    max_f1_scores_valid = []
    
    results_path = 'gcn/results/node_embeddings/'
    os.makedirs(results_path, exist_ok=True)
    
    ##all_miRNA_embeddings, miRNA_cluster_labels, miRNA_names, all_disease_embeddings, disease_cluster_labels, disease_names = (net, dl_train, device, 4)
    all_embeddings_initial_miRNA, cluster_labels_initial_miRNA, graph_name_initial_miRNA, all_embeddings_initial_disease, cluster_labels_initial_disease, graph_name_initial_disease = calculate_cluster_labels(best_model, dl_train, device)

    ##all_embeddings_initial_miRNA, cluster_labels_initial_miRNA, graph_name_initial_miRNA = calculate_cluster_labels_miRNA(best_model, dl_train, device)
    ##print('cluster_labels_initial--------------------------\n',cluster_labels_initial)
    all_embeddings_initial_miRNA = all_embeddings_initial_miRNA.reshape(all_embeddings_initial_miRNA.shape[0], -1)  # Flatten 
    save_path_heatmap_initial_miRNA = os.path.join(results_path, f'heatmap_miRNA_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_matrix_initial_miRNA = os.path.join(results_path, f'matrix_miRNA_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_pca_initial_miRNA = os.path.join(results_path, f'pca_miRNA_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_t_SNE_initial_miRNA = os.path.join(results_path, f't-SNE_miRNA_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')

    ##all_embeddings_initial_disease, cluster_labels_initial_disease, graph_name_initial_disease = calculate_cluster_labels_disease(best_model, dl_train, device)
    ##print('cluster_labels_initial--------------------------\n',cluster_labels_initial)
    all_embeddings_initial_disease = all_embeddings_initial_disease.reshape(all_embeddings_initial_disease.shape[0], -1)  # Flatten 
    save_path_heatmap_initial_disease = os.path.join(results_path, f'heatmap_disease_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_matrix_initial_disease = os.path.join(results_path, f'matrix_disease_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_pca_initial_disease = os.path.join(results_path, f'pca_disease_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_t_SNE_initial_disease = os.path.join(results_path, f't-SNE_disease_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
                
    for data in dl_train:
        graph, _ = data
        # Get all node embeddings from the graph
        node_embeddings_initial = best_model.get_node_embeddings(graph).detach().cpu().numpy()

        # Extract the 'node_type' attribute from the graph
        node_types = graph.ndata['node_type'].cpu().numpy()
        # Create a mask for miRNA nodes (assuming 'miRNA' is represented as a string in node_type)
        miRNA_mask = (node_types == 0)

        # Filter the embeddings using the miRNA mask
        miRNA_node_embeddings = node_embeddings_initial[miRNA_mask]
        disease_mask = (node_types == 1)

        # Filter the embeddings using the disease mask
        disease_node_embeddings = node_embeddings_initial[disease_mask]
                
        ##graph_path = os.path.join(data_path, f'raw/{graph_name_initial_miRNA}.pkl')
        graph_path = os.path.join(data_path, 'raw/emb_train.pkl')
        nx_graph = pickle.load(open(graph_path, 'rb'))
        
        # Filter and get all miRNA nodes
        miRNA_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node].get('node_type') == 'miRNA']
        print('cluster_labels_miRNA=====================\n',len(cluster_labels_initial_miRNA) )
        print('len(miRNA_nodes)=====================\n',len(miRNA_nodes))
        ##assert len(cluster_labels_initial_miRNA) == len(miRNA_nodes)+139, "Cluster labels and number of nodes must match"
        node_to_index_initial_miRNA = {node: idx for idx, node in enumerate(miRNA_nodes)}

        disease_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node].get('node_type') == 'disease']

        print('cluster_labels_disease=====================\n',len(cluster_labels_initial_disease))
        print('len(disease_nodes)=====================\n',len(disease_nodes))    
        ##assert len(cluster_labels_initial_disease) == len(disease_nodes)+3, "Cluster labels and number of nodes must match"
        node_to_index_initial_disease = {node: idx for idx, node in enumerate(disease_nodes)}
                
        miRNA_dic_initial, first_node_miRNA_in_cluster_initial, first_node_embedding_in_cluster_initial_miRNA = populate_miRNA_dic(nx_graph, miRNA_node_embeddings, node_to_index_initial_miRNA, cluster_labels_initial_miRNA)
        disease_dic_initial, first_node_disease_in_cluster_initial, first_node_embedding_in_cluster_initial_disease = populate_disease_dic(nx_graph, disease_node_embeddings, node_to_index_initial_disease, cluster_labels_initial_disease)

        ##print('first_node_miRNA_in_cluster_initial-------------------------------\n', first_node_miRNA_in_cluster_initial)
        miRNA_list = list(first_node_miRNA_in_cluster_initial.values())
        miRNA_embedding_list_initial = list(first_node_embedding_in_cluster_initial_miRNA.values())
        
        disease_list = list(first_node_disease_in_cluster_initial.values())
        disease_embedding_list_initial = list(first_node_embedding_in_cluster_initial_disease.values())

        disease_plot_cosine_similarity_matrix_for_clusters_with_values(disease_embedding_list_initial, disease_list, save_path_matrix_initial_disease)
        miRNA_plot_cosine_similarity_matrix_for_clusters_with_values(miRNA_embedding_list_initial, miRNA_list, save_path_matrix_initial_miRNA)
        
        create_heatmap(disease_embedding_list_initial, disease_list, save_path_heatmap_initial_disease)
        create_heatmap(miRNA_embedding_list_initial, miRNA_list, save_path_heatmap_initial_miRNA)

        
        break

    visualize_embeddings_tsne(all_embeddings_initial_miRNA, cluster_labels_initial_miRNA, miRNA_list, save_path_t_SNE_initial_miRNA)
    visualize_embeddings_pca(all_embeddings_initial_miRNA, cluster_labels_initial_miRNA, miRNA_list, save_path_pca_initial_miRNA)
    silhouette_avg_ = silhouette_score(all_embeddings_initial_miRNA, cluster_labels_initial_miRNA)
    davies_bouldin_ = davies_bouldin_score(all_embeddings_initial_miRNA, cluster_labels_initial_miRNA)
    summary_  = f"Silhouette Score: {silhouette_avg_}\n"
    summary_ += f"Davies-Bouldin Index: {davies_bouldin_}\n"

    save_file_= os.path.join(results_path, f'miRNA_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.txt')
    with open(save_file_, 'w') as f:
        f.write(summary_)


    visualize_embeddings_tsne(all_embeddings_initial_disease, cluster_labels_initial_disease, disease_list, save_path_t_SNE_initial_disease)
    visualize_embeddings_pca(all_embeddings_initial_disease, cluster_labels_initial_disease, disease_list, save_path_pca_initial_disease)
    silhouette_avg_ = silhouette_score(all_embeddings_initial_disease, cluster_labels_initial_disease)
    davies_bouldin_ = davies_bouldin_score(all_embeddings_initial_disease, cluster_labels_initial_disease)
    summary_  = f"Silhouette Score: {silhouette_avg_}\n"
    summary_ += f"Davies-Bouldin Index: {davies_bouldin_}\n"

    save_file_= os.path.join(results_path, f'disease_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.txt')
    with open(save_file_, 'w') as f:
        f.write(summary_)    
            
    loss_per_epoch_train, loss_per_epoch_valid = [], []
    f1_per_epoch_train, f1_per_epoch_valid = [], []    
    accuracy_per_epoch_train, accuracy_per_epoch_valid = [], []
    
    # Start training  
    with tqdm(total=num_epochs, desc="Training", unit="epoch", leave=False) as pbar:
        for epoch in range(num_epochs):
            loss_per_graph = []
            f1_per_graph = []
            accuracy_per_graph = []
            net.train()
            for data in dl_train:
                graph, name = data
                name = name[0]
                logits = net(graph)
                labels = graph.ndata['significance'].unsqueeze(-1)
                weight_ = weight[labels.data.view(-1).long()].view_as(labels)

                loss = criterion(logits, labels)
                loss_weighted = loss * weight_
                loss_weighted = loss_weighted.mean()

                # Update parameters
                optimizer.zero_grad()
                loss_weighted.backward()
                optimizer.step()
                
                # Append output metrics
                loss_per_graph.append(loss_weighted.item())
                preds = (logits.sigmoid() > 0.5).int()
                labels = labels.squeeze(1).int()
                f1 = metrics.f1_score(labels, preds)
                accuracy = metrics.accuracy_score(labels, preds)
                f1_per_graph.append(f1)
                accuracy_per_graph.append(accuracy)

            running_loss = np.array(loss_per_graph).mean()
            running_f1 = np.array(f1_per_graph).mean()
            running_accuracy = np.array(accuracy_per_graph).mean()
            loss_per_epoch_train.append(running_loss)
            f1_per_epoch_train.append(running_f1)
            accuracy_per_epoch_train.append(running_accuracy)

            # Validation iteration
            with torch.no_grad():
                loss_per_graph = []
                f1_per_graph_val = []
                accuracy_per_graph_val = []
                net.eval()
                for data in dl_valid:
                    graph, name = data
                    name = name[0]
                    logits = net(graph)
                    labels = graph.ndata['significance'].unsqueeze(-1)
                    weight_ = weight[labels.data.view(-1).long()].view_as(labels)
                    loss = criterion(logits, labels)
                    loss_weighted = loss * weight_
                    loss_weighted = loss_weighted.mean()
                    loss_per_graph.append(loss_weighted.item())
                    preds = (logits.sigmoid() > 0.5).int()
                    labels = labels.squeeze(1).int()
                    f1 = metrics.f1_score(labels, preds)
                    accuracy = metrics.accuracy_score(labels, preds)
                    f1_per_graph_val.append(f1)
                    accuracy_per_graph_val.append(accuracy)

                running_loss = np.array(loss_per_graph).mean()
                running_f1_val = np.array(f1_per_graph_val).mean()
                running_accuracy_val = np.array(accuracy_per_graph_val).mean()
                loss_per_epoch_valid.append(running_loss)
                f1_per_epoch_valid.append(running_f1_val)
                accuracy_per_epoch_valid.append(running_accuracy_val)

                max_f1_train = max(f1_per_epoch_train)
                max_f1_valid = max(f1_per_epoch_valid)
                max_f1_scores_train.append(max_f1_train)
                max_f1_scores_valid.append(max_f1_valid)
                
                if running_loss < best_valid_loss:
                    best_train_loss = running_loss
                    best_valid_loss = running_loss
                    best_f1_score = running_f1
                    best_model.load_state_dict(copy.deepcopy(net.state_dict()))
                    print(f"Best F1 Score: {best_f1_score}")

            pbar.update(1)
            print(f"Epoch {epoch + 1} - F1 Train: {running_f1}, F1 Valid: {running_f1_val}, Accuracy Train: {running_accuracy}, Accuracy Valid: {running_accuracy_val}")

    ##all_embeddings, cluster_labels, miRNA_names, all_disease_embeddings, disease_cluster_labels, disease_names = calculate_cluster_labels(best_model, dl_train, device)
    all_embeddings_miRNA, cluster_labels_miRNA, graph_name_initial_miRNA, all_embeddings_disease, cluster_labels_disease, graph_name_disease = calculate_cluster_labels(best_model, dl_train, device)
    
    all_embeddings_miRNA = all_embeddings_miRNA.reshape(all_embeddings_miRNA.shape[0], -1)  # Flatten 
    save_path_heatmap_miRNA = os.path.join(results_path, f'heatmap_miRNA_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_matrix_miRNA = os.path.join(results_path, f'matrix_miRNA_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_pca_miRNA = os.path.join(results_path, f'pca_miRNA_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_t_SNE_miRNA = os.path.join(results_path, f't-SNE_miRNA_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')

    ##all_embeddings_disease, cluster_labels_disease, graph_name_disease = calculate_cluster_labels_disease(best_model, dl_train, device)
    ##print('cluster_labels--------------------------\n',cluster_labels)
    all_embeddings_disease = all_embeddings_disease.reshape(all_embeddings_disease.shape[0], -1)  # Flatten 
    save_path_heatmap_disease = os.path.join(results_path, f'heatmap_disease_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_matrix_disease = os.path.join(results_path, f'matrix_disease_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_pca_disease = os.path.join(results_path, f'pca_disease_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
    save_path_t_SNE_disease = os.path.join(results_path, f't-SNE_disease_dim{out_feats}_lay{num_layers}_epo{num_epochs}_.png')
                
                    
    all_embeddings_miRNA = all_embeddings_miRNA.reshape(all_embeddings_miRNA.shape[0], -1)  # Flatten 
    print('cluster_labels=========================\n', cluster_labels_miRNA)

    cos_sim = np.dot(all_embeddings_miRNA, all_embeddings_miRNA.T)
    norms = np.linalg.norm(all_embeddings_miRNA, axis=1)
    cos_sim /= np.outer(norms, norms)
    
    if plot:
        loss_path = os.path.join(results_path, f'loss_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
        f1_path = os.path.join(results_path, f'f1_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
        accuracy_path = os.path.join(results_path, f'accuracy_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')

        draw_loss_plot(loss_per_epoch_train, loss_per_epoch_valid, loss_path)
        draw_f1_plot(f1_per_epoch_train, f1_per_epoch_valid, f1_path)
        draw_accuracy_plot(accuracy_per_epoch_train, accuracy_per_epoch_valid, accuracy_path)
        
    torch.save(best_model.state_dict(), model_path)

    for data in dl_train:
        graph, _ = data
        # Get all node embeddings from the graph
        node_embeddings = best_model.get_node_embeddings(graph).detach().cpu().numpy()

        # Extract the 'node_type' attribute from the graph
        node_types = graph.ndata['node_type'].cpu().numpy()
        
        print('node_types=====================\n',node_types)

        # Create a mask for miRNA nodes (assuming 'miRNA' is represented as a string in node_type)
        miRNA_mask = (node_types == 0)

        # Filter the embeddings using the miRNA mask
        miRNA_node_embeddings = node_embeddings[miRNA_mask]

        print(f"Shape of miRNA node embeddings: {miRNA_node_embeddings.shape}")

        disease_mask = (node_types == 1)

        # Filter the embeddings using the disease mask
        disease_node_embeddings = node_embeddings[disease_mask]

        print(f"Shape of disease node embeddings: {disease_node_embeddings.shape}")
                
        ##graph_path = os.path.join(data_path, f'raw/{graph_name_miRNA}.pkl')
        graph_path = os.path.join(data_path, 'raw/emb_train.pkl')
        nx_graph = pickle.load(open(graph_path, 'rb'))
        
        # Filter and get all miRNA nodes
        miRNA_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node].get('node_type') == 'miRNA']
    
        print('cluster_labels_miRNA=====================\n',len(cluster_labels_miRNA))
        print('len(miRNA_nodes)=====================\n',len(miRNA_nodes))

        ##assert len(cluster_labels_miRNA) == len(miRNA_nodes)+139, "Cluster labels and number of nodes must match"
        node_to_index_miRNA = {node: idx for idx, node in enumerate(miRNA_nodes)}

        disease_nodes = [node for node in nx_graph.nodes if nx_graph.nodes[node].get('node_type') == 'disease']

        print('cluster_labels_disease=====================\n',len(cluster_labels_disease))
        print('len(disease_nodes)=====================\n',len(disease_nodes))
               
        ##assert len(cluster_labels_disease) == len(disease_nodes)+3, "Cluster labels and number of nodes must match"
        node_to_index_disease = {node: idx for idx, node in enumerate(disease_nodes)}
                
        miRNA_dic, first_node_miRNA_in_cluster, first_node_embedding_in_cluster_miRNA = populate_miRNA_dic(nx_graph, miRNA_node_embeddings, node_to_index_miRNA, cluster_labels_miRNA)
        disease_dic, first_node_disease_in_cluster, first_node_embedding_in_cluster_disease = populate_disease_dic(nx_graph, disease_node_embeddings, node_to_index_disease, cluster_labels_disease)

        print('first_node_miRNA_in_cluster=====================\n',first_node_miRNA_in_cluster)
        print('first_node_disease_in_cluster=====================\n',first_node_disease_in_cluster)
                
        ##print('first_node_miRNA_in_cluster-------------------------------\n', first_node_miRNA_in_cluster)
        miRNA_list = list(first_node_miRNA_in_cluster.values())
        miRNA_embedding_list = list(first_node_embedding_in_cluster_miRNA.values())
        
        ##print('miRNA_embedding_list=====================\n',miRNA_embedding_list)
        print('miRNA_list=====================\n',miRNA_list)

        disease_list = list(first_node_disease_in_cluster.values())
        disease_embedding_list = list(first_node_embedding_in_cluster_disease.values())

        ##print('disease_embedding_list=====================\n',disease_embedding_list)
        print('disease_list=====================\n',disease_list)

        
        

        disease_plot_cosine_similarity_matrix_for_clusters_with_values(disease_embedding_list, disease_list, save_path_matrix_disease)
        miRNA_plot_cosine_similarity_matrix_for_clusters_with_values(miRNA_embedding_list, miRNA_list, save_path_matrix_miRNA)
        
        create_heatmap(disease_embedding_list, disease_list, save_path_heatmap_disease)
        create_heatmap(miRNA_embedding_list, miRNA_list, save_path_heatmap_miRNA)

        
        break

    visualize_embeddings_tsne(all_embeddings_miRNA, cluster_labels_miRNA, miRNA_list, save_path_t_SNE_miRNA)
    visualize_embeddings_pca(all_embeddings_miRNA, cluster_labels_miRNA, miRNA_list, save_path_pca_miRNA)
    silhouette_avg_miRNA = silhouette_score(all_embeddings_miRNA, cluster_labels_miRNA)
    davies_bouldin_miRNA = davies_bouldin_score(all_embeddings_miRNA, cluster_labels_miRNA)
    summary_  = f"Silhouette Score: {silhouette_avg_miRNA}\n"
    summary_ += f"Davies-Bouldin Index: {davies_bouldin_miRNA}\n"

    save_file_= os.path.join(results_path, f'miRNA_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.txt')
    with open(save_file_, 'w') as f:
        f.write(summary_)


    visualize_embeddings_tsne(all_embeddings_disease, cluster_labels_disease, disease_list, save_path_t_SNE_disease)
    visualize_embeddings_pca(all_embeddings_disease, cluster_labels_disease, disease_list, save_path_pca_disease)
    silhouette_avg_disease = silhouette_score(all_embeddings_disease, cluster_labels_disease)
    davies_bouldin_disease = davies_bouldin_score(all_embeddings_disease, cluster_labels_disease)
    summary_  = f"Silhouette Score: {silhouette_avg_}\n"
    summary_ += f"Davies-Bouldin Index: {davies_bouldin_}\n"

    save_file_= os.path.join(results_path, f'disease_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.txt')
    with open(save_file_, 'w') as f:
        f.write(summary_)    

    print(f"Silhouette Score%%%%%%%%%%%%###########################: {silhouette_avg_disease}")
    print(f"Davies-Bouldin Index: {davies_bouldin_disease}")

    summary = f"Epoch {num_epochs} - Max F1 Train: {max_f1_train}, Max F1 Valid: {max_f1_valid}\n"
    summary += f"Best Train Loss: {best_train_loss}\n"
    summary += f"Best Validation Loss: {best_valid_loss}\n"
    summary += f"Best F1 Score: {max_f1_train}\n"
    summary += f"Silhouette Score: {silhouette_avg_disease}\n"
    summary += f"Davies-Bouldin Index: {davies_bouldin_disease}\n"

    save_file = os.path.join(results_path, f'head{num_heads}_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.txt')
    with open(save_file, 'w') as f:
        f.write(summary)


    ## save_to_neo4j(graph_train, miRNA_dic, miRNA_mapping, pathway_map, gene_id_to_name_mapping, gene_id_to_symbol_mapping, neo4j_uri, neo4j_user, neo4j_password)
    
    miRNA_embeddings_initial = pd.DataFrame.from_dict(miRNA_dic_initial, orient='index')
    miRNA_embeddings_initial.to_csv('gcn/data/miRNA_embeddings_initial.csv', index_label='miRNA')

    miRNA_embeddings = pd.DataFrame.from_dict(miRNA_dic, orient='index')
    miRNA_embeddings.to_csv('gcn/data/pretrain_miRNA_embeddings.csv', index_label='miRNA')
    
    disease_embeddings_initial = pd.DataFrame.from_dict(disease_dic_initial, orient='index')
    disease_embeddings_initial.to_csv('gcn/data/disease_embeddings_initial.csv', index_label='disease')

    disease_embeddings = pd.DataFrame.from_dict(disease_dic, orient='index')
    disease_embeddings.to_csv('gcn/data/pretrain_disease_embeddings.csv', index_label='disease')
    
    return model_path

# Function to save embeddings to a CSV file
def save_embeddings_to_csv(embeddings, names, file_path):
    df = pd.DataFrame({
        'name': names,
        'embedding': [embedding.tolist() for embedding in embeddings]
    })
    df.to_csv(file_path, index=False)
    
if __name__ == '__main__':
    hyperparams = {
        'num_epochs': 100,
        'out_feats': 128,
        'num_layers': 2,
        'lr': 0.001,
        'batch_size': 1,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    train(hyperparams=hyperparams)
