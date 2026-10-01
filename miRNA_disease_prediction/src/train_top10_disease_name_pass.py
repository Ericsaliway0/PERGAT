import json
import os
from matplotlib import pyplot as plt
import pandas as pd
import torch
import itertools
import dgl
import numpy as np
import scipy.sparse as sp
from dgl.dataloading import GraphDataLoader
from models import LinkPredictor, GATModel, MLPPredictor, FocalLoss
from utils import (plot_scores, compute_hits_k, compute_auc, compute_f1, compute_focalloss,
                    compute_accuracy, compute_precision, compute_recall, compute_map,
                    compute_focalloss_with_symmetrical_confidence, compute_auc_with_symmetrical_confidence,
                    compute_f1_with_symmetrical_confidence, compute_accuracy_with_symmetrical_confidence,
                    compute_precision_with_symmetrical_confidence, compute_recall_with_symmetrical_confidence,
                    compute_map_with_symmetrical_confidence)
from scipy.stats import sem
from torch.optim.lr_scheduler import StepLR,ExponentialLR

import json
import networkx as nx
import dgl
import numpy as np
import torch

def load_graph_data(file_path):
    # Load data from JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Create a directed graph
    G_nx = nx.DiGraph()

    # Create a mapping for edge types to numerical values
    edge_type_mapping = {}

    # Node ID to name mapping
    node_id_to_name = {}
    node_counter = 0

    # Iterate over the data and add nodes and edges
    for item in data:
        source = item['miRNA']
        target = item['disease']
        relationship_type = item['relation']['type']

        # Add source and target nodes
        source_name = source['properties']['name']
        target_name = target['properties']['name']

        if source_name not in G_nx:
            G_nx.add_node(source_name, **source['properties'])
            node_id_to_name[node_counter] = source_name
            node_counter += 1

        if target_name not in G_nx:
            G_nx.add_node(target_name, **target['properties'])
            node_id_to_name[node_counter] = target_name
            node_counter += 1

        # Add edge with numerical type
        if relationship_type not in edge_type_mapping:
            edge_type_mapping[relationship_type] = len(edge_type_mapping)
        G_nx.add_edge(source_name, target_name, type=edge_type_mapping[relationship_type])  

    # Convert the NetworkX graph to a DGL graph
    G_dgl = dgl.from_networkx(G_nx, edge_attrs=['type'])

    # Extract node features
    node_features = torch.tensor([G_nx.nodes[node]['embedding'] for node in G_nx.nodes()], dtype=torch.float32)
    G_dgl.ndata['feat'] = node_features

    return G_dgl, node_features, node_id_to_name

def train_and_evaluate(args, G_dgl, node_features, node_id_to_name):
    u, v = G_dgl.edges()
    eids = np.arange(G_dgl.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    val_size = int(len(eids) * 0.1)
    train_size = G_dgl.number_of_edges() - test_size - val_size

    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    val_pos_u, val_pos_v = u[eids[test_size:test_size + val_size]], v[eids[test_size:test_size + val_size]]
    train_pos_u, train_pos_v = u[eids[test_size + val_size:]], v[eids[test_size + val_size:]]

    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(G_dgl.number_of_nodes(), G_dgl.number_of_nodes()))
    adj_neg = 1 - adj.todense() - np.eye(G_dgl.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), G_dgl.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    val_neg_u, val_neg_v = neg_u[neg_eids[test_size:test_size + val_size]], neg_v[neg_eids[test_size:test_size + val_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size + val_size:]], neg_v[neg_eids[test_size + val_size:]]

    train_g = dgl.remove_edges(G_dgl, eids[:test_size + val_size])

    def create_graph(u, v, num_nodes):
        assert len(u) == len(v), "Source and destination nodes must have the same length"
        return dgl.graph((u, v), num_nodes=num_nodes)

    train_pos_g = create_graph(train_pos_u, train_pos_v, G_dgl.number_of_nodes())
    train_neg_g = create_graph(train_neg_u, train_neg_v, G_dgl.number_of_nodes())
    val_pos_g = create_graph(val_pos_u, val_pos_v, G_dgl.number_of_nodes())
    val_neg_g = create_graph(val_neg_u, val_neg_v, G_dgl.number_of_nodes())
    test_pos_g = create_graph(test_pos_u, test_pos_v, G_dgl.number_of_nodes())
    test_neg_g = create_graph(test_neg_u, test_neg_v, G_dgl.number_of_nodes())

    model = GATModel(
        node_features.shape[1],
        out_feats=args.out_feats,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        feat_drop=args.feat_drop,
        attn_drop=args.attn_drop,
        do_train=True
    )

    pred = MLPPredictor(args.input_size, args.hidden_size)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')

    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.lr)

    scheduler = ExponentialLR(optimizer, gamma=0.9)

    output_path = './link_prediction_gat/results/'
    os.makedirs(output_path, exist_ok=True)

    train_f1_scores = []
    val_f1_scores = []
    train_focal_loss_scores = []
    val_focal_loss_scores = []
    train_auc_scores = []
    val_auc_scores = []
    train_map_scores = []
    val_map_scores = []
    train_recall_scores = []
    val_recall_scores = []
    train_acc_scores = []
    val_acc_scores = []
    train_precision_scores = []
    val_precision_scores = []

    for e in range(args.epochs):
        model.train()
        h = model(train_g, train_g.ndata['feat'])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)

        pos_labels = torch.ones_like(pos_score)
        neg_labels = torch.zeros_like(neg_score)

        all_scores = torch.cat([pos_score, neg_score])
        all_labels = torch.cat([pos_labels, neg_labels])

        loss = criterion(all_scores, all_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print(f'In epoch {e}, loss: {loss.item()}')

        with torch.no_grad():
            h_train = model(train_g, train_g.ndata['feat'])
            train_pos_score = pred(train_pos_g, h_train)
            train_neg_score = pred(train_neg_g, h_train)
            train_f1 = compute_f1(train_pos_score, train_neg_score)
            train_f1_scores.append(train_f1.item())
            train_focal_loss = compute_focalloss(train_pos_score, train_neg_score)
            train_focal_loss_scores.append(train_focal_loss)
            train_auc = compute_auc(train_pos_score, train_neg_score)
            train_auc_scores.append(train_auc.item())
            train_map = compute_map(train_pos_score, train_neg_score)
            train_map_scores.append(train_map.item())
            train_recall = compute_recall(train_pos_score, train_neg_score)
            train_recall_scores.append(train_recall.item())
            train_acc = compute_accuracy(train_pos_score, train_neg_score)
            train_acc_scores.append(train_acc)
            train_precision = compute_precision(train_pos_score, train_neg_score)
            train_precision_scores.append(train_precision)

            h_val = model(train_g, train_g.ndata['feat'])
            val_pos_score = pred(val_pos_g, h_val)
            val_neg_score = pred(val_neg_g, h_val)
            val_f1 = compute_f1(val_pos_score, val_neg_score)
            val_f1_scores.append(val_f1.item())
            val_focal_loss = compute_focalloss(val_pos_score, val_neg_score)
            val_focal_loss_scores.append(val_focal_loss)
            val_auc = compute_auc(val_pos_score, val_neg_score)
            val_auc_scores.append(val_auc.item())
            val_map = compute_map(val_pos_score, val_neg_score)
            val_map_scores.append(val_map.item())
            val_recall = compute_recall(val_pos_score, val_neg_score)
            val_recall_scores.append(val_recall.item())
            val_acc = compute_accuracy(val_pos_score, val_neg_score)
            val_acc_scores.append(val_acc)
            val_precision = compute_precision(val_pos_score, val_neg_score)
            val_precision_scores.append(val_precision)

    epochs = range(args.epochs)

    with torch.no_grad():
        model.eval()
        h_test = model(G_dgl, G_dgl.ndata['feat'])
        test_pos_score = pred(test_pos_g, h_test)
        test_neg_score = pred(test_neg_g, h_test)
        test_auc, test_auc_err = compute_auc_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_f1, test_f1_err = compute_f1_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_focal_loss, test_focal_loss_err = compute_focalloss_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_precision, test_precision_err = compute_precision_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_recall, test_recall_err = compute_recall_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_hits_k = compute_hits_k(test_pos_score, test_neg_score, k=10)
        test_map, test_map_err = compute_map_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_accuracy, test_accuracy_err = compute_accuracy_with_symmetrical_confidence(test_pos_score, test_neg_score)

        print(f'Test AUC: {test_auc:.4f} ± {test_auc_err:.4f} | Test F1: {test_f1:.4f} ± {test_f1_err:.4f} | Test FocalLoss: {test_focal_loss:.4f} ± {test_focal_loss_err:.4f} |Test Accuracy: {test_accuracy:.4f} ± {test_accuracy_err:.4f} | Test Precision: {test_precision:.4f} ± {test_precision_err:.4f} | Test Recall: {test_recall:.4f} ± {test_recall_err:.4f} | Test Hits@10: {test_hits_k:.4f} | Test MAP: {test_map:.4f} ± {test_map_err:.4f}')

    plot_scores(
        train_f1_scores, val_f1_scores,
        train_focal_loss_scores, val_focal_loss_scores,
        train_auc_scores, val_auc_scores,
        train_map_scores, val_map_scores,
        train_recall_scores, val_recall_scores,
        train_acc_scores, val_acc_scores,
        train_precision_scores, val_precision_scores,
        epochs, output_path, args
    )

    torch.save(model.state_dict(), os.path.join(output_path, 'gat_model.pt'))
    torch.save(pred.state_dict(), os.path.join(output_path, 'mlp_predictor.pt'))

    # Save the top 10 predictions with miRNA and disease names
    top_10_predictions = []
    for u, v, score in zip(test_pos_u, test_pos_v, test_pos_score):
        top_10_predictions.append({
            'source': node_id_to_name[u.item()],
            'destination': node_id_to_name[v.item()],
            'score': score.item()
        })

    top_10_predictions.sort(key=lambda x: x['score'], reverse=True)
    top_10_predictions = top_10_predictions[:10]

    df_top_10 = pd.DataFrame(top_10_predictions)
    df_top_10.to_csv(os.path.join(output_path, 'top_10_predictions.csv'), index=False)

import argparse
##from data_loader import load_graph_data
##from train import train_and_evaluate

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description='MLP Predictor')
    parser.add_argument('--in-feats', type=int, default=128, help='Dimension of the first layer')
    parser.add_argument('--out-feats', type=int, default=128, help='Dimension of the final layer')
    parser.add_argument('--num-heads', type=int, default=1, help='Number of heads')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for the optimizer')
    parser.add_argument('--input-size', type=int, default=2, help='Input size for the first linear layer')
    parser.add_argument('--hidden-size', type=int, default=16, help='Hidden size for the first linear layer')
    parser.add_argument('--feat-drop', type=float, default=0.0, help='Feature dropout rate')
    parser.add_argument('--attn-drop', type=float, default=0.0, help='Attention dropout rate')
    args = parser.parse_args()

    ##G_dgl, node_features = load_graph_data('data/miRNA_disease_embeddings.json')
    G_dgl, node_features, node_id_to_name = load_graph_data('data/miRNA_disease_embeddings.json')
    print('node_features.shape============\n',node_features)
    train_and_evaluate(args, G_dgl, node_features, node_id_to_name)

    





## python link_prediction_gat/main.py --in-feats 128 --out-feats 128 --num-heads 4 --num-layers 2 --lr 0.01 --input-size 2 --hidden-size 16 --feat-drop 0.1 --attn-drop 0.1 --epochs 2001
