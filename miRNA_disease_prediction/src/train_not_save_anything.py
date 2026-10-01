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
from torch.optim.lr_scheduler import StepLR, ExponentialLR

import json
import networkx as nx
import dgl
import numpy as np
import torch

def ori_load_graph_data(file_path):
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

import json
import networkx as nx
import dgl
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

    # Check if all edges have the 'type' attribute and add default value if missing
    for u, v, data in G_nx.edges(data=True):
        if 'type' not in data:
            data['type'] = -1  # Assign a default value for missing 'type'

    # Convert the NetworkX graph to a DGL graph
    G_dgl = dgl.from_networkx(G_nx, edge_attrs=['type'])

    # Extract node features, ensuring 'embedding' exists for each node
    node_features = []
    for node in G_nx.nodes():
        if 'embedding' in G_nx.nodes[node]:
            node_features.append(G_nx.nodes[node]['embedding'])
        else:
            # Handle missing 'embedding' (use zero vector or other default value)
            node_features.append([0] * 128)  # Assuming embedding size is 128

    node_features = torch.tensor(node_features, dtype=torch.float32)
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

    for epoch in range(args.epochs):
        model.train()
        pred.train()

        h = model(train_g, node_features)
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)

        pos_labels = torch.ones(pos_score.shape)
        neg_labels = torch.zeros(neg_score.shape)

        score = torch.cat([pos_score, neg_score])
        labels = torch.cat([pos_labels, neg_labels])

        loss = criterion(score, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            model.eval()
            pred.eval()

            def get_scores(pos_g, neg_g, h):
                pos_score = pred(pos_g, h)
                neg_score = pred(neg_g, h)

                pos_labels = torch.ones(pos_score.shape)
                neg_labels = torch.zeros(neg_score.shape)

                score = torch.cat([pos_score, neg_score])
                labels = torch.cat([pos_labels, neg_labels])

                return score, labels

            train_score, train_labels = get_scores(train_pos_g, train_neg_g, h)
            val_score, val_labels = get_scores(val_pos_g, val_neg_g, h)

            train_loss = criterion(train_score, train_labels)
            val_loss = criterion(val_score, val_labels)

            train_f1 = compute_f1(train_score, train_labels)
            val_f1 = compute_f1(val_score, val_labels)
            train_f1_scores.append(train_f1)
            val_f1_scores.append(val_f1)

            train_focal_loss = compute_focalloss(train_score, train_labels)
            val_focal_loss = compute_focalloss(val_score, val_labels)
            train_focal_loss_scores.append(train_focal_loss)
            val_focal_loss_scores.append(val_focal_loss)

            train_auc = compute_auc(train_score, train_labels)
            val_auc = compute_auc(val_score, val_labels)
            train_auc_scores.append(train_auc)
            val_auc_scores.append(val_auc)

            train_map = compute_map(train_score, train_labels)
            val_map = compute_map(val_score, val_labels)
            train_map_scores.append(train_map)
            val_map_scores.append(val_map)

            train_recall = compute_recall(train_score, train_labels)
            val_recall = compute_recall(val_score, val_labels)
            train_recall_scores.append(train_recall)
            val_recall_scores.append(val_recall)

            train_acc = compute_accuracy(train_score, train_labels)
            val_acc = compute_accuracy(val_score, val_labels)
            train_acc_scores.append(train_acc)
            val_acc_scores.append(val_acc)

            train_precision = compute_precision(train_score, train_labels)
            val_precision = compute_precision(val_score, val_labels)
            train_precision_scores.append(train_precision)
            val_precision_scores.append(val_precision)

            print(f"Epoch {epoch + 1}/{args.epochs}, "
                f"Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}, "
                f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, "
                f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, "
                f"Train MAP: {train_map:.4f}, Val MAP: {val_map:.4f}, "
                f"Train Recall: {train_recall:.4f}, Val Recall: {val_recall:.4f}, "
                f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
                f"Train Precision: {train_precision:.4f}, Val Precision: {val_precision:.4f}")

    with torch.no_grad():
        model.eval()
        pred.eval()

        h = model(G_dgl, node_features)
        test_pos_score = pred(test_pos_g, h)
        test_neg_score = pred(test_neg_g, h)

        test_pos_labels = torch.ones(test_pos_score.shape)
        test_neg_labels = torch.zeros(test_neg_score.shape)

        test_score = torch.cat([test_pos_score, test_neg_score])
        test_labels = torch.cat([test_pos_labels, test_neg_labels])

        test_loss = criterion(test_score, test_labels)
        test_f1 = compute_f1(test_score, test_labels)
        test_auc = compute_auc(test_score, test_labels)
        test_map = compute_map(test_score, test_labels)
        test_recall = compute_recall(test_score, test_labels)
        test_acc = compute_accuracy(test_score, test_labels)
        test_precision = compute_precision(test_score, test_labels)

        print(f"Test Loss: {test_loss.item():.4f}, Test F1: {test_f1:.4f}, "
            f"Test AUC: {test_auc:.4f}, Test MAP: {test_map:.4f}, "
            f"Test Recall: {test_recall:.4f}, Test Acc: {test_acc:.4f}, "
            f"Test Precision: {test_precision:.4f}")

    '''plot_scores(
        train_f1_scores, val_f1_scores,
        train_focal_loss_scores, val_focal_loss_scores,
        train_auc_scores, val_auc_scores,
        train_map_scores, val_map_scores,
        train_recall_scores, val_recall_scores,
        train_acc_scores, val_acc_scores,
        train_precision_scores, val_precision_scores,
        args.epochs, output_path)'''

    test_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=G_dgl.number_of_nodes())
    top_10_predictions = []

    for node in range(G_dgl.number_of_nodes()):
        h = model(G_dgl, node_features)
        scores = pred(test_g, h).squeeze()
        _, indices = torch.topk(scores, 10)

        for idx in indices:
            source_node = node_id_to_name[node]
            target_node = node_id_to_name[idx.item()]
            source_type = G_dgl.ndata['feat'][node].type  # Assuming node features include 'type'
            target_type = G_dgl.ndata['feat'][idx.item()].type  # Assuming node features include 'type'

            if source_type == 'miRNA' and target_type == 'disease':
                top_10_predictions.append((source_node, target_node, scores[idx].item()))

    output_file = './link_prediction_gat/top_10_predictions.csv'
    df = pd.DataFrame(top_10_predictions, columns=['miRNA', 'Disease', 'Score'])
    df.to_csv(output_file, index=False)
    print(f"Top 10 predictions saved to {output_file}")

import argparse

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

    G_dgl, node_features, node_id_to_name = load_graph_data('data/miRNA_disease_embeddings.json')
    print('node_features.shape============\n',node_features)
    train_and_evaluate(args, G_dgl, node_features, node_id_to_name)
