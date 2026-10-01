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
from .models import LinkPredictor, GATModel, MLPPredictor, FocalLoss
from .utils import (plot_scores, compute_hits_k, compute_auc, compute_f1, compute_focalloss,
                    compute_accuracy, compute_precision, compute_recall, compute_map,
                    compute_focalloss_with_symmetrical_confidence, compute_auc_with_symmetrical_confidence,
                    compute_f1_with_symmetrical_confidence, compute_accuracy_with_symmetrical_confidence,
                    compute_precision_with_symmetrical_confidence, compute_recall_with_symmetrical_confidence,
                    compute_map_with_symmetrical_confidence)
from scipy.stats import sem
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import networkx as nx

##    if 'hsa-' in node_id_to_name[u.item()] and 'cancer' in node_id_to_name[v.item()]:
    
def train_and_evaluate(args, G_dgl, node_features, node_id_to_name):
    u, v = G_dgl.edges()
    eids = np.arange(G_dgl.number_of_edges())
    eids = np.random.permutation(eids)

    '''test_size = int(len(eids) * 0.1)
    val_size = int(len(eids) * 0.1)
    train_size = G_dgl.number_of_edges() - test_size - val_size'''

    # Count edges where the destination node is associated with 'cancer'
    '''for idx, name in node_id_to_name.items():
        if 'Cancer' in name or 'cancer' in name:
            print('name____________________\n',name)'''
            
    cancer_nodes = {name for idx, name in node_id_to_name.items() if 'Cancer' in name or 'cancer' in name} ##name.lower()}
    ##print('cancer_nodes\n',cancer_nodes)
    cancer_edges = [i for i, node in enumerate(v.numpy()) if node_id_to_name[node] in cancer_nodes]
    ##print('cancer_edges\n',cancer_edges)
    test_size = len(cancer_edges)
    ##print('test_size==========\n',test_size)
    

    # Define validation size as a fraction of the remaining edges
    val_size = int((len(eids) - test_size) * 0.1)
    train_size = len(eids) - test_size - val_size

    # Split the edges into test, validation, and training sets
    test_eids = np.array(cancer_edges)[:test_size]
    remaining_eids = np.setdiff1d(eids, test_eids)
    val_eids = remaining_eids[:val_size]
    train_eids = remaining_eids[val_size:]

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

    '''plot_scores(
        train_f1_scores, val_f1_scores,
        train_focal_loss_scores, val_focal_loss_scores,
        train_auc_scores, val_auc_scores,
        train_map_scores, val_map_scores,
        train_recall_scores, val_recall_scores,
        train_acc_scores, val_acc_scores,
        train_precision_scores, val_precision_scores,
        epochs, output_path, args
    )'''

    # Save the test results to a JSON file in the specified format

    # Save the test results to a CSV file
    test_results = {
        ##'Learning Rate': [args.lr],
        ##'Epochs': [args.epochs],
        ##'Input Features': [args.input_size],
        ##'Output Features': [args.out_feats],
        'Test AUC': [f'{test_auc:.4f} ± {test_auc_err:.4f}'],
        'Test F1 Score': [f'{test_f1:.4f} ± {test_f1_err:.4f}'],
        'Test Precision': [f'{test_precision:.4f} ± {test_precision_err:.4f}'],
        'Test Recall': [f'{test_recall:.4f} ± {test_recall_err:.4f}'],
        'Test Hit': [f'{test_hits_k:.4f}'],
        'Test mAP': [f'{test_map:.4f} ± {test_map_err:.4f}']
    }

    df_test_results = pd.DataFrame(test_results)
    filename = f'test_results_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.csv'
    df_test_results.to_csv(os.path.join(output_path, filename), index=False)

    '''test_results = {
        'Learning Rate': args.lr,
        'Epochs': args.epochs,
        'Input Features': args.input_size,
        'Output Features': args.out_feats,
        'Test AUC': f'{test_auc:.4f} ± {test_auc_err:.4f}',
        'Test F1 Score': f'{test_f1:.4f} ± {test_f1_err:.4f}',
        'Test Precision': f'{test_precision:.4f} ± {test_precision_err:.4f}',
        'Test Recall': f'{test_recall:.4f} ± {test_recall_err:.4f}',
        'Test Hit': f'{test_hits_k:.4f}',
        'Test mAP': f'{test_map:.4f} ± {test_map_err:.4f}'
    }

    filename = f'test_results_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.json'
    with open(os.path.join(output_path, filename), 'w') as f:
        json.dump(test_results, f)'''
        
    # Save the top 20 predictions with miRNA and disease names
    top_predictions = []
    for u, v, score in zip(test_pos_u, test_pos_v, test_pos_score):
        ##print('node_id_to_name[u.item()]-----------------\n',node_id_to_name[u.item()])
        if 'hsa-' in node_id_to_name[u.item()] or 'EBV-' in node_id_to_name[u.item()] or 'hcmv-' in node_id_to_name[u.item()] or 'mdv1-' in node_id_to_name[u.item()] or 'kshv-' in node_id_to_name[u.item()]:
            if 'cancer' in node_id_to_name[v.item()] or 'Cancer' in node_id_to_name[v.item()]:
            ##if 'hsa-mir' in node_id_to_name[u.item()] and 'hsa-mir' not in node_id_to_name[v.item()]:
                top_predictions.append({
                    'source': node_id_to_name[u.item()],  # miRNA as source
                    'destination': node_id_to_name[v.item()],  # Disease as destination
                    'score': score.item()
                })
            ##print('node_id_to_name[u.item()]===============\n',node_id_to_name[u.item()])

    top_predictions.sort(key=lambda x: x['score'], reverse=True)
    top_predictions = top_predictions[:50]

    df_top_50 = pd.DataFrame(top_predictions)
    filename_ = f'top_scores_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.csv'
    with open(os.path.join(output_path, filename_), 'w') as f:
        ##json.dump(test_results, f)
        df_top_50.to_csv(os.path.join(output_path, filename_), index=False)
        
    print('test_size==========\n',test_size)
