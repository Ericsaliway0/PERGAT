import csv
import json
import os
from matplotlib import pyplot as plt
import pandas as pd
import torch
import itertools
import dgl
import numpy as np
import scipy.sparse as sp
from .models import GATModel, MLPPredictor, FocalLoss
from .utils import (plot_training_validation_metrics, plot_roc_pr_curves, find_evidence, compute_hits_k, compute_auc, compute_f1, compute_focalloss,
                    compute_accuracy, compute_precision, compute_recall, compute_map,
                    compute_focalloss_with_symmetrical_confidence, compute_auc_with_symmetrical_confidence,
                    compute_f1_with_symmetrical_confidence, compute_accuracy_with_symmetrical_confidence,
                    compute_precision_with_symmetrical_confidence, compute_recall_with_symmetrical_confidence,
                    compute_map_with_symmetrical_confidence)
from scipy.stats import sem
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import networkx as nx
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.model_selection import LeaveOneOut
 

def train_and_evaluate(args, G_dgl, node_features, node_id_to_name):
    u, v = G_dgl.edges()
    eids = np.arange(G_dgl.number_of_edges())

    # Convert node IDs to node names
    u_names = [node_id_to_name[uid.item()] for uid in u]
    v_names = [node_id_to_name[vid.item()] for vid in v]
    
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(G_dgl.number_of_nodes(), G_dgl.number_of_nodes()))
    adj_neg = 1 - adj.todense() - np.eye(G_dgl.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    
    all_fold_results = pd.DataFrame()
    fold_results = []
    all_fold_results_breast_cancer = []
    all_fold_results_cancer = []
   

    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
        

    # Process top predictions
    top_predictions_u = []
    top_predictions_v = []
    
    ##kf = KFold(n_splits=5, shuffle=True, random_state=66)
    kf = KFold(n_splits=5, shuffle=True)
    output_path = 'results/'
    os.makedirs(output_path, exist_ok=True)

    for fold, (train_idx, test_idx) in enumerate(kf.split(eids)):
        print(f'Fold {fold + 1}')

        val_size = int(len(train_idx) * 0.8)
        train_idx, val_idx = train_idx[val_size:], train_idx[:val_size]

        '''np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)'''
        
        '''train_u_names = [u_names[i] for i in train_idx]
        train_v_names = [v_names[i] for i in train_idx]

        # Exclude "breast cancer" edges from the training set
        # Combine conditions for both u_names and v_names

        breast_cancer_mask = np.array([
            "stomach neoplasms" in u_name.lower() or "stomach neoplasms" in v_name.lower()
            for u_name, v_name in zip(train_u_names, train_v_names)
        ])

        # Count the number of filtered-out edges
        num_filtered_edges = np.sum(breast_cancer_mask)
        print(f'Total number of filtered-out edges related to "stomach neoplasms": {num_filtered_edges}')

        # Apply the mask to filter out such edges from the training set
        
        train_idx = train_idx[~breast_cancer_mask]'''  # Invert the mask to exclude "breast cancer" nodes


        train_pos_u, train_pos_v = u[train_idx], v[train_idx]
        val_pos_u, val_pos_v = u[val_idx], v[val_idx]
        test_pos_u, test_pos_v = u[test_idx], v[test_idx]

        neg_eids = np.random.choice(len(neg_u), G_dgl.number_of_edges())
        train_neg_u, train_neg_v = neg_u[neg_eids[:len(train_idx)]], neg_v[neg_eids[:len(train_idx)]]
        val_neg_u, val_neg_v = neg_u[neg_eids[len(train_idx):len(train_idx) + len(val_idx)]], neg_v[neg_eids[len(train_idx):len(train_idx) + len(val_idx)]]
        test_neg_u, test_neg_v = neg_u[neg_eids[len(train_idx) + len(val_idx):]], neg_v[neg_eids[len(train_idx) + len(val_idx):]]

        train_g = dgl.remove_edges(G_dgl, test_idx.tolist() + val_idx.tolist())

        def create_graph(u, v, num_nodes):
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
        
        '''model = GraphSAGE(
            in_feats=node_features.size(1), 
            hidden_feats=args.hidden_feats, 
            out_feats=args.out_feats, 
            num_layers=args.num_layers
        )'''

        '''model = GCNModel(
            node_features.shape[1], 
            dim_latent=args.out_feats, 
            num_layers=args.num_layers, 
            do_train=True
        )'''

        pred = MLPPredictor(args.input_size, args.hidden_size)
        criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1000, gamma=0.1)

        fold_train_accuracies = []
        fold_val_accuracies = []
        fold_train_losses = []
        fold_val_losses = []

        # Using tqdm for progress bar
        for e in tqdm(range(args.epochs), desc=f"Training Fold {fold + 1}"):
        ##for e in range(args.epochs):
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

            fold_train_losses.append(loss.item())

            with torch.no_grad():
                model.eval()
                h_val = model(G_dgl, G_dgl.ndata['feat'])
                val_pos_score = pred(val_pos_g, h_val)
                val_neg_score = pred(val_neg_g, h_val)
                
                val_all_scores = torch.cat([val_pos_score, val_neg_score])
                val_all_labels = torch.cat([torch.ones_like(val_pos_score), torch.zeros_like(val_neg_score)])
                
                val_loss = criterion(val_all_scores, val_all_labels)
                fold_val_losses.append(val_loss.item())
                
                val_acc = ((val_pos_score > 0.5).sum().item() + (val_neg_score <= 0.5).sum().item()) / (len(val_pos_score) + len(val_neg_score))
                fold_val_accuracies.append(val_acc)

                train_acc = ((pos_score > 0.5).sum().item() + (neg_score <= 0.5).sum().item()) / (len(pos_score) + len(neg_score))
                fold_train_accuracies.append(train_acc)
            
            if e % 5 == 0:
                print(f'Fold {fold + 1} | Epoch {e} | Loss: {loss.item()} | Train Accuracy: {train_acc:.4f} | Val Accuracy: {val_acc:.4f}')
        
        train_accuracies.append(fold_train_accuracies)
        val_accuracies.append(fold_val_accuracies)
        train_losses.append(fold_train_losses)
        val_losses.append(fold_val_losses)
        
        ##print('val_losses================\n',val_losses)

        with torch.no_grad():
            model.eval()
            h_test = model(G_dgl, G_dgl.ndata['feat'])
            test_pos_score = pred(test_pos_g, h_test)
            test_neg_score = pred(test_neg_g, h_test)
            
            # Apply sigmoid to test scores
            test_pos_score = torch.sigmoid(test_pos_score)
            test_neg_score = torch.sigmoid(test_neg_score)
            test_auc, test_auc_err = compute_auc_with_symmetrical_confidence(test_pos_score, test_neg_score)
            test_f1, test_f1_err = compute_f1_with_symmetrical_confidence(test_pos_score, test_neg_score)
            test_focal_loss, test_focal_loss_err = compute_focalloss_with_symmetrical_confidence(test_pos_score, test_neg_score)
            test_precision, test_precision_err = compute_precision_with_symmetrical_confidence(test_pos_score, test_neg_score)
            test_recall, test_recall_err = compute_recall_with_symmetrical_confidence(test_pos_score, test_neg_score)
            test_hits_k = compute_hits_k(test_pos_score, test_neg_score, k=10)
            test_map, test_map_err = compute_map_with_symmetrical_confidence(test_pos_score, test_neg_score)
            test_accuracy, test_accuracy_err = compute_accuracy_with_symmetrical_confidence(test_pos_score, test_neg_score)

            test_metrics = (
                f'Test AUC: {test_auc:.4f} ± {test_auc_err:.4f} | Test F1: {test_f1:.4f} ± {test_f1_err:.4f} | '
                f'Test FocalLoss: {test_focal_loss:.4f} ± {test_focal_loss_err:.4f} | Test Accuracy: {test_accuracy:.4f} ± {test_accuracy_err:.4f} | '
                f'Test Precision: {test_precision:.4f} ± {test_precision_err:.4f} | Test Recall: {test_recall:.4f} ± {test_recall_err:.4f} | '
                f'Test Hits@10: {test_hits_k:.4f} | Test MAP: {test_map:.4f} ± {test_map_err:.4f}'
            )
            print(test_metrics)

            # Save the test metrics to a .txt file
            output_path_test_metrics = f'test_curve_5_fold_drop{args.attn_drop}_head{args.num_heads}_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.txt'

            with open(os.path.join(output_path, output_path_test_metrics), 'a') as f:
                f.write(f'Fold {fold + 1}:\n')
                f.write(test_metrics + '\n\n')

            true_labels = torch.cat([torch.ones(len(test_pos_score)), torch.zeros(len(test_neg_score))])
            predicted_scores = torch.cat([test_pos_score, test_neg_score]).cpu().numpy()

            fold_results.append((true_labels.cpu().numpy(), predicted_scores))

            fold_result_data = pd.DataFrame({
                'Fold': [fold + 1],
                'Test AUC': [test_auc],
                'Test AUC Err': [test_auc_err],
                'Test F1 Score': [test_f1],
                'Test F1 Score Err': [test_f1_err],
                'Test Precision': [test_precision],
                'Test Precision Err': [test_precision_err],
                'Test Recall': [test_recall],
                'Test Recall Err': [test_recall_err],
                'Test Hit': [test_hits_k],
                'Test mAP': [test_map],
                'Test mAP Err': [test_map_err],
                'Test FocalLoss': [test_focal_loss],
                'Test FocalLoss Err': [test_focal_loss_err],
                'Test Accuracy': [test_accuracy],
                'Test Accuracy Err': [test_accuracy_err]
            })

            all_fold_results = pd.concat([all_fold_results, fold_result_data], ignore_index=True)

        '''plot_roc_curves(fold_results, output_path_cross_roc)
        plot_pr_curves(fold_results, output_path_cross_pr)'''
        output_path_roc_pr = os.path.join(output_path, f'roc_pr_curve_5_fold_drop{args.attn_drop}_head{args.num_heads}_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.png')
        plot_roc_pr_curves(fold_results, output_path_roc_pr)



        avg_train_accuracies = np.mean(train_accuracies, axis=0)
        avg_val_accuracies = np.mean(val_accuracies, axis=0)
        avg_train_losses = np.mean(train_losses, axis=0)
        avg_val_losses = np.mean(val_losses, axis=0)

        plot_training_validation_metrics(
            train_accuracies, avg_train_accuracies,
            val_accuracies, avg_val_accuracies,
            train_losses, avg_train_losses,
            val_losses, avg_val_losses,
            output_path, args
        )


        # Process top predictions

        for uu, vv, score in zip(test_pos_u, test_pos_v, test_pos_score):
            if any(x in node_id_to_name[uu.item()] for x in ['rno-', 'mmu-', 'hsa-', 'EBV-', 'hcmv-', 'mdv1-', 'kshv-']):
                if not any(x in node_id_to_name[vv.item()] for x in ['rno-', 'mmu-', 'hsa-', 'EBV-', 'hcmv-', 'mdv1-', 'kshv-']):
                    top_predictions_u.append({
                        'source': node_id_to_name[uu.item()],
                        'destination': node_id_to_name[vv.item()],
                        'score': score.item()
                    })
            elif any(x in node_id_to_name[vv.item()] for x in ['rno-', 'mmu-', 'hsa-', 'EBV-', 'hcmv-', 'mdv1-', 'kshv-']):
                if not any(x in node_id_to_name[uu.item()] for x in ['rno-', 'mmu-', 'hsa-', 'EBV-', 'hcmv-', 'mdv1-', 'kshv-']):
                    top_predictions_v.append({
                        'source': node_id_to_name[vv.item()],
                        'destination': node_id_to_name[uu.item()],
                        'score': score.item()
                        })

        



        # Define a function to save DataFrame
        def save_dataframe(df, filename):
            df.to_csv(os.path.join(output_path, filename), index=False)

        # Sort predictions by score in descending order
        top_predictions_u.sort(key=lambda x: x['score'], reverse=True)
        top_predictions_v.sort(key=lambda x: x['score'], reverse=True)

        # Convert top predictions to DataFrames
        df_top_u = pd.DataFrame(top_predictions_u)
        df_top_v = pd.DataFrame(top_predictions_v)

        # Save the top predictions DataFrames
        filename_u = f'top_scores_u_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.csv'
        save_dataframe(df_top_u, filename_u)
        filename_v = f'top_scores_v_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.csv'
        save_dataframe(df_top_v, filename_v)

        # Load reference data
        file2 = 'data/_combined_hmdd_miR2Disease_miRNA_disease.csv'
        df2 = pd.read_csv(file2)

        # Merge top predictions with reference data
        df1 = pd.concat([df_top_u, df_top_v], ignore_index=True).sort_values(by='score', ascending=False)

        # Prepare the final DataFrame
        combined_df = df1.copy()

        def get_evidence(row):
            # Initialize a set to hold the unique evidence references
            evidence_set = set()

            # Check if the source matches any miRNA in df2 and add its reference(s) to the set
            if (df2['miRNA'] == row['source']).any():
                evidence_set.update(df2[df2['miRNA'] == row['source']]['reference'].tolist())

            # Check if the destination matches any miRNA in df2 and add its reference(s) to the set
            if (df2['miRNA'] == row['destination']).any():
                evidence_set.update(df2[df2['miRNA'] == row['destination']]['reference'].tolist())

            # If there is evidence, return it as a comma-separated string, otherwise return 'Unconfirmed'
            if evidence_set:
                return ', '.join(evidence_set)
            else:
                return 'Unconfirmed'


        # Set the evidence column
        ## list
        '''def get_evidence(row):
            # Initialize a list to hold the evidence
            evidence_list = []

            # Check if the source matches any miRNA in df2 and add its reference to the list
            if (df2['miRNA'] == row['source']).any():
                evidence_list.extend(df2[df2['miRNA'] == row['source']]['reference'].tolist())

            # Check if the destination matches any miRNA in df2 and add its reference to the list
            if (df2['miRNA'] == row['destination']).any():
                evidence_list.extend(df2[df2['miRNA'] == row['destination']]['reference'].tolist())

            # Remove duplicates by converting the list to a set and back to a list
            unique_evidence_list = list(set(evidence_list))

            # If there is evidence, return it, otherwise return 'Unconfirmed'
            if unique_evidence_list:
                # Option 1: Return as a list
                return unique_evidence_list
                # Option 2: Return as a concatenated string (comma-separated)
                # return ', '.join(unique_evidence_list)
            else:
                return 'Unconfirmed'
            '''



        combined_df['evidence'] = combined_df.apply(get_evidence, axis=1)

        # Save the combined DataFrame
        filename_breast_cancer = f'top_breast_cancer_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.csv'
        combined_df.to_csv(os.path.join(output_path, filename_breast_cancer), index=False)

        print('Combined DataFrame saved to:', filename_breast_cancer)
