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
from .utils import (plot_roc_curves, plot_pr_curves, compute_hits_k, compute_auc, compute_f1, compute_focalloss,
                    compute_accuracy, compute_precision, compute_recall, compute_map,
                    compute_focalloss_with_symmetrical_confidence, compute_auc_with_symmetrical_confidence,
                    compute_f1_with_symmetrical_confidence, compute_accuracy_with_symmetrical_confidence,
                    compute_precision_with_symmetrical_confidence, compute_recall_with_symmetrical_confidence,
                    compute_map_with_symmetrical_confidence)
from scipy.stats import sem
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import networkx as nx
from sklearn.model_selection import KFold

import copy
import itertools
import os
import re

import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR

from .models import GATModel, MLPPredictor, FocalLoss
from .utils import plot_roc_curves, plot_pr_curves

MIRNA_PREFIXES = ("rno-", "mmu-", "hsa-", "EBV-", "hcmv-", "mdv1-", "kshv-")
CANCER_WORDS = ("cancer", "neoplasm", "carcinoma", "tumor", "tumour", "leukemia", "lymphoma", "melanoma",
                "glioma", "sarcoma", "myeloma", "blastoma")


def is_mirna(name):
    return any(p in name for p in MIRNA_PREFIXES)


def is_cancer(name):
    n = name.lower()
    return (not is_mirna(name)) and any(w in n for w in CANCER_WORDS)


def norm_name(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def to_prob(t):
    """Scores -> probabilities. The predictor is trained with a logit-based focal loss, so apply sigmoid unless
    the scores already lie in [0, 1]."""
    t = t.detach().cpu().float()
    return (torch.sigmoid(t) if (t.min() < 0 or t.max() > 1) else t).numpy()


def val_f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-12)
    return float(th[int(np.argmax(f))]) if len(th) else 0.5


def hits_at_k(pos, neg, k=10):
    """OGB-style Hits@K: share of positives scored above the K-th highest negative."""
    kth = np.sort(neg)[-min(k, len(neg))]
    return float((pos > kth).mean())


def lp_metrics(y, p, thr_fixed, thr_val, pos, neg):
    out = dict(auc=roc_auc_score(y, p), ap=average_precision_score(y, p), hits10=hits_at_k(pos, neg, 10),
               hits50=hits_at_k(pos, neg, 50), thr_val=thr_val)
    for tag, thr in (("fixed", thr_fixed), ("valF1", thr_val)):
        yh = (p >= thr).astype(int)
        out.update({f"accuracy@{tag}": accuracy_score(y, yh), f"balanced_acc@{tag}": balanced_accuracy_score(y, yh),
                    f"precision@{tag}": precision_score(y, yh, zero_division=0),
                    f"recall@{tag}": recall_score(y, yh, zero_division=0), f"f1@{tag}": f1_score(y, yh, zero_division=0),
                    f"mcc@{tag}": matthews_corrcoef(y, yh)})
    return out


def reverse_eids(G, eids):
    """Edge ids of the reverse edges (v->u) of the given edges, if present (undirected graphs stored both ways)."""
    u, v = G.find_edges(torch.as_tensor(eids))
    has = G.has_edges_between(v, u)
    if not has.any():
        return np.array([], dtype=np.int64)
    return G.edge_ids(v[has], u[has]).numpy()

def train_and_evaluate(args, G_dgl, node_features, node_id_to_name):
    u, v = G_dgl.edges()
    eids = np.arange(G_dgl.number_of_edges())
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(G_dgl.number_of_nodes(), G_dgl.number_of_nodes()))
    adj_neg = 1 - adj.todense() - np.eye(G_dgl.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)
    
    all_fpr = []
    all_tpr = []
    all_auc = []
    
    fold_results = []
    metric_results = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    output_path = './link_prediction_gat/results/'
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(eids)):
        print(f'Fold {fold + 1}')
       
        val_size = int(len(train_idx) * 0.1)
        train_idx, val_idx = train_idx[val_size:], train_idx[:val_size]

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

        pred = MLPPredictor(args.input_size, args.hidden_size)
        criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1000, gamma=0.1)
        
        output_path_cross = f'./link_prediction_gat/results/roc_curve_5_fold.png'

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
                print(f'Fold {fold + 1} | Epoch {e} | Loss: {loss.item()}')
        
        with torch.no_grad():
            model.eval()
            h_test = model(train_g, train_g.ndata['feat'])
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

            metric_results.append({
                'Fold': fold + 1,
                'AUC': (test_auc, test_auc_err),
                'F1': (test_f1, test_f1_err),
                'Accuracy': (test_accuracy, test_accuracy_err),
                'Precision': (test_precision, test_precision_err),
                'Recall': (test_recall, test_recall_err),
                'mAP': (test_map, test_map_err),
            })

            '''print(f'Fold {fold + 1} Test Results:')
            print(f'AUC: {test_auc:.4f} ± {test_auc_err:.4f}')
            print(f'F1: {test_f1:.4f} ± {test_f1_err:.4f}')
            print(f'FocalLoss: {test_focal_loss:.4f} ± {test_focal_loss_err:.4f}')
            print(f'Accuracy: {test_accuracy:.4f} ± {test_accuracy_err:.4f}')
            print(f'Precision: {test_precision:.4f} ± {test_precision_err:.4f}')
            print(f'Recall: {test_recall:.4f} ± {test_recall_err:.4f}')
            print(f'Hits@10: {test_hits_k:.4f}')
            print(f'MAP: {test_map:.4f} ± {test_map_err:.4f}')'''
 
            # Concatenate positive and negative scores and labels
            scores = torch.cat([test_pos_score, test_neg_score]).cpu().numpy()
            labels = torch.cat([torch.ones_like(test_pos_score), torch.zeros_like(test_neg_score)]).cpu().numpy()
            
            # Compute FPR, TPR, and AUC
            '''fpr, tpr, _ = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)
            
            # Store the FPR, TPR, and AUC for this fold
            all_fpr.append(fpr)
            all_tpr.append(tpr)
            all_auc.append(roc_auc)'''
            
            ##print(f'Fold {fold + 1} | AUC: {roc_auc:.4f}')
                    
            # Prepare true labels and scores
            '''true_labels = torch.cat([torch.ones_like(test_pos_score), torch.zeros_like(test_neg_score)]).cpu().numpy()
            predicted_scores = torch.cat([test_pos_score, test_neg_score]).cpu().numpy()
            
            # Plot and save the ROC curve
            plot_and_save_roc_curve(true_labels, predicted_scores, fold, output_path_cross)'''
    
               
            print(f'Test AUC: {test_auc:.4f} ± {test_auc_err:.4f} | Test F1: {test_f1:.4f} ± {test_f1_err:.4f} | Test FocalLoss: {test_focal_loss:.4f} ± {test_focal_loss_err:.4f} |Test Accuracy: {test_accuracy:.4f} ± {test_accuracy_err:.4f} | Test Precision: {test_precision:.4f} ± {test_precision_err:.4f} | Test Recall: {test_recall:.4f} ± {test_recall_err:.4f} | Test Hits@10: {test_hits_k:.4f} | Test MAP: {test_map:.4f} ± {test_map_err:.4f}')

            # Prepare true labels and predicted scores
            true_labels = torch.cat([torch.ones(len(test_pos_score)), torch.zeros(len(test_neg_score))])
            predicted_scores = torch.cat([test_pos_score, test_neg_score])
            
            # Append results for this fold
            fold_results.append((true_labels.numpy(), predicted_scores.detach().numpy()))



        # Save results for the fold
        test_results = {
            'Test AUC': [f'{test_auc:.4f} ± {test_auc_err:.4f}'],
            'Test F1 Score': [f'{test_f1:.4f} ± {test_f1_err:.4f}'],
            'Test Precision': [f'{test_precision:.4f} ± {test_precision_err:.4f}'],
            'Test Recall': [f'{test_recall:.4f} ± {test_recall_err:.4f}'],
            'Test Hit': [f'{test_hits_k:.4f}'],
            'Test mAP': [f'{test_map:.4f} ± {test_map_err:.4f}']
        }

        df_test_results = pd.DataFrame(test_results)
        ##output_path = f'./link_prediction_gat/results/fold_{fold + 1}/'
        os.makedirs(output_path, exist_ok=True)
        filename = f'test_results_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.csv'
        df_test_results.to_csv(os.path.join(output_path, filename), index=False)

    # Save fold metrics as a LaTeX table, with scores and confidence intervals
    # displayed as percentages to match the manuscript table format.
    def latex_metric(value_error):
        value, error = value_error
        return f'{value * 100:.2f}±{error * 100:.2f}'

    metric_columns = ['AUC', 'F1', 'Accuracy', 'Precision', 'Recall', 'mAP']
    latex_rows = []
    for result in metric_results:
        values = ' & '.join(latex_metric(result[column]) for column in metric_columns)
        latex_rows.append(f"\t\t{result['Fold']} & {values} " + r"\\")

    mean_values = []
    for column in metric_columns:
        mean_value = np.mean([result[column][0] for result in metric_results])
        mean_error = np.mean([result[column][1] for result in metric_results])
        mean_values.append(latex_metric((mean_value, mean_error)))
    latex_rows.append('\t\t\\textbf{Mean} & ' + ' & '.join(f'\\textbf{{{value}}}' for value in mean_values) + ' \\\\')

    latex_table = '\n'.join([
        '\\begin{table}[h]',
        '\t\\centering',
        '\t\\captionsetup{font=scriptsize}',
        '\t\\scriptsize',
        '\t\\setlength{\\tabcolsep}{2.6pt}',
        '\t\\begin{tabular}{lllllll}',
        '\t\t\\hline',
        '\t\t\\textbf{Fold} & \\textbf{AUC (\\%)} & \\textbf{F1 (\\%)} & \\textbf{Acc. (\\%)} & \\textbf{Prec. (\\%)} & \\textbf{Recall (\\%)} & \\textbf{mAP (\\%)} \\\\',
        '\t\t\\hline',
        *latex_rows,
        '\t\t\\hline',
        '\t\\end{tabular}',
        '\t\\captionsetup{font=scriptsize}',
        '\t\\caption{The effectiveness of our approach assessed using 5-fold cross-validation. Mean scores are shown in bold.}',
        '\t\\label{tab:table33}',
        '\\end{table}',
    ]) + '\n'
    latex_filename = f'fold_metrics_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.tex'
    with open(os.path.join(output_path, latex_filename), 'w', encoding='utf-8') as latex_file:
        latex_file.write(latex_table)

    # After the cross-validation loop, plot and save the ROC curves
    plot_roc_curves(fold_results, output_path_cross)
    
    top_predictions_u = []
    top_predictions_v = []
    for u, v, score in zip(test_pos_u, test_pos_v, test_pos_score):
        if 'rno-' in node_id_to_name[u.item()] or 'mmu-' in node_id_to_name[u.item()] or 'hsa-' in node_id_to_name[u.item()] or 'EBV-' in node_id_to_name[u.item()] or 'hcmv-' in node_id_to_name[u.item()] or 'mdv1-' in node_id_to_name[u.item()] or 'kshv-' in node_id_to_name[u.item()]:
            if ('cancer' in node_id_to_name[v.item()] or 'Cancer' in node_id_to_name[v.item()]) and 'rno-' not in node_id_to_name[v.item()] and 'mmu-' not in node_id_to_name[v.item()] and 'hsa-' not in node_id_to_name[v.item()] and 'EBV-' not in node_id_to_name[v.item()] and 'hcmv-' not in node_id_to_name[v.item()] and 'mdv1-' not in node_id_to_name[v.item()] and 'kshv-' not in node_id_to_name[v.item()]:
                top_predictions_u.append({
                    'source': node_id_to_name[u.item()],  # miRNA as source
                    'destination': node_id_to_name[v.item()],  # Disease as destination
                    'score': score.item()
                })
        elif 'rno-' in node_id_to_name[v.item()] or 'mmu-' in node_id_to_name[v.item()] or 'hsa-' in node_id_to_name[v.item()] or 'EBV-' in node_id_to_name[v.item()] or 'hcmv-' in node_id_to_name[v.item()] or 'mdv1-' in node_id_to_name[v.item()] or 'kshv-' in node_id_to_name[v.item()]:
            if ('cancer' in node_id_to_name[u.item()] or 'Cancer' in node_id_to_name[u.item()]) and 'rno-' not in node_id_to_name[u.item()] and 'mmu-' not in node_id_to_name[u.item()] and 'hsa-' not in node_id_to_name[u.item()] and 'EBV-' not in node_id_to_name[u.item()] and 'hcmv-' not in node_id_to_name[u.item()] and 'mdv1-' not in node_id_to_name[u.item()] and 'kshv-' not in node_id_to_name[u.item()]:
                top_predictions_v.append({
                    'source': node_id_to_name[u.item()],  # Disease as source
                    'destination': node_id_to_name[v.item()],  # miRNA as destination
                    'score': score.item()
                })

    top_predictions_u.sort(key=lambda x: x['score'], reverse=True)
    top_predictions_v.sort(key=lambda x: x['score'], reverse=True)
    top_predictions_u = top_predictions_u[:27061]
    top_predictions_v = top_predictions_v[:27061]

    df_top_u = pd.DataFrame(top_predictions_u)
    filename_u = f'top_scores_u_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.csv'
    df_top_u.to_csv(os.path.join(output_path, filename_u), index=False)

    df_top_v = pd.DataFrame(top_predictions_v)
    filename_v = f'top_scores_v_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.csv'
    df_top_v.to_csv(os.path.join(output_path, filename_v), index=False)

    ##print('test_size==========\n', test_size)

    file2 = 'gat/data/merged_all_data.csv'
    filename_u_cancer = f'top_cancer_u_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.csv'

    df1 = pd.read_csv(os.path.join(output_path, filename_u))
    df2 = pd.read_csv(file2)

    # Merge the DataFrames based on 'miRNA' == 'source' and 'disease' == 'destination'
    merged_df = pd.merge(df2, df1, left_on=['miRNA', 'disease'], right_on=['source', 'destination'])

    # Display the rows with matching miRNA and disease with source and destination
    print(merged_df)

    # Optionally, save the matched rows to a new CSV file
    with open(os.path.join(output_path, filename_u_cancer), 'w') as f:
        merged_df.to_csv(f, index=False)

    # Load the file with descriptions
    file3 = 'data/hmdd_alldata_v4_updated.csv'
    df3 = pd.read_csv(file3)

    # Filter predictions where miRNA equals the column miRNA, and the column description contains breast cancer
    filtered_df = df1[df1['source'].isin(df3[df3['description'].str.contains('breast cancer', case=False, na=False)]['miRNA'])]


    # Check for similarity with df2 and merge 
    # result_df = pd.merge(cancer_miRNA_df, merged_df[['miRNA', 'disease', 'reference']], on=['miRNA', 'disease'], how='left')

    ##merged_df_bc = pd.merge(df2, filtered_df, left_on=['miRNA', 'disease'], right_on=['source', 'destination'])
    # Check for similarity with df2 where the disease is 'breast cancer'
    ##merged_df_bc = pd.merge(df2[df2['disease'].str.contains('breast cancer', case=False, na=False)], filtered_df, left_on=['miRNA', 'disease'], right_on=['source', 'destination'])


    # Append ', breast cancer related' to the destination field
    ##filtered_df['destination'] = filtered_df['destination'] + ', breast cancer related'

    # Check for similarity with df2 where miRNA equals disease
    ##merged_df_bc = pd.merge(df2, filtered_df, left_on=['miRNA', 'disease'], right_on=['source', 'destination'])
    merged_df_bc = pd.merge(df2[df2['disease'].str.contains('breast cancer', case=False, na=False)], filtered_df, left_on=['miRNA'], right_on=['source'])



    # Save the filtered predictions to a new CSV file
    filename_breast_cancer = f'top_breast_cancer_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.csv'
    with open(os.path.join(output_path, filename_breast_cancer), 'w') as f:
        merged_df_bc.to_csv(f, index=False)

    '''filename_v_cancer = f'top_cancer_v_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.csv'

    df1 = pd.read_csv(os.path.join(output_path, filename_v))
    df2 = pd.read_csv(file2)

    # Merge the DataFrames based on 'miRNA' == 'source' and 'disease' == 'destination'
    merged_df = pd.merge(df2, df1, left_on=['miRNA', 'disease'], right_on=['destination', 'source'])

    # Display the rows with matching miRNA and disease with source and destination
    print(merged_df)

    # Optionally, save the matched rows to a new CSV file
    with open(os.path.join(output_path, filename_v_cancer), 'w') as f:
        ##json.dump(test_results, f)
        merged_df.to_csv(f, index=False)'''
                
    ##merged_df.to_csv('output_path/data/matched_rows_epoch24_u.csv', index=False)
     


def train_and_evaluate_ori(args, G_dgl, node_features, node_id_to_name, seed=42, output_path="results/"):
    os.makedirs(output_path, exist_ok=True)
    N, E = G_dgl.number_of_nodes(), G_dgl.number_of_edges()
    u, v = G_dgl.edges()
    eids = np.arange(E)
    tag = f"lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}"

    adj = sp.coo_matrix((np.ones(E), (u.numpy(), v.numpy())), shape=(N, N)).tocsr()
    adj = ((adj + adj.T) > 0).astype(np.int8)                       # symmetric: a reverse edge also counts as known
    adj_neg = 1 - adj.toarray() - np.eye(N, dtype=np.int8)
    neg_u, neg_v = np.where(adj_neg > 0)
    del adj_neg
    print(f"nodes={N:,} edges={E:,} candidate negatives={len(neg_u):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patience = getattr(args, "patience", 30)
    fold_curves, fold_rows, fold_states = [], [], []

    for fold, (trainval_idx, test_idx) in enumerate(KFold(5, shuffle=True, random_state=seed).split(eids)):
        rng = np.random.RandomState(seed + fold)                    # reproducible negatives (original was unseeded)
        trainval_idx = rng.permutation(trainval_idx)
        n_val = int(len(trainval_idx) * 0.1)
        val_idx, train_idx = trainval_idx[:n_val], trainval_idx[n_val:]

        # disjoint negatives, sampled WITHOUT replacement (original sampled with replacement -> duplicates across splits)
        neg_sel = rng.choice(len(neg_u), size=min(E, len(neg_u)), replace=False)
        a, b = len(train_idx), len(train_idx) + len(val_idx)
        tn_ = (neg_u[neg_sel[:a]], neg_v[neg_sel[:a]])
        vn_ = (neg_u[neg_sel[a:b]], neg_v[neg_sel[a:b]])
        te_n = (neg_u[neg_sel[b:b + len(test_idx)]], neg_v[neg_sel[b:b + len(test_idx)]])

        hold_val = np.unique(np.concatenate([val_idx, reverse_eids(G_dgl, val_idx)]))
        hold_test = np.unique(np.concatenate([test_idx, reverse_eids(G_dgl, test_idx)]))
        train_g = dgl.remove_edges(G_dgl, np.unique(np.concatenate([hold_val, hold_test])).tolist())   # val/test: train edges only
        test_g = dgl.remove_edges(G_dgl, hold_test.tolist())                                           # test: train+val edges
        train_g, test_g = train_g.to(device), test_g.to(device)

        mk = lambda uu, vv: dgl.graph((torch.as_tensor(uu), torch.as_tensor(vv)), num_nodes=N).to(device)
        tp_g, tn_g = mk(u[train_idx], v[train_idx]), mk(*tn_)
        vp_g, vn_g = mk(u[val_idx], v[val_idx]), mk(*vn_)
        ep_g, en_g = mk(u[test_idx], v[test_idx]), mk(*te_n)

        torch.manual_seed(seed + fold)
        model = GATModel(node_features.shape[1], out_feats=args.out_feats, num_layers=args.num_layers,
                         num_heads=args.num_heads, feat_drop=args.feat_drop, attn_drop=args.attn_drop,
                         do_train=True).to(device)
        pred = MLPPredictor(args.input_size, args.hidden_size).to(device)
        criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
        opt = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.lr)
        sched = StepLR(opt, step_size=1000, gamma=0.1)
        feat_tr, feat_te = train_g.ndata["feat"], test_g.ndata["feat"]

        best = dict(auc=-1.0, ep=-1, state=None); bad = 0
        for e in range(args.epochs):
            model.train(); pred.train()
            h = model(train_g, feat_tr)
            ps, ns = pred(tp_g, h), pred(tn_g, h)
            loss = criterion(torch.cat([ps, ns]), torch.cat([torch.ones_like(ps), torch.zeros_like(ns)]))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()

            model.eval(); pred.eval()
            with torch.no_grad():
                h = model(train_g, feat_tr)
                vp, vn = to_prob(pred(vp_g, h)), to_prob(pred(vn_g, h))
            val_auc = roc_auc_score(np.r_[np.ones(len(vp)), np.zeros(len(vn))], np.r_[vp, vn])
            if val_auc > best["auc"]:
                best = dict(auc=val_auc, ep=e, state=(copy.deepcopy(model.state_dict()), copy.deepcopy(pred.state_dict()))); bad = 0
            else:
                bad += 1
            if e % 5 == 0:
                print(f"Fold {fold + 1} | Epoch {e} | Loss {loss.item():.4f} | Val AUC {val_auc:.4f}")
            if bad >= patience:
                break

        model.load_state_dict(best["state"][0]); pred.load_state_dict(best["state"][1]); model.eval(); pred.eval()
        with torch.no_grad():
            h = model(train_g, feat_tr)
            vp, vn = to_prob(pred(vp_g, h)), to_prob(pred(vn_g, h))
            thr = val_f1_threshold(np.r_[np.ones(len(vp)), np.zeros(len(vn))], np.r_[vp, vn])   # threshold from VALIDATION only
            h_t = model(test_g, feat_te)
            tpos, tneg = to_prob(pred(ep_g, h_t)), to_prob(pred(en_g, h_t))
        y = np.r_[np.ones(len(tpos)), np.zeros(len(tneg))]; p = np.r_[tpos, tneg]
        row = dict(Fold=fold + 1, best_epoch=best["ep"], val_auc=best["auc"], **lp_metrics(y, p, 0.5, thr, tpos, tneg))
        fold_rows.append(row); fold_curves.append((y, p))
        fold_states.append((copy.deepcopy(model.state_dict()), copy.deepcopy(pred.state_dict())))
        print(f"Fold {fold + 1} test | AUC {row['auc']:.4f} AP {row['ap']:.4f} | @0.5: F1 {row['f1@fixed']:.4f} "
              f"P {row['precision@fixed']:.4f} R {row['recall@fixed']:.4f} | @val-thr({thr:.3f}): F1 {row['f1@valF1']:.4f} "
              f"P {row['precision@valF1']:.4f} R {row['recall@valF1']:.4f} MCC {row['mcc@valF1']:.4f}")

    results = pd.DataFrame(fold_rows)
    results.to_csv(os.path.join(output_path, f"all_folds_results_{tag}.csv"), index=False)
    print("\n5-fold mean +/- std"); print(results.drop(columns="Fold").agg(["mean", "std"]).T.round(4).to_string())
    plot_roc_curves(fold_curves, os.path.join(output_path, f"roc_curve_5_fold_{tag}.png"))
    plot_pr_curves(fold_curves, os.path.join(output_path, f"pr_curve_5_fold_{tag}.png"))

    # ===== novel miRNA-disease predictions: score CANDIDATE pairs (not already known edges) =====
    names = [node_id_to_name[i] for i in range(N)]
    mir_ids = np.array([i for i, n in enumerate(names) if is_mirna(n)])
    dis_ids = np.array([i for i, n in enumerate(names) if is_cancer(n)])
    print(f"\nmiRNA nodes={len(mir_ids)}  cancer-disease nodes={len(dis_ids)}")
    cu, cv = np.repeat(mir_ids, len(dis_ids)), np.tile(dis_ids, len(mir_ids))
    cu_t, cv_t = torch.as_tensor(cu), torch.as_tensor(cv)
    known = (G_dgl.has_edges_between(cu_t, cv_t) | G_dgl.has_edges_between(cv_t, cu_t)).numpy()
    cu, cv = cu[~known], cv[~known]
    print(f"candidate pairs={len(cu):,} (known edges excluded: {int(known.sum()):,})")

    G_full = G_dgl.to(device); feat_full = G_full.ndata["feat"]
    score_sum = np.zeros(len(cu))
    for ms, ps_ in fold_states:                                     # ensemble of the 5 fold models
        model.load_state_dict(ms); pred.load_state_dict(ps_); model.eval(); pred.eval()
        with torch.no_grad():
            h = model(G_full, feat_full)
            for s in range(0, len(cu), 500_000):                    # both orientations: predictor may be order-sensitive
                sl = slice(s, s + 500_000)
                g1 = dgl.graph((torch.as_tensor(cu[sl]), torch.as_tensor(cv[sl])), num_nodes=N).to(device)
                g2 = dgl.graph((torch.as_tensor(cv[sl]), torch.as_tensor(cu[sl])), num_nodes=N).to(device)
                score_sum[sl] += 0.5 * (to_prob(pred(g1, h)) + to_prob(pred(g2, h)))
    novel = pd.DataFrame({"miRNA": [names[i] for i in cu], "disease": [names[i] for i in cv], "score": score_sum / len(fold_states)})
    novel = novel.sort_values("score", ascending=False).reset_index(drop=True)
    novel["rank"] = np.arange(1, len(novel) + 1); novel["percentile"] = novel["rank"] / len(novel)
    novel.to_csv(os.path.join(output_path, f"novel_predictions_{tag}.csv"), index=False)

    # ---- literature check (data/merged_all_data.csv), matched on normalised names ----
    lit = pd.read_csv("data/merged_all_data.csv")
    lit["k_m"], lit["k_d"] = lit["miRNA"].map(norm_name), lit["disease"].map(norm_name)
    novel["k_m"], novel["k_d"] = novel["miRNA"].map(norm_name), novel["disease"].map(norm_name)
    node_m, node_d = {norm_name(names[i]) for i in mir_ids}, {norm_name(names[i]) for i in dis_ids}
    in_nodes = lit["k_m"].isin(node_m) & lit["k_d"].isin(node_d)
    matched = lit.merge(novel, on=["k_m", "k_d"], suffixes=("", "_pred"))
    print(f"\nliterature rows={len(lit):,} | both entities present as graph nodes={int(in_nodes.sum()):,} "
          f"| found among NOVEL candidates={len(matched):,}")
    print(f"disease names in literature file not in graph (first 10): "
          f"{sorted(set(lit.loc[~lit['k_d'].isin(node_d), 'disease']))[:10]}")
    if len(matched):
        print(f"median percentile of literature-confirmed pairs among all candidates: {matched['percentile'].median():.4f}")
    matched.drop(columns=["k_m", "k_d"]).to_csv(os.path.join(output_path, f"top_cancer_u_{tag}.csv"), index=False)

    # ---- breast-cancer subset on novel predictions ----
    hmdd = pd.read_csv("data/hmdd_alldata_v4_updated.csv")
    bc_mirnas = set(hmdd.loc[hmdd["description"].str.contains("breast cancer", case=False, na=False), "miRNA"])
    bc = novel[novel["miRNA"].isin(bc_mirnas) & novel["disease"].str.contains("breast", case=False, na=False)]
    bc.drop(columns=["k_m", "k_d"]).to_csv(os.path.join(output_path, f"top_breast_cancer_{tag}.csv"), index=False)
    return results

 