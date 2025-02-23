## (kg39) ericsali@erics-MacBook-Pro-4 gnn_pathways % python gat/__pertag_driver_gene_prediction_chebnet_gpu_usage_pass_distr_2048.py --model_type ACGNN --net_type CPDB --score_threshold 0.99 --hidden_feats 1024 --learning_rate 0.001 --num_epochs 105
## (kg39) ericsali@erics-MacBook-Pro-4 gnn_pathways % python gat/__pertag_driver_gene_prediction_chebnet_gpu_usage_pass_distr_2048.py --model_type ACGNN --net_type HIPPIE --score_threshold 0.99 --in_feats 2048 --hidden_feats 256 --learning_rate 0.001 --num_epochs 105

## (kg39) ericsali@erics-MacBook-Pro-4 gnn_pathways % python gat/__pertag_driver_gene_prediction_chebnet_gpu_usage_pass_distr_2048.py --model_type ACGNN --score_threshold 0.99 --learning_rate 0.001 --num_epochs 204
## p_value in average predicted score
## (kg39) ericsali@erics-MacBook-Pro-4 gnn_pathways % python gat/__pertag_driver_gene_prediction_chebnet_gpu_usage_pass_distr_2048.py --model_type ACGNN --net_type STRING --score_threshold 0.99 --learning_rate 0.001 --num_epochs 505
## python gat/_gene_label_prediction_tsne_pertag.py --model_type Chebnet --net_type pathnet --score_threshold 0.4 --learning_rate 0.001 --num_epochs 65 
## (kg39) ericsali@erics-MBP-4 gnn_pathways % python gat/_gene_label_prediction_tsne_sage.py --model_type EMOGI --net_type ppnet --score_threshold 0.5 --learning_rate 0.001 --num_epochs 100 
## (kg39) ericsali@erics-MBP-4 gnn_pathways % python gat/_gene_label_prediction_tsne_pertag.py --model_type ATTAG --net_type ppnet --score_threshold 0.9 --learning_rate 0.001 --num_epochs 201

import json
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy.stats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from torch_geometric.nn import GCNConv
from .models import ACGNN, HGDC, EMOGI, MTGCN, GCN, GAT, GraphSAGE, GIN, Chebnet, FocalLoss

def choose_model(model_type, in_feats, hidden_feats, out_feats):
    if model_type == 'GraphSAGE':
        return GraphSAGE(in_feats, hidden_feats, out_feats)
    elif model_type == 'GAT':
        return GAT(in_feats, hidden_feats, out_feats, num_heads=1)
    elif model_type == 'GCN':
        return GCN(in_feats, hidden_feats, out_feats)
    elif model_type == 'GIN':
        return GIN(in_feats, hidden_feats, out_feats)
    elif model_type == 'HGDC':
        return GAT(in_feats, hidden_feats, out_feats, num_heads=1)
    elif model_type == 'EMOGI':
        return GAT(in_feats, hidden_feats, out_feats, num_heads=1)
    elif model_type == 'MTGCN':
        return GCN(in_feats, hidden_feats, out_feats)
    elif model_type == 'Chebnet':
        return ATTAG(in_feats, hidden_feats, out_feats)
    elif model_type == 'ACGNN':
        return ACGNN(in_feats, hidden_feats, out_feats)
    else:
        raise ValueError("Invalid model type. Choose from ['GraphSAGE', 'GAT', 'EMOGI', 'HGDC', 'MTGCN', 'GCN', 'GIN', 'Chebnet', 'ACGNN'].")

def save_and_plot_results_no_error_bar_pass(predicted_above, predicted_below, degrees_above, degrees_below, avg_above, avg_below, args):

    # Save predictions and degrees
    output_dir = 'gat/results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    def save_csv(data, filename, header):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(header)
            csvwriter.writerows(data)
        print(f"File saved: {filepath}")

    save_csv(predicted_above, f'{args.model_type}_above_threshold.csv', ['Gene', 'Score'])
    save_csv(predicted_below, f'{args.model_type}_below_threshold.csv', ['Gene', 'Score'])
    save_csv(degrees_above.items(), f'{args.model_type}_degrees_above.csv', ['Gene', 'Degree'])
    save_csv(degrees_below.items(), f'{args.model_type}_degrees_below.csv', ['Gene', 'Degree'])

    # Degree comparison barplot
    data = pd.DataFrame({
        'Threshold': ['Above', 'Below'],
        'Average Degree': [avg_above, avg_below]
    })
    plt.figure(figsize=(8, 6))
    sns.barplot(data=data, x='Threshold', y='Average Degree', palette="viridis")
    plt.title('Average Degree Comparison')
    plt.savefig(os.path.join(output_dir, f'{args.model_type}_degree_comparison.png'))
    plt.show()

def plot_roc_curve(labels, scores, filename):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", color="blue")
    plt.plot([0, 1], [0, 1], color="salmon", linestyle="--")
    plt.title("Receiver Operating Characteristic Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    ##plt.grid(alpha=0.4)
    plt.savefig(filename)
    plt.show()
    print(f"ROC Curve saved to {filename}")

def plot_pr_curve(labels, scores, filename):
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.4f})", color="green")
    ##plt.plot([0, 1], [1, 0], color="salmon", linestyle="--")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    ##plt.grid(alpha=0.4)
    plt.savefig(filename)
    plt.show()
    print(f"Precision-Recall Curve saved to {filename}")

def load_graph_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    nodes = {}
    edges = []
    labels = []
    embeddings = []

    for entry in data:
        source = entry["source"]["properties"]
        target = entry["target"]["properties"]
        relation = entry["relation"]["type"]

        # Add source node
        if source["name"] not in nodes:
            nodes[source["name"]] = len(nodes)
            embeddings.append(source["embedding"])
            labels.append(source.get("label", -1) if source.get("label") is not None else -1)

        # Add target node
        if target["name"] not in nodes:
            nodes[target["name"]] = len(nodes)
            embeddings.append(target["embedding"])
            labels.append(target.get("label", -1) if target.get("label") is not None else -1)

        # Add edge
        edges.append((nodes[source["name"]], nodes[target["name"]]))

    # Convert embeddings and labels to tensors
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return nodes, edges, embeddings_tensor, labels_tensor

def load_oncokb_genes(filepath):
    with open(filepath, 'r') as f:
        return set(line.strip() for line in f)

def plot_and_analyze_ori(args):
    # File path for the saved predictions
    csv_file_path = os.path.join(
        'gat/results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_predicted_scores_threshold{args.score_threshold}_epo{args.num_epochs}.csv'
    )

    results = []

    # Read the CSV file
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            node_name, score, label = row
            score = float(score)
            label = int(label)
            results.append((node_name, score, label))
            
            # Check if label is 0 and print the row
            if label == 0:
                print(f"Node Name: {node_name}, Score: {score}, Label: {label}")


    # Extract scores and labels
    scores = np.array([row[1] for row in results])
    labels = np.array([row[2] for row in results])
    

    # Define group labels in the desired order
    group_labels = [1, 2, 0, 3]
    average_scores = []

    # Calculate average scores for each group
    for label in group_labels:
        group_scores = scores[labels == label]
        avg_score = np.mean(group_scores) if len(group_scores) > 0 else 0.0
        average_scores.append(avg_score)

    # Perform statistical tests to calculate p-values
    p_values = {}
    comparisons = [(1, 2), (1, 0), (1, 3)]  # Pairs to compare
    for group1, group2 in comparisons:
        scores1 = scores[labels == group1]
        scores2 = scores[labels == group2]
        if len(scores1) > 1 and len(scores2) > 1:
            _, p_value = ttest_ind(scores1, scores2, equal_var=False)
            p_values[(group1, group2)] = p_value
        else:
            p_values[(group1, group2)] = np.nan

    # Save average scores and p-values to another CSV
    avg_csv_path = os.path.join(
        'gat/results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_group_avg_scores_pvalues_epo{args.num_epochs}_2048.csv'
    )
    os.makedirs(os.path.dirname(avg_csv_path), exist_ok=True)
    with open(avg_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Group Label', 'Average Score', 'Comparison', 'P-Value'])
        for label, avg_score in zip(group_labels, average_scores):
            writer.writerow([f'Group {label}', avg_score, '', ''])
        for (group1, group2), p_value in p_values.items():
            writer.writerow(['', '', f'Group {group1} vs Group {group2}', p_value])
    print(f"Average scores and p-values saved to {avg_csv_path}")

    # Plot the bar chart
    plt.figure(figsize=(8, 6))
    bars = plt.bar(range(len(group_labels)), average_scores,
                   color=['green', 'red', 'blue', 'orange'], edgecolor='black', alpha=0.8)
    for bar, avg_score in zip(bars, average_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{avg_score:.4f}',
                 ha='center', va='bottom', fontsize=12)
    plt.xticks(range(len(group_labels)), ['Ground-truth (1)', 'Predicted (2)', 'Non-driver (0)', 'Other (3)'])
    plt.xlabel('Gene Groups', fontsize=14)
    plt.ylabel('Average Score', fontsize=14)
    plt.title('Average Scores for Each Gene Group', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    bar_plot_path = os.path.join(
        'gat/results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_group_avg_scores_barplot_epo{args.num_epochs}_2048.png'
    )
    plt.savefig(bar_plot_path)
    print(f"Bar plot saved to {bar_plot_path}")
    plt.show()

    '''
    
    # Convert gene_labels to NumPy array
    ##gene_labels = gene_labels.cpu().numpy()

    # Calculate average scores for each group in order of label 1, 2, 0, 3
    group_labels = [1, 2, 0, 3]  # Define the groups in the desired order
    average_scores = [np.mean(scores[gene_labels == label]) if (gene_labels == label).sum() > 0 else 0.0
                    for label in group_labels]

    p_values = {}
    for g1, g2 in [(1, 2), (1, 0), (1, 3)]:
        scores1 = scores[gene_labels == g1]
        scores2 = scores[gene_labels == g2]
        p_values[(g1, g2)] = ttest_ind(scores1, scores2, equal_var=False).pvalue if len(scores1) > 1 and len(scores2) > 1 else np.nan

    # Save averages and p-values
    avg_csv_path = os.path.join('gat/results/gene_prediction/',
                                f'{args.model_type}_avg_scores_pvalues.csv')
    with open(avg_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Group Label', 'Average Score', 'Comparison', 'P-Value'])
        for label, avg_score in zip(group_labels, average_scores):
            writer.writerow([f'Group {label}', avg_score, '', ''])
        for (g1, g2), p_val in p_values.items():
            writer.writerow(['', '', f'Group {g1} vs Group {g2}', p_val])
    print(f"Average scores and p-values saved to {avg_csv_path}")

    # Plot average scores
    plt.figure(figsize=(8, 6))
    bars = plt.bar(range(len(group_labels)), average_scores,
                   color=['green', 'red', 'blue', 'orange'], edgecolor='black', alpha=0.8)
    for bar, avg in zip(bars, average_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{avg:.4f}',
                 ha='center', va='bottom', fontsize=12)
    plt.xticks(range(len(group_labels)), ['Ground-truth (1)', 'Predicted (2)', 'Non-driver (0)', 'Other (3)'])
    plt.xlabel('Gene Groups', fontsize=14)
    plt.ylabel('Average Score', fontsize=14)
    plt.title('Average Scores by Gene Group', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    bar_plot_path = os.path.join('gat/results/gene_prediction/',
                                 f'{args.model_type}_group_avg_scores.png')
    plt.savefig(bar_plot_path)
    print(f"Bar plot saved to {bar_plot_path}")
    plt.show()'''

def plot_and_analyze(args):
    # File path for the saved predictions
    csv_file_path = os.path.join(
        'gat/results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_predicted_scores_threshold{args.score_threshold}_epo{args.num_epochs}.csv'
    )

    results = []

    # Read the CSV file
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            node_name, score, label = row
            score = float(score)
            label = int(label)
            results.append((node_name, score, label))
    
    # Extract scores and labels as NumPy arrays
    scores = np.array([row[1] for row in results])
    labels = np.array([row[2] for row in results])
    
    # Define group labels in the desired order
    group_labels = [1, 2, 0, 3]
    average_scores = []

    # Calculate average scores for each group
    for label in group_labels:
        group_scores = scores[labels == label]
        avg_score = np.mean(group_scores) if len(group_scores) > 0 else 0.0
        average_scores.append(avg_score)

    # Perform statistical tests to calculate p-values
    p_values = {}
    comparisons = [(1, 2), (1, 0), (1, 3), (2, 3)]  # Added (2 vs. 3)
    for group1, group2 in comparisons:
        scores1 = scores[labels == group1]
        scores2 = scores[labels == group2]
        if len(scores1) > 1 and len(scores2) > 1:
            _, p_value = ttest_ind(scores1, scores2, equal_var=False)
            p_values[(group1, group2)] = p_value
        else:
            p_values[(group1, group2)] = np.nan

    # Save average scores and p-values to another CSV
    avg_csv_path = os.path.join(
        'gat/results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_group_avg_scores_pvalues_threshold{args.score_threshold}_epo{args.num_epochs}_2048.csv'
    )
    os.makedirs(os.path.dirname(avg_csv_path), exist_ok=True)
    with open(avg_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Group Label', 'Average Score', 'Comparison', 'P-Value'])
        for label, avg_score in zip(group_labels, average_scores):
            writer.writerow([f'Group {label}', avg_score, '', ''])
        for (group1, group2), p_value in p_values.items():
            writer.writerow(['', '', f'Group {group1} vs Group {group2}', p_value])
    
    print(f"Average scores and p-values saved to {avg_csv_path}")

    # Plot the bar chart
    plt.figure(figsize=(8, 6))
    bars = plt.bar(range(len(group_labels)), average_scores,
                   color=['green', 'red', 'blue', 'orange'], edgecolor='black', alpha=0.8)
    for bar, avg_score in zip(bars, average_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{avg_score:.4f}',
                 ha='center', va='bottom', fontsize=12)
    plt.xticks(range(len(group_labels)), ['Ground-truth (1)', 'Predicted (2)', 'Non-driver (0)', 'Other (3)'])
    plt.xlabel('Gene Groups', fontsize=14)
    plt.ylabel('Average Score', fontsize=14)
    plt.title('Average Scores for Each Gene Group', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the bar plot
    bar_plot_path = os.path.join(
        'gat/results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_group_avg_scores_barplot_threshold{args.score_threshold}_epo{args.num_epochs}_2048.png'
    )
    plt.savefig(bar_plot_path)
    print(f"Bar plot saved to {bar_plot_path}")
    plt.show()

def save_and_plot_results(predicted_above, predicted_below, degrees_above, degrees_below, avg_above, avg_below, avg_error_above, avg_error_below, args):

    # Save predictions and degrees
    output_dir = 'gat/results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    def save_csv(data, filename, header):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(header)
            csvwriter.writerows(data)
        print(f"File saved: {filepath}")

    save_csv(predicted_above, f'{args.model_type}_{args.net_type}_above_threshold.csv', ['Gene', 'Score'])
    save_csv(predicted_below, f'{args.model_type}_{args.net_type}_below_threshold.csv', ['Gene', 'Score'])
    save_csv(degrees_above.items(), f'{args.model_type}_{args.net_type}_degrees_above.csv', ['Gene', 'Degree'])
    save_csv(degrees_below.items(), f'{args.model_type}_{args.net_type}_degrees_below.csv', ['Gene', 'Degree'])

    # Degree comparison barplot with error bars
    data = pd.DataFrame({
        'Threshold': ['Above', 'Below'],
        'Average Degree': [avg_above, avg_below],
        'Error': [avg_error_above, avg_error_below]  # Add error values
    })
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(data['Threshold'], data['Average Degree'], yerr=data['Error'], capsize=5, color=['green', 'red'], edgecolor='black', alpha=0.8)

    # Add error bars explicitly (optional, can be done directly in the bar plot)
    for bar, error in zip(bars, data['Error']):
        plt.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(), yerr=error, fmt='none', color='black', capsize=5, linestyle='--')

    plt.title('Average Degree Comparison with Error Bars')
    plt.savefig(os.path.join(output_dir, f'{args.model_type}_{args.net_type}_degree_comparison_with_error_bars.png'))
    plt.show()
    print(f"Degree comparison plot saved to {os.path.join(output_dir, f'{args.model_type}_{args.net_type}_degree_comparison_with_error_bars.png')}")
