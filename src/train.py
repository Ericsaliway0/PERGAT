## (kg39) ericsali@erics-MacBook-Pro-4 gnn_pathways % python gat/__pertag_driver_gene_prediction_chebnet_gpu_usage_pass_distr_2048.py --model_type ACGNN --net_type CPDB --score_threshold 0.99 --hidden_feats 1024 --learning_rate 0.001 --num_epochs 105
## (kg39) ericsali@erics-MacBook-Pro-4 gnn_pathways % python gat/__pertag_driver_gene_prediction_chebnet_gpu_usage_pass_distr_2048.py --model_type ACGNN --net_type HIPPIE --score_threshold 0.99 --in_feats 2048 --hidden_feats 256 --learning_rate 0.001 --num_epochs 105

## (kg39) ericsali@erics-MacBook-Pro-4 gnn_pathways % python gat/__pertag_driver_gene_prediction_chebnet_gpu_usage_pass_distr_2048.py --model_type ACGNN --score_threshold 0.99 --learning_rate 0.001 --num_epochs 204
## p_value in average predicted score
## (kg39) ericsali@erics-MacBook-Pro-4 gnn_pathways % python gat/__pertag_driver_gene_prediction_chebnet_gpu_usage_pass_distr_2048.py --model_type ACGNN --net_type STRING --score_threshold 0.99 --learning_rate 0.001 --num_epochs 505
## python gat/_gene_label_prediction_tsne_pertag.py --model_type Chebnet --net_type pathnet --score_threshold 0.4 --learning_rate 0.001 --num_epochs 65 
## (kg39) ericsali@erics-MBP-4 gnn_pathways % python gat/_gene_label_prediction_tsne_sage.py --model_type EMOGI --net_type ppnet --score_threshold 0.5 --learning_rate 0.001 --num_epochs 100 
## (kg39) ericsali@erics-MBP-4 gnn_pathways % python gat/_gene_label_prediction_tsne_pertag.py --model_type ATTAG --net_type ppnet --score_threshold 0.9 --learning_rate 0.001 --num_epochs 201

import dgl
import torch
import torch.nn as nn
import numpy as np
import os
import time
import psutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import scipy.stats
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from matplotlib.lines import Line2D
from scipy.stats import ttest_ind
from torch_geometric.nn import GCNConv
from .models import ACGNN, HGDC, EMOGI, MTGCN, GCN, GAT, GraphSAGE, GIN, Chebnet, FocalLoss
from src.utils import (choose_model, plot_roc_curve, plot_pr_curve, load_graph_data, 
                       load_oncokb_genes, plot_and_analyze, save_and_plot_results)


def train(args):
    # Initialize metrics
    epoch_times = []
    cpu_usages = []
    gpu_usages = []
    data_path = os.path.join('gat/data/multiomics_meth/', f'{args.net_type}_omics_ppi_embeddings_2048.json')
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1  # Mask for labeled nodes
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)  # Mask for testing on all nodes
    graph = dgl.add_self_loop(graph)
    in_feats = embeddings.shape[1]
    hidden_feats = args.hidden_feats
    out_feats = 1  # Binary classification (driver gene or not)

    # Model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define output directory
    output_dir = 'gat/results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)

    # File path for saving model architecture
    model_csv_path = os.path.join(output_dir, f'{args.model_type}_model_structure.csv')

    # Choose the model
    model = choose_model(args.model_type, in_feats, hidden_feats, out_feats)
    model = model.to(args.device)

    loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = FocalLoss(alpha=0.25, gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Move data to device
    graph = graph.to(device)
    features = graph.ndata['feat'].to(device)
    labels = graph.ndata['label'].to(device).float()  # BCE loss expects float labels
    train_mask = graph.ndata['train_mask'].to(device)
    test_mask = graph.ndata['test_mask'].to(device)
        
    # Training loop with progress bar
    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch"):
        epoch_start = time.time()

        # Measure CPU usage
        cpu_usage = psutil.cpu_percent(interval=None)
        cpu_usages.append(cpu_usage)

        # Measure GPU usage (if CUDA available)
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated(device) / 2048 ** 2  # Convert bytes to MB
        else:
            gpu_usage = 0.0  # No GPU available
        gpu_usages.append(gpu_usage)
        
        model.train()
        logits = model(graph, features).squeeze()
        loss = loss_fn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record time per epoch
        epoch_end = time.time()
        epoch_times.append(epoch_end - epoch_start)

        # Optionally display current metrics
        tqdm.write(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {loss.item():.4f}, CPU: {cpu_usage}%, GPU: {gpu_usage:.2f} MB")

    # Testing and predictions
    model.eval()
    with torch.no_grad():
        logits = model(graph, features).squeeze()
        scores = torch.sigmoid(logits).cpu().numpy()
        print("Predicted Scores:", scores[test_mask.cpu().numpy()])


    
    # Convert epoch_times to floats during summation
    total_time = sum(epoch_times)  # Total time in seconds
    average_time_per_epoch = total_time / args.num_epochs  # Average time per epoch
    # Calculate average memory usage for CPU and GPU
    average_cpu_usage = sum(cpu_usages) / args.num_epochs  # CPU usage in MB
    average_gpu_usage = sum(gpu_usages) / args.num_epochs  # GPU usage in MB
    
    save_model_details(model, args, model_csv_path, in_feats, hidden_feats, out_feats)
    label_scores = save_predicted_scores(scores, labels, nodes, args)
    save_average_scores(label_scores, args)
    plot_average_scores(label_scores, args)
    plot_score_distributions(label_scores, args)
    save_performance_metrics(epoch_times, cpu_usages, gpu_usages, args)
    # After calculating total_time, average_time_per_epoch, etc.
    save_overall_metrics(total_time, average_time_per_epoch, average_cpu_usage, average_gpu_usage, args, output_dir)


    ##################################################################
    # Rank nodes by scores for only non-labeled nodes
    node_names = list(nodes.keys())
    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]  # Indices of non-labeled nodes
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)


    process_predictions(ranking, args, "gat/data/796_drivers.txt", "gat/data/oncokb_1172.txt", "gat/data/ongene_803.txt", "gat/data/ncg_8886.txt", "gat/data/intogen_23444.txt", node_names, non_labeled_nodes)

    # Load driver and reference gene sets


    # Calculate statistics
    non_labeled_nodes_count = len(non_labeled_nodes)
    ground_truth_driver_nodes = [i for i, label in enumerate(labels) if label == 1]
    ground_truth_non_driver_nodes = [i for i, label in enumerate(labels) if label == 0]
    

    # Save both above and below threshold scores, sorted by scores in descending order
    predicted_driver_nodes_above_threshold = sorted(
        [(node_names[i], scores[i]) for i in non_labeled_nodes if scores[i] >= args.score_threshold],
        key=lambda x: x[1],
        reverse=True
    )
    predicted_driver_nodes_below_threshold = sorted(
        [(node_names[i], scores[i]) for i in non_labeled_nodes if scores[i] < args.score_threshold],
        key=lambda x: x[1],
        reverse=True
    )

    from collections import defaultdict
    import statistics

    # Get the ground truth driver gene indices and names
    ground_truth_driver_indices = [i for i, label in enumerate(labels) if label == 1]
    ground_truth_driver_names = {node_names[i] for i in ground_truth_driver_indices}

    # Save predictions (above and below threshold) to CSV
    output_file_above = os.path.join(
        'gat/results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_predicted_driver_genes_above_epo{args.num_epochs}_2048.csv'
    )
    output_file_below = os.path.join(
        'gat/results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_predicted_driver_genes_below_epo{args.num_epochs}_2048.csv'
    )

    with open(output_file_above, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Gene Name', 'Score'])  # Header row
        csvwriter.writerows(predicted_driver_nodes_above_threshold)

    with open(output_file_below, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Gene Name', 'Score'])  # Header row
        csvwriter.writerows(predicted_driver_nodes_below_threshold)

    print(f"Predicted driver genes (above threshold) saved to {output_file_above}")
    print(f"Predicted driver genes (below threshold) saved to {output_file_below}")

    # Calculate degrees for nodes above and below the threshold (connecting only to label 1 nodes)
    degree_counts_above = defaultdict(int)
    degree_counts_below = defaultdict(int)

    for src, dst in edges:
        src_name = node_names[src]
        dst_name = node_names[dst]

        # Count only connections to ground truth driver genes (label 1 nodes)
        if dst_name in ground_truth_driver_names:
            if src_name in [gene for gene, _ in predicted_driver_nodes_above_threshold]:
                degree_counts_above[src_name] += 1
            elif src_name in [gene for gene, _ in predicted_driver_nodes_below_threshold]:
                degree_counts_below[src_name] += 1

    # Sort degrees by degree count in descending order
    sorted_degree_counts_above = sorted(degree_counts_above.items(), key=lambda x: x[1], reverse=True)
    sorted_degree_counts_below = sorted(degree_counts_below.items(), key=lambda x: x[1], reverse=True)

    # Save degrees of predicted driver genes connecting to ground truth driver genes (above threshold)
    degree_output_file_above = os.path.join(
        'gat/results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_predicted_driver_gene_degrees_above_epo{args.num_epochs}_2048.csv'
    )
    with open(degree_output_file_above, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Predicted Driver Gene', 'Degree'])  # Header row
        csvwriter.writerows(sorted_degree_counts_above)
        ##csvwriter.writerow(['Average Degree', average_degree_above])  # Save average degree

    print(f"Degrees of predicted driver genes (above threshold) saved to {degree_output_file_above}")



    # Prepare DataFrame for nodes with degrees
    nodes_with_degrees = []

    # Above threshold
    for gene, degree in degree_counts_above.items():
        nodes_with_degrees.append({'Gene_Set': 'Above Threshold', 'Degree': degree})

    # Below threshold
    for gene, degree in degree_counts_below.items():
        nodes_with_degrees.append({'Gene_Set': 'Below Threshold', 'Degree': degree})

    # Convert to DataFrame
    nodes_with_degrees = pd.DataFrame(nodes_with_degrees)

    ##print(f"sorted_degree_counts_below: {sorted_degree_counts_below}")
    sorted_degree_counts_above_value = [value for _, value in sorted_degree_counts_above if value <= 20]
    sorted_degree_counts_below_value = [value for _, value in sorted_degree_counts_below if value <= 20]
    
    output_above_file = os.path.join('gat/results/gene_prediction/', f'{args.model_type}_{args.net_type}_output_above_file_epo{args.num_epochs}_2048.csv')
    output_below_file = os.path.join('gat/results/gene_prediction/', f'{args.model_type}_{args.net_type}_output_below_file_epo{args.num_epochs}_2048.csv')

    calculate_and_save_prediction_stats(non_labeled_nodes, labels, node_names, scores, args)

    plot_degree_distributions(sorted_degree_counts_above_value, sorted_degree_counts_below_value, args, output_dir)

    generate_kde_and_curves(logits, node_names, degree_counts_above, degree_counts_below, labels, train_mask, args)

    plot_model_performance(args)




def load_gene_set(file_path):
    """
    Load a gene list from a file and return as a set.
    
    Args:
    - file_path: Path to the file containing genes, one per line.
    
    Returns:
    - Set of gene names.
    """
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f)

def save_predictions_to_csv(predicted_genes, output_dir, model_type, net_type, num_epochs):
    """
    Save the predicted genes with their sources to a CSV file.
    
    Args:
    - predicted_genes: List of tuples (gene, score, sources) to save.
    - output_dir: Directory to save the CSV file.
    - model_type, net_type, num_epochs: For naming the output file.
    """
    os.makedirs(output_dir, exist_ok=True)
    predicted_genes_csv_path = os.path.join(output_dir, f'{model_type}_{net_type}_predicted_driver_genes_epo{num_epochs}_2048.csv')
    df_predictions = pd.DataFrame(predicted_genes, columns=["Gene", "Score", "Confirmed Sources"])
    df_predictions.to_csv(predicted_genes_csv_path, index=False)
    print(f"Predicted driver genes with confirmed sources saved to {predicted_genes_csv_path}")

def save_confirmed_predictions_to_csv(confirmed_predictions, output_dir, model_type, net_type, num_epochs):
    """
    Save confirmed predicted genes to a CSV file.
    
    Args:
    - confirmed_predictions: List of tuples (gene, score, sources).
    - output_dir: Directory to save the CSV file.
    - model_type, net_type, num_epochs: For naming the output file.
    """
    confirmed_predictions_csv_path = os.path.join(output_dir, f'{model_type}_{net_type}_confirmed_predicted_genes_epo{num_epochs}_2048.csv')
    df_confirmed = pd.DataFrame(confirmed_predictions, columns=["Gene", "Score", "Source"])
    df_confirmed.to_csv(confirmed_predictions_csv_path, index=False)
    print(f"Confirmed predicted genes saved to {confirmed_predictions_csv_path}")

def save_predicted_known_drivers(predicted_driver_genes, output_dir, model_type, net_type, num_epochs):
    """
    Save predicted known cancer driver genes to a CSV file.
    
    Args:
    - predicted_driver_genes: List of predicted cancer driver genes.
    - output_dir: Directory to save the CSV file.
    - model_type, net_type, num_epochs: For naming the output file.
    """
    predicted_drivers_csv_path = os.path.join(output_dir, f'{model_type}_{net_type}_predicted_known_drivers_epo{num_epochs}_2048.csv')
    df = pd.DataFrame(predicted_driver_genes, columns=["Gene"])
    df.to_csv(predicted_drivers_csv_path, index=False)
    print(f"Predicted known driver genes saved to {predicted_drivers_csv_path}")

def process_predictions(ranking, args, drivers_file_path, oncokb_file_path, ongene_file_path, ncg_file_path, intogen_file_path, node_names, non_labeled_nodes):
    """
    Process and save the predicted driver genes, confirmed sources, and known drivers.
    
    Args:
    - ranking: List of tuples (gene, score) representing ranked predictions.
    - args: Argument object containing model and network type, and score threshold.
    - drivers_file_path, oncokb_file_path, ongene_file_path, ncg_file_path, intogen_file_path: Paths to the confirmation gene files.
    - node_names, non_labeled_nodes: Information about node names and indices for matching.
    """
    # Load data from the confirmation files
    oncokb_genes = load_gene_set(oncokb_file_path)
    ongene_genes = load_gene_set(ongene_file_path)
    ncg_genes = load_gene_set(ncg_file_path)
    intogen_genes = load_gene_set(intogen_file_path)

    # Threshold for the score
    score_threshold = args.score_threshold

    confirmed_predictions = []
    predicted_genes = []

    for node, score in ranking:
        if score >= score_threshold:
            sources = []  # Accumulate sources confirming the gene
            if node in oncokb_genes:
                sources.append("OncoKB")
            if node in ongene_genes:
                sources.append("OnGene")
            if node in ncg_genes:
                sources.append("NCG")
            if node in intogen_genes:
                sources.append("IntOGen")
            if sources:  # If the gene is confirmed by at least one source
                confirmed_predictions.append((node, score, ", ".join(sources)))
            predicted_genes.append((node, score, ", ".join(sources) if sources else ""))

    # Save predictions to a CSV file
    save_predictions_to_csv(predicted_genes, 'gat/results/gene_prediction/', args.model_type, args.net_type, args.num_epochs)
    save_confirmed_predictions_to_csv(confirmed_predictions, 'gat/results/gene_prediction/', args.model_type, args.net_type, args.num_epochs)

    # Load known cancer driver genes
    with open(drivers_file_path, 'r') as f:
        known_drivers = set(line.strip() for line in f)

    # Collect predicted cancer driver genes that match the known drivers
    predicted_driver_genes = [node_names[i] for i in non_labeled_nodes if node_names[i] in known_drivers]

    # Save the predicted known cancer driver genes to a CSV file
    save_predicted_known_drivers(predicted_driver_genes, 'gat/results/gene_prediction/', args.model_type, args.net_type, args.num_epochs)

def save_overall_metrics(total_time, average_time_per_epoch, average_cpu_usage, average_gpu_usage, args, output_dir):
    """
    Save the overall performance metrics to a CSV file.

    Args:
    - total_time: Total training time in seconds.
    - average_time_per_epoch: Average time per epoch in seconds.
    - average_cpu_usage: Average CPU usage in MB.
    - average_gpu_usage: Average GPU usage in MB.
    - args: Argument object containing model and network type.
    - output_dir: The directory where the results will be saved.
    """
    # Save overall metrics
    df_overall_metrics = pd.DataFrame([{
        "Model Type": args.model_type,
        "Total Time": f"{total_time:.4f}s",
        "Average Time per Epoch": f"{average_time_per_epoch:.4f}s",
        "Average CPU Usage (MB)": f"{average_cpu_usage:.2f}",
        "Average GPU Usage (MB)": f"{average_gpu_usage:.2f}"
    }])
    
    # Define path to save the CSV
    overall_metrics_csv_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_overall_performance_epo{args.num_epochs}_2048.csv')
    
    # Save to CSV
    df_overall_metrics.to_csv(overall_metrics_csv_path, index=False)
    print(f"Overall performance metrics saved to {overall_metrics_csv_path}")

def calculate_and_save_prediction_stats(non_labeled_nodes, labels, node_names, scores, args):
    """
    Calculate prediction statistics and save them to a CSV file.

    Parameters:
    - non_labeled_nodes: List of nodes without labels
    - labels: List of ground truth labels for the nodes
    - node_names: List of node names corresponding to the nodes
    - scores: List of predicted scores for the nodes
    - args: Arguments containing model and network type, score threshold, and number of epochs
    """
    # Calculate statistics
    non_labeled_nodes_count = len(non_labeled_nodes)
    ground_truth_driver_nodes = [i for i, label in enumerate(labels) if label == 1]
    ground_truth_non_driver_nodes = [i for i, label in enumerate(labels) if label == 0]

    predicted_driver_nodes = [node_names[i] for i in non_labeled_nodes if scores[i] >= args.score_threshold]

    # Prepare data to save to CSV
    stats_output_file = os.path.join('gat/results/gene_prediction/', f'{args.model_type}_{args.net_type}_prediction_stats_{args.num_epochs}.csv')
    
    with open(stats_output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Non-Labeled Nodes Count', 'Driver Genes', 'Non-Driver Genes', 'Total Testing Nodes', 'Predicted Driver Genes'])
        csvwriter.writerow([
            non_labeled_nodes_count,
            len(ground_truth_driver_nodes),
            len(ground_truth_non_driver_nodes),
            len(ground_truth_driver_nodes) + len(ground_truth_non_driver_nodes),
            len(predicted_driver_nodes)
        ])

    print(f"Prediction statistics saved to {stats_output_file}")

def plot_degree_distributions(sorted_degree_counts_above, sorted_degree_counts_below, args, output_dir):
    """
    Generates a box plot comparing interaction degrees of PCGs vs. Other Genes with KCGs.
    
    Parameters:
    - sorted_degree_counts_above: List of degrees for PCGs.
    - sorted_degree_counts_below: List of degrees for other genes.
    - args: Arguments containing model and training configuration.
    - output_dir: Directory to save the plot.
    """

    print("Generating box plot for degree distributions...")

    degree_data = [sorted_degree_counts_above, sorted_degree_counts_below]

    plt.figure(figsize=(3, 4))

    # Create the box plot
    boxplot = plt.boxplot(
        degree_data,
        vert=True,
        patch_artist=True,  # Allows customization of box color
        flierprops=dict(marker='o', markerfacecolor='grey', markeredgecolor='grey', markersize=5, alpha=0.2),  # Outliers
        boxprops=dict(color='black'),  # Box border color
        medianprops=dict(color='blue', linewidth=2),  # Median line style
        whiskerprops=dict(color='black', linewidth=1.5),  # Whiskers
        capprops=dict(color='black', linewidth=1.5)  # Caps
    )

    # Customize frame
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # Remove top frame line
    ax.spines['right'].set_visible(False)  # Remove right frame line

    # X-axis labels
    plt.xticks([1, 2], ['PCGs', 'Other'], fontsize=8)  
    plt.yticks(fontsize=8)
    plt.ylabel('Interaction Degrees with KCGs', fontsize=10, labelpad=10) 

    # Assign different colors to box plots
    colors = ['green', 'skyblue']
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_plot_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_degree_distributions_epo{args.num_epochs}_2048.png')
    plt.savefig(output_plot_path, bbox_inches='tight')

    plt.tight_layout()
    plt.show()
    print(f"Box plot saved to {output_plot_path}")

def generate_kde_and_curves(logits, node_names, degree_counts_above, degree_counts_below, labels, train_mask, args):
    """
    Generates KDE plot comparing ACGNN score ranks with KCG interaction ranks, 
    computes Spearman correlation, and saves the KDE plot.
    Also computes and saves ROC and PR curves.

    Parameters:
    - logits: Tensor of model outputs before applying sigmoid.
    - node_names: List of node names.
    - degree_counts_above, degree_counts_below: Dictionaries mapping nodes to degree counts.
    - labels: Ground truth labels.
    - train_mask: Boolean mask indicating training samples.
    - args: Arguments containing model and training configuration.
    - output_dir: Directory to save plots.
    """

    print("Preparing data for KDE plot...")
    
    # Convert logits to probabilities
    scores = torch.sigmoid(logits).cpu().numpy()

    # Compute degree ranks
    degrees = [
        degree_counts_above.get(node_names[i], 0) +
        degree_counts_below.get(node_names[i], 0)
        for i in range(len(node_names))
    ]

    # Create DataFrame
    plot_data = pd.DataFrame({
        "Prob_pos_ranked": pd.Series(scores).rank(),
        "Degree_ranked": pd.Series(degrees).rank()
    })

    # KDE Plot
    print("Generating KDE plot...")
    plt.figure(figsize=(4, 4))
    sns.kdeplot(
        x=plot_data["Prob_pos_ranked"],
        y=plot_data["Degree_ranked"],
        cmap="Reds", fill=True,
        alpha=0.7, levels=50, thresh=0.05
    )

    # Spearman correlation
    correlation, p_value = scipy.stats.spearmanr(
        plot_data["Prob_pos_ranked"], plot_data["Degree_ranked"]
    )

    # Labels and formatting
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('ACGNN score rank', fontsize=10, labelpad=10)
    plt.ylabel('KCG interaction rank', fontsize=12, labelpad=10)

    # Add correlation text
    legend_text = f"Spearman R: {correlation:.4f}\nP-value: {p_value:.3e}"
    plt.text(
        0.05, 0.95, legend_text,
        fontsize=8, transform=plt.gca().transAxes,
        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )

    kde_output_path = os.path.join('gat/results/gene_prediction/', f'{args.model_type}_{args.net_type}_kde_plot_epo{args.num_epochs}_2048.png')
    plt.savefig(kde_output_path, bbox_inches='tight')
    print(f"KDE plot saved to {kde_output_path}")

    plt.tight_layout()
    plt.show()

    # Extract labeled scores and labels
    labeled_scores = scores[train_mask.cpu().numpy()]
    labeled_labels = labels[train_mask.cpu().numpy()]

    # Convert to NumPy arrays if necessary
    labeled_scores_np = labeled_scores.cpu().detach().numpy() if isinstance(labeled_scores, torch.Tensor) else labeled_scores
    labeled_labels_np = labeled_labels.cpu().detach().numpy() if isinstance(labeled_labels, torch.Tensor) else labeled_labels

    # Save ROC and PR curves
    output_file_roc = os.path.join('gat/results/gene_prediction/', f'{args.model_type}_{args.net_type}_epo{args.num_epochs}_2048_roc_curves.png')
    output_file_pr = os.path.join('gat/results/gene_prediction/', f'{args.model_type}_{args.net_type}_epo{args.num_epochs}_2048_pr_curves.png')

    plot_roc_curve(labeled_labels_np, labeled_scores_np, output_file_roc)
    plot_pr_curve(labeled_labels_np, labeled_scores_np, output_file_pr)

    print(f"ROC curve saved to {output_file_roc}")
    print(f"PR curve saved to {output_file_pr}")

def plot_model_performance(args):
    """
    Generates and saves a scatter plot comparing AUROC and AUPRC values 
    for different models across multiple networks.

    Parameters:
    - models: List of model names.
    - networks: List of network names.
    - auroc: 2D list of AUROC scores (rows: models, cols: networks).
    - auprc: 2D list of AUPRC scores (rows: models, cols: networks).
    - args: Arguments containing model and training configuration.
    - output_dir: Directory to save the plot.
    """


    # Define models and networks
    models = ["ACGNN", "HGDC", "EMOGI", "MTGCN", "GCN", "GAT", "GraphSAGE", "GIN", "Chebnet"]
    networks = ["CPDB", "STRING", "HIPPIE"]

    # AUPRC values for ONGene and OncoKB for each model (rows: models, cols: networks)
    auroc = [
        [0.9652, 0.9578, 0.9297],  # ACGNN ACGNN & 0.9652 & 0.9783 & 0.9578 & 0.9738 & 0.9297 & 0.9597 \\
        [0.6776, 0.7133, 0.6525],  # HGDC
        [0.6735, 0.8184, 0.6672],  # EMOGI
        [0.6862, 0.7130, 0.6762],  # MTGCN
        [0.6915, 0.6688, 0.6708],  # GCN
        [0.6670, 0.8166, 0.6478],  # GAT
        [0.6664, 0.6166, 0.6571],  # GraphSAGE
        [0.5836, 0.5173, 0.5844],  # GIN
        [0.8017, 0.8777, 0.7409]   # Chebnet
    ]

    auprc = [
        [0.9783, 0.9738, 0.9597],  # ACGNN
        [0.7288, 0.7740, 0.7634],  # HGDC
        [0.7230, 0.8737, 0.7960],  # EMOGI
        [0.7712, 0.7878, 0.7785],  # MTGCN
        [0.7730, 0.7681, 0.7675],  # GCN
        [0.7086, 0.8791, 0.7496],  # GAT
        [0.7522, 0.7182, 0.7624],  # GraphSAGE
        [0.6405, 0.5918, 0.6791],  # GIN
        [0.8622, 0.9159, 0.8443]   # Chebnet
    ]

    # Compute averages for each model
    avg_auroc = np.mean(auroc, axis=1)
    avg_auprc = np.mean(auprc, axis=1)

    # Define colors for models and unique shapes for networks
    colors = ['red', 'grey', 'blue', 'green', 'purple', 'orange', 'cyan', 'brown', 'pink']
    network_markers = ['P', '^', 's']  # One shape for each network
    avg_marker = 'o'  # Marker for average points

    # Create the plot
    plt.figure(figsize=(8, 7))

    # Plot individual points for each model and network
    for i, model in enumerate(models):
        for j, network in enumerate(networks):
            plt.scatter(auprc[i][j], auroc[i][j], color=colors[i], 
                        marker=network_markers[j], s=90, alpha=0.6)

    # Add average points for each model
    for i, model in enumerate(models):
        plt.scatter(avg_auprc[i], avg_auroc[i], color=colors[i], marker=avg_marker, 
                    s=240, edgecolor='none', alpha=0.5)

    # Create legends for models (colors) and networks (shapes)
    model_legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], 
                            markersize=14, label=models[i], alpha=0.5) for i in range(len(models))]
    network_legend = [Line2D([0], [0], marker=network_markers[i], color='k', linestyle='None', 
                            markersize=8, label=networks[i]) for i in range(len(networks))]

    # Add legends
    network_legend_artist = plt.legend(handles=network_legend, loc='lower right', title="Networks", fontsize=12, title_fontsize=14, frameon=True)
    plt.gca().add_artist(network_legend_artist)
    plt.legend(handles=model_legend, loc='upper left', fontsize=12, frameon=True)

    # Labels and title
    plt.ylabel("AUPRC", fontsize=14)
    plt.xlabel("AUROC", fontsize=14)

    # Save the plot
    comp_output_path = os.path.join('gat/results/gene_prediction/', f'{args.model_type}_{args.net_type}_comp_plot_epo{args.num_epochs}_2048.png')
    plt.savefig(comp_output_path, bbox_inches='tight')
    
    print(f"Comparison plot saved to {comp_output_path}")

    # Show plot
    plt.tight_layout()
    plt.show()

def save_model_details(model, args, model_csv_path, in_feats, hidden_feats, out_feats):
    """
    Extracts model details and saves them to a CSV file.

    Parameters:
    - model: The neural network model.
    - args: Arguments containing model configuration.
    - model_csv_path: File path to save the model details.
    - in_feats: Number of input features.
    - hidden_feats: Number of hidden layer features.
    - out_feats: Number of output features.
    """
    # Count layers and parameters
    num_layers = sum(1 for _ in model.children())  # Count layers
    total_params = sum(p.numel() for p in model.parameters())  # Count parameters

    # Detect attention layers
    attention_layer_nodes = None
    for layer in model.children():
        if hasattr(layer, 'heads'):  # Assuming attention layers have 'heads' attribute
            attention_layer_nodes = layer.heads

    # Detect residual connections
    has_residual = any(isinstance(layer, nn.Identity) for layer in model.modules())

    # Prepare data for CSV
    model_data = {
        "Method": [args.model_type],
        "Number of Layers": [num_layers],
        "Input Layer Nodes": [in_feats],
        "Hidden Layer Nodes": [hidden_feats],
        "Attention Layer Nodes": [attention_layer_nodes if attention_layer_nodes else "N/A"],
        "Output Layer Nodes": [out_feats],
        "Total Parameters": [total_params],
        "Residual Connection": ["Yes" if has_residual else "No"]
    }

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(model_data)
    df.to_csv(model_csv_path, index=False)
    print(f"Model architecture saved to {model_csv_path}")

def save_predicted_scores(scores, labels, nodes, args):
    """
    Saves predicted scores and labels to a CSV file.

    Parameters:
    - scores: List of predicted scores.
    - labels: List of ground-truth labels.
    - nodes: Dictionary of node names.
    - args: Arguments containing model configuration.
    """
    # Initialize variables to calculate average scores and standard deviations
    label_scores = {0: [], 1: [], 2: [], 3: []}  # Groups for each label

    # Define CSV file path
    csv_file_path = os.path.join(
        'gat/results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_predicted_scores_threshold{args.score_threshold}_epo{args.num_epochs}.csv'
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    # Save results to CSV
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Node Name', 'Score', 'Label'])  # Header

        for i, score in enumerate(scores):
            label = int(labels[i].item())  # Ensure label is an integer
            
            if label in [1, 0]:  # Ground-truth labels
                writer.writerow([list(nodes.keys())[i], score, label])
                label_scores[label].append(score)
            elif label == -1 and score >= args.score_threshold:  # Predicted driver genes
                writer.writerow([list(nodes.keys())[i], score, 2])
                label_scores[2].append(score)
            else:  # Non-labeled nodes or other
                writer.writerow([list(nodes.keys())[i], score, 3])
                label_scores[3].append(score)

    print(f"Predicted scores and labels saved to {csv_file_path}")

    return label_scores  # Returning for further analysis if needed

def save_average_scores(label_scores, args):
    """
    Calculates and saves the average score, standard deviation, and number of nodes per label.

    Parameters:
    - label_scores: Dictionary with labels as keys and lists of scores as values.
    - args: Arguments containing model configuration.
    """
    # Define CSV file path
    average_scores_file = os.path.join(
        'gat/results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_average_scores_by_label_threshold{args.score_threshold}_epo{args.num_epochs}.csv'
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(average_scores_file), exist_ok=True)

    # Save average scores to CSV
    with open(average_scores_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Label', 'Average Score', 'Standard Deviation', 'Number of Nodes'])  # Header

        for label, scores_list in label_scores.items():
            if scores_list:  # Check if the list is not empty
                avg_score = np.mean(scores_list)
                std_dev = np.std(scores_list)
                num_nodes = len(scores_list)
            else:
                avg_score = 0.0  # Default if no nodes in the label group
                std_dev = 0.0
                num_nodes = 0

            writer.writerow([label, avg_score, std_dev, num_nodes])

    print(f"Average scores by label saved to {average_scores_file}")

def plot_average_scores(label_scores, args):
    """
    Plots average scores with error bars and saves the figure.

    Parameters:
    - label_scores: Dictionary with labels as keys and lists of scores as values.
    - args: Arguments containing model configuration.
    """
    labels_list = []
    avg_scores = []
    std_devs = []

    for label, scores_list in label_scores.items():
        if scores_list:
            labels_list.append(label)
            avg_scores.append(np.mean(scores_list))
            std_devs.append(np.std(scores_list))

    if not labels_list:
        print("No valid scores to plot.")
        return

    # Define plot save path
    plot_path = os.path.join(
        'gat/results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_average_scores_with_error_bars_threshold{args.score_threshold}_epo{args.num_epochs}.png'
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.bar(labels_list, avg_scores, yerr=std_devs, capsize=5, color='skyblue', alpha=0.7)
    plt.xlabel('Label')
    plt.ylabel('Average Score')
    plt.title('Average Scores by Label with Error Bars')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Save and close plot
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Error bar plot saved to {plot_path}")

def plot_score_distributions(label_scores, args):
    """
    Plots score distributions for each label and saves the figures.

    Parameters:
    - label_scores: Dictionary with labels as keys and lists of scores as values.
    - args: Arguments containing model configuration.
    """
    for label, scores_list in label_scores.items():
        if scores_list:
            plt.figure(figsize=(8, 6))
            plt.hist(scores_list, bins=20, alpha=0.7, color='#98f5e1', edgecolor='black')

            # Set labels and tick sizes
            plt.xlabel('Score', fontsize=16)
            plt.ylabel('Frequency', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            # Customize grid and tick appearance
            plt.tick_params(axis='both', which='major', length=6, width=2, direction='inout', grid_color='gray', grid_alpha=0.5)
            plt.grid(axis='y', linestyle='--', alpha=0.6)

            # Define plot save path
            plot_path = os.path.join(
                'gat/results/gene_prediction/',
                f'{args.model_type}_{args.net_type}_score_distribution_label{label}_threshold{args.score_threshold}_epo{args.num_epochs}.png'
            )

            # Ensure directory exists
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)

            # Save and close plot
            plt.savefig(plot_path)
            plt.close()

            print(f"Score distribution for label {label} saved to {plot_path}")

def save_performance_metrics(epoch_times, cpu_usages, gpu_usages, args):
    """
    Saves performance metrics per epoch, including time per epoch, CPU, and GPU usage.

    Parameters:
    - epoch_times: List of epoch durations (in seconds).
    - cpu_usages: List of CPU memory usage per epoch (in MB).
    - gpu_usages: List of GPU memory usage per epoch (in MB).
    - args: Arguments containing model and training configuration.
    - output_dir: Directory to save the metrics CSV file.
    """

    # Compute total and average performance metrics
    total_time = sum(epoch_times)
    avg_time_per_epoch = total_time / args.num_epochs
    avg_cpu_usage = sum(cpu_usages) / args.num_epochs
    avg_gpu_usage = sum(gpu_usages) / args.num_epochs

    # Create DataFrame with per-epoch metrics
    df_metrics = pd.DataFrame({
        "Epoch": range(1, args.num_epochs + 1),
        "Time per Epoch (s)": epoch_times,
        "CPU Usage (MB)": cpu_usages,
        "GPU Usage (MB)": gpu_usages
    })

    # Define CSV path
    metrics_csv_path = os.path.join(
        'gat/results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_performance_metrics_epo{args.num_epochs}_2048.csv'
    )

    # Save to CSV
    df_metrics.to_csv(metrics_csv_path, index=False)

    print(f"Epoch performance metrics saved to {metrics_csv_path}")

    # Print summary statistics
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Average Time per Epoch: {avg_time_per_epoch:.2f} seconds")
    print(f"Average CPU Usage: {avg_cpu_usage:.2f} MB")
    print(f"Average GPU Usage: {avg_gpu_usage:.2f} MB")

