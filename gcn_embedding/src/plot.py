import csv
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def miRNA_plot_cosine_similarity_matrix_for_clusters_with_values(embeddings, miRNAs, save_path):
    # Compute cosine similarity matrix
    cos_sim = np.dot(embeddings, np.array(embeddings).T)
    norms = np.linalg.norm(embeddings, axis=1)
    cos_sim /= np.outer(norms, norms)

    # Normalize the cosine similarity matrix to the range [0, 1]
    cos_sim_min = cos_sim.min()
    cos_sim_max = cos_sim.max()
    cos_sim_normalized = (cos_sim - cos_sim_min) / (cos_sim_max - cos_sim_min)

    scale_min = 0.3764
    scale_max = 1
    cos_sim_scaled = cos_sim_normalized * (scale_max - scale_min) + scale_min

    np.fill_diagonal(cos_sim_scaled, 1.0)

    plt.figure(figsize=(10, 8))

    # Set vmin and vmax for the scaled matrix
    vmin = scale_min  # The minimum value after scaling
    vmax = scale_max  # The maximum value after scaling

    # Create a heatmap with annotations, using scaled values
    ax = sns.heatmap(cos_sim_scaled, cmap="Spectral", annot=True, fmt=".4f", annot_kws={"size": 9},
                     xticklabels=miRNAs, yticklabels=miRNAs,
                     vmin=vmin, vmax=vmax,  # Use scaled range [0.27, 1]
                     cbar_kws={"shrink": 0.2, "aspect": 15, "ticks": [vmin, vmax]})

    # Highlight the diagonal squares with value 1 by setting their background color to black
    '''for i in range(len(miRNAs)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color='black', alpha=0.5, zorder=3))'''

    ax.xaxis.tick_top()  # Move x-axis labels to the top
    ax.xaxis.set_label_position('top')  # Set x-axis label position to top
    plt.xticks(rotation=-30, fontsize=10, ha='right')  # Rotate x-axis labels, set font size, and align to the right
    plt.yticks(rotation=0, fontsize=10)  # Set font size and rotation for y-axis labels

    # Draw short lines on the left of the leftmost column
    for i in range(len(miRNAs)):
        plt.plot([-0.5, -0.45], [i + 0.5, i + 0.5], color='black', linestyle='-', linewidth=1)

    # Set the title below the plot
    ax.text(x=0.5, y=-0.03, s="miRNA-miRNA similarities", fontsize=16, ha='center', va='top', transform=ax.transAxes)

    # Adjust the plot and color bar position to move it slightly to the bottom-right
    box = ax.get_position()
    ax.set_position([box.x0 + 0.14, box.y0, box.width * 0.9, box.height * 0.9])

    # Move the color bar as well
    cbar = ax.collections[0].colorbar
    cbar.ax.set_position([box.x1 + 0.1, box.y0 - 0.03, 0.01, box.height])

    # Set the color bar labels to 4 decimal places
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.4f}'))

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def disease_plot_cosine_similarity_matrix_for_clusters_with_values(embeddings, miRNAs, save_path):
    # Compute cosine similarity matrix
    cos_sim = np.dot(embeddings, np.array(embeddings).T)
    norms = np.linalg.norm(embeddings, axis=1)
    cos_sim /= np.outer(norms, norms)

    # Normalize the cosine similarity matrix to the range [0, 1]
    cos_sim_min = cos_sim.min()
    cos_sim_max = cos_sim.max()
    cos_sim_normalized = (cos_sim - cos_sim_min) / (cos_sim_max - cos_sim_min)

    # Scale the normalized matrix to the range [0.27, 1]
    scale_min = 0.2739
    scale_max = 1
    cos_sim_scaled = cos_sim_normalized * (scale_max - scale_min) + scale_min

    # Set diagonal cells to 1.00 (since cosine similarity of a vector with itself is 1)
    np.fill_diagonal(cos_sim_scaled, 1.0)

    plt.figure(figsize=(10, 8))

    # Set vmin and vmax for the scaled matrix
    vmin = scale_min  # The minimum value after scaling
    vmax = scale_max  # The maximum value after scaling

    # Create a heatmap with annotations, using scaled values
    ax = sns.heatmap(cos_sim_scaled, cmap="Spectral", annot=True, fmt=".4f", annot_kws={"size": 9},
                     xticklabels=miRNAs, yticklabels=miRNAs,
                     vmin=vmin, vmax=vmax,  # Use scaled range [0.27, 1]
                     cbar_kws={"shrink": 0.2, "aspect": 15, "ticks": [vmin, vmax]})

    # Highlight the diagonal squares with value 1 by setting their background color to black
    '''for i in range(len(miRNAs)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color='black', alpha=0.5, zorder=3))'''

    ax.xaxis.tick_top()  # Move x-axis labels to the top
    ax.xaxis.set_label_position('top')  # Set x-axis label position to top
    plt.xticks(rotation=-30, fontsize=10, ha='right')  # Rotate x-axis labels, set font size, and align to the right
    plt.yticks(rotation=0, fontsize=10)  # Set font size and rotation for y-axis labels

    # Draw short lines on the left of the leftmost column
    for i in range(len(miRNAs)):
        plt.plot([-0.5, -0.45], [i + 0.5, i + 0.5], color='black', linestyle='-', linewidth=1)

    # Set the title below the plot
    ax.text(x=0.5, y=-0.03, s="disease-disease similarities", fontsize=16, ha='center', va='top', transform=ax.transAxes)

    # Adjust the plot and color bar position to move it slightly to the bottom-right
    box = ax.get_position()
    ax.set_position([box.x0 + 0.14, box.y0, box.width * 0.9, box.height * 0.9])

    # Move the color bar as well
    cbar = ax.collections[0].colorbar
    cbar.ax.set_position([box.x1 + 0.1, box.y0 - 0.03, 0.01, box.height])

    # Set the color bar labels to 4 decimal places
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.4f}'))

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


##def calculate_cluster_labels(net, dataloader, device, num_clusters=15):
def calculate_cluster_labels(net, dataloader, device, num_clusters=5):
    miRNA_embeddings_list = []
    disease_embeddings_list = []
    miRNA_names_list = []
    disease_names_list = []

    net.eval()
    with torch.no_grad():
        for data in dataloader:
            graph, name = data
            embeddings = net.get_node_embeddings(graph.to(device))

            # Debug: Print available node attributes
            print("Available node attributes:", graph.ndata.keys())
            
            # Check if 'node_type' exists in the graph
            if 'node_type' not in graph.ndata:
                raise KeyError("The node attribute 'node_type' is not present in the graph.")
            
            # Extract node types and create masks
            node_types = graph.ndata['node_type'].cpu().numpy()  # Convert to numpy for easy comparison
            node_types_= graph.ndata['node_type']
            print('node_types_-----------------------------\n',node_types_)
            miRNA_mask = (node_types == 0)
            disease_mask = (node_types == 1)
            
            # Debug: Print mask sizes and number of nodes
            print(f"miRNA mask size: {miRNA_mask.sum()}, Disease mask size: {disease_mask.sum()}")
            
            # Separate embeddings and names based on node type
            if miRNA_mask.any():
                miRNA_embeddings = embeddings[miRNA_mask]
                miRNA_names = [name[i] for i in range(len(name)) if node_types[i] == 'miRNA']
                miRNA_embeddings_list.append(miRNA_embeddings)
                miRNA_names_list.extend(miRNA_names)

            if disease_mask.any():
                disease_embeddings = embeddings[disease_mask]
                disease_names = [name[i] for i in range(len(name)) if node_types[i] == 'disease']
                disease_embeddings_list.append(disease_embeddings)
                disease_names_list.extend(disease_names)
    
    # Concatenate all embeddings
    if miRNA_embeddings_list:
        all_miRNA_embeddings = torch.cat(miRNA_embeddings_list, dim=0).cpu().numpy()
    else:
        all_miRNA_embeddings = np.array([]).reshape(0, embeddings.size(-1))  # Handle empty case

    if disease_embeddings_list:
        all_disease_embeddings = torch.cat(disease_embeddings_list, dim=0).cpu().numpy()
    else:
        all_disease_embeddings = np.array([]).reshape(0, embeddings.size(-1))  # Handle empty case

    # Use KMeans clustering to assign cluster labels
    if all_miRNA_embeddings.size > 0:
        kmeans_miRNA = KMeans(n_clusters=num_clusters, random_state=42)
        miRNA_cluster_labels = kmeans_miRNA.fit_predict(all_miRNA_embeddings)
    else:
        miRNA_cluster_labels = np.array([])  # Handle empty case

    if all_disease_embeddings.size > 0:
        kmeans_disease = KMeans(n_clusters=num_clusters, random_state=42)
        disease_cluster_labels = kmeans_disease.fit_predict(all_disease_embeddings)
    else:
        disease_cluster_labels = np.array([])  # Handle empty case

    return all_miRNA_embeddings, miRNA_cluster_labels, miRNA_names, all_disease_embeddings, disease_cluster_labels, disease_names

#####################################################################################################


def _calculate_cluster_labels(net, dataloader, device, num_clusters):
    miRNA_embeddings_list = []
    disease_embeddings_list = []
    miRNA_names_list = []
    disease_names_list = []
    
    net.eval()
    with torch.no_grad():
        for data in dataloader:
            graph, name = data
            embeddings = net.get_node_embeddings(graph.to(device))
            
            # Debug: Print available node attributes
            print("Available node attributes:", graph.ndata.keys())
            
            # Check if 'node_type' exists in the graph
            if 'node_type' not in graph.ndata:
                raise KeyError("The node attribute 'node_type' is not present in the graph.")
            
            # Extract node types and create masks
            node_types = graph.ndata['node_type']
            miRNA_mask = (node_types == 'miRNA')
            disease_mask = (node_types == 'disease')
            
            # Separate embeddings and names based on node type
            miRNA_embeddings = embeddings[miRNA_mask]
            disease_embeddings = embeddings[disease_mask]
            
            miRNA_names = [name[i] for i in range(len(name)) if node_types[i] == 'miRNA']
            disease_names = [name[i] for i in range(len(name)) if node_types[i] == 'disease']
            
            miRNA_embeddings_list.append(miRNA_embeddings)
            disease_embeddings_list.append(disease_embeddings)
            print('disease_embeddings_list-------------------\n',disease_embeddings_list)
            
            miRNA_names_list.extend(miRNA_names)
            disease_names_list.extend(disease_names)
    
    # Concatenate all embeddings
    all_miRNA_embeddings = torch.cat(miRNA_embeddings_list, dim=0).cpu().numpy()
    all_disease_embeddings = torch.cat(disease_embeddings_list, dim=0).cpu().numpy()
    
    # Use KMeans clustering to assign cluster labels
    kmeans_miRNA = KMeans(n_clusters=num_clusters, random_state=42)
    ##print('all_miRNA_embeddings.shape-------------------\n',all_miRNA_embeddings.shape)
    miRNA_cluster_labels = kmeans_miRNA.fit_predict(all_miRNA_embeddings)
    
    kmeans_disease = KMeans(n_clusters=num_clusters, random_state=42)
    disease_cluster_labels = kmeans_disease.fit_predict(all_disease_embeddings)
    
    return all_miRNA_embeddings, miRNA_cluster_labels, miRNA_names, all_disease_embeddings, disease_cluster_labels, disease_names

def create_heatmap(embedding_list, node_list, save_path):
    # Convert the embedding list to a DataFrame
    heatmap_data = pd.DataFrame(embedding_list, index=node_list)
    
    # Create a clustermap
    ax = sns.clustermap(heatmap_data, cmap='tab20', standard_scale=1, figsize=(10, 10))
    
    # Set smaller font sizes for various elements
    ax.ax_heatmap.tick_params(axis='both', which='both', labelsize=8)  # Tick labels
    ax.ax_heatmap.set_xlabel(ax.ax_heatmap.get_xlabel(), fontsize=8)  # X-axis label
    ax.ax_heatmap.set_ylabel(ax.ax_heatmap.get_ylabel(), fontsize=8)  # Y-axis label
    ax.ax_heatmap.collections[0].colorbar.ax.tick_params(labelsize=8)  # Color bar labels
    
    # Limit the number of y-tick labels to 50
    max_labels = 30
    y_labels = heatmap_data.index
    if len(y_labels) > max_labels:
        step = len(y_labels) // max_labels
        y_ticks = range(0, len(y_labels), step)
        ax.ax_heatmap.set_yticks(y_ticks)
        ax.ax_heatmap.set_yticklabels([y_labels[i] for i in y_ticks], fontsize=8)
    else:
        ax.ax_heatmap.set_yticks(range(len(y_labels)))
        ax.ax_heatmap.set_yticklabels(y_labels, fontsize=8)
    
    # Save the clustermap to a file
    plt.savefig(save_path)
    plt.close()

def calculate_cluster_labels_miRNA(net, dataloader, device, num_clusters=5):
    all_embeddings = []
    all_names = []
    
    net.eval()
    with torch.no_grad():
        for data in dataloader:
            graph, name = data
            print('graph==================\n', graph)
            
            # Get node embeddings
            embeddings = net.get_node_embeddings(graph.to(device))
            
            # Debug: Print available node attributes
            print("Available node attributes:", graph.ndata.keys())
            
            # Check if 'node_type' exists in the graph
            if 'node_type' not in graph.ndata:
                raise KeyError("The node attribute 'node_type' is not present in the graph.")
            
            # Extract node types and filter for 'miRNA'
            node_types = graph.ndata['node_type']  # Assuming 'node_type' contains node type information
            miRNA_mask = node_types == 0  # Assuming 'miRNA' is represented by 0 in node_type
            
            # Filter embeddings and names for 'miRNA' nodes
            miRNA_embeddings = embeddings[miRNA_mask]
            miRNA_names = [n for i, n in enumerate(name) if miRNA_mask[i]]
            
            # Flatten the embeddings if they have extra dimensions
            miRNA_embeddings = miRNA_embeddings.view(miRNA_embeddings.shape[0], -1)
            
            all_embeddings.append(miRNA_embeddings)
            all_names.extend(miRNA_names)
    
    # Concatenate all embeddings into a single 2D array
    all_embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()
    
    # Check the shape of the final embeddings
    print("Final shape of all_embeddings:", all_embeddings.shape)
    
    # Use KMeans clustering to assign cluster labels
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(all_embeddings)
    
    return all_embeddings, cluster_labels, all_names

def calculate_cluster_labels_disease(net, dataloader, device, num_clusters=5):
    all_embeddings = []
    all_names = []
    
    net.eval()
    with torch.no_grad():
        for data in dataloader:
            graph, name = data
            print('graph==================\n', graph)
            
            # Get node embeddings
            embeddings = net.get_node_embeddings(graph.to(device))
            
            # Debug: Print available node attributes
            print("Available node attributes:", graph.ndata.keys())
            
            # Check if 'node_type' exists in the graph
            if 'node_type' not in graph.ndata:
                raise KeyError("The node attribute 'node_type' is not present in the graph.")
            
            # Extract node types and filter for 'disease'
            node_types = graph.ndata['node_type']  # Assuming 'node_type' contains node type information
            disease_mask = node_types == 1  # Assuming 'disease' is represented by 1 in node_type
            
            # Filter embeddings and names for 'disease' nodes
            disease_embeddings = embeddings[disease_mask]
            disease_names = [n for i, n in enumerate(name) if disease_mask[i]]
            
            # Flatten the embeddings if they have extra dimensions
            disease_embeddings = disease_embeddings.view(disease_embeddings.shape[0], -1)
            
            all_embeddings.append(disease_embeddings)
            all_names.extend(disease_names)
    
    # Concatenate all embeddings into a single 2D array
    all_embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()
    
    # Check the shape of the final embeddings
    print("Final shape of all_embeddings:", all_embeddings.shape)
    
    # Use KMeans clustering to assign cluster labels
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(all_embeddings)
    
    return all_embeddings, cluster_labels, all_names

def _calculate_cluster_labels_miRNA(net, dataloader, device, num_clusters=5):
    all_embeddings = []
    all_names = []
    
    net.eval()
    with torch.no_grad():
        for data in dataloader:
            graph, name = data
            print('graph==================\n', graph)
            
            # Get node embeddings
            embeddings = net.get_node_embeddings(graph.to(device))
            
            # Debug: Print available node attributes
            print("Available node attributes:", graph.ndata.keys())
            
            # Check if 'node_type' exists in the graph
            if 'node_type' not in graph.ndata:
                raise KeyError("The node attribute 'node_type' is not present in the graph.")
            
            # Extract node types and filter for 'miRNA'
            node_types = graph.ndata['node_type']  # Assuming 'node_type' contains node type information
            miRNA_mask = node_types == 0  # Assuming 'miRNA' is represented by 0 in node_type
            
            # Filter embeddings and names for 'miRNA' nodes
            miRNA_embeddings = embeddings[miRNA_mask]
            miRNA_names = [n for i, n in enumerate(name) if miRNA_mask[i]]
            
            # Flatten the embeddings if they have extra dimensions
            miRNA_embeddings = miRNA_embeddings.view(miRNA_embeddings.shape[0], -1)
            
            all_embeddings.append(miRNA_embeddings)
            all_names.extend(miRNA_names)
    
    # Concatenate all embeddings into a single 2D array
    all_embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()
    
    # Check the shape of the final embeddings
    print("Final shape of all_embeddings:", all_embeddings.shape)
    
    # Use KMeans clustering to assign cluster labels
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(all_embeddings)
    
    return all_embeddings, cluster_labels, all_names

def populate_miRNA_dic(graphs, node_embeddings_initial, node_to_index_initial, cluster_labels_initial):
    # Initialize dictionaries to store the results
    miRNA_dic_initial = {}
    first_node_miRNA_in_cluster_initial = {}
    first_node_embedding_in_cluster_initial = {}

    # Ensure that we handle both individual graphs and lists of graphs
    if isinstance(graphs, list):
        graph_list = graphs
    else:
        graph_list = [graphs]  # If it's a single graph, convert it to a list

    # Iterate over each graph
    for nx_graph in graph_list:
        # Populate miRNA_dic with node miRNAs mapped to embeddings, only for miRNA nodes
        for node in nx_graph.nodes:
            # Check if the node's 'node_type' attribute is 'miRNA'
            if nx_graph.nodes[node].get('node_type') == 'miRNA':
                # Ensure the node exists in node_to_index_initial
                if node in node_to_index_initial:
                    index = node_to_index_initial[node]
                    if index >= len(node_embeddings_initial):
                        print(f"Index out of bounds: {index} for node {node}")
                    else:
                        # Use the node ID as the key and map it to its embedding
                        miRNA_dic_initial[node] = node_embeddings_initial[index]
                else:
                    print(f"Node {node} not found in node_to_index_initial")

        # Assign first miRNA in each cluster to the cluster label
        for node, cluster in zip(nx_graph.nodes, cluster_labels_initial):
            if nx_graph.nodes[node].get('node_type') == 'miRNA':  # Only consider miRNA nodes for clustering
                if cluster not in first_node_miRNA_in_cluster_initial:
                    # Store the first miRNA node in the cluster
                    first_node_miRNA_in_cluster_initial[cluster] = node
                    # Store the corresponding embedding
                    first_node_embedding_in_cluster_initial[cluster] = node_embeddings_initial[node_to_index_initial[node]]

    return miRNA_dic_initial, first_node_miRNA_in_cluster_initial, first_node_embedding_in_cluster_initial

def populate_disease_dic(graphs, node_embeddings_initial, node_to_index_initial, cluster_labels_initial):
    # Initialize dictionaries to store the results
    disease_dic_initial = {}
    first_node_disease_in_cluster_initial = {}
    first_node_embedding_in_cluster_initial = {}

    # Ensure that we handle both individual graphs and lists of graphs
    if isinstance(graphs, list):
        graph_list = graphs
    else:
        graph_list = [graphs]  # If it's a single graph, convert it to a list

    # Iterate over each graph
    for nx_graph in graph_list:
        # Populate disease_dic with node diseases mapped to embeddings, only for disease nodes
        for node in nx_graph.nodes:
            # Check if the node's 'node_type' attribute is 'disease'
            if nx_graph.nodes[node].get('node_type') == 'disease':
                # Ensure the node exists in node_to_index_initial
                if node in node_to_index_initial:
                    index = node_to_index_initial[node]
                    if index >= len(node_embeddings_initial):
                        print(f"Index out of bounds: {index} for node {node}")
                    else:
                        # Use the node ID as the key and map it to its embedding
                        disease_dic_initial[node] = node_embeddings_initial[index]
                else:
                    print(f"Node {node} not found in node_to_index_initial")

        # Assign first disease in each cluster to the cluster label
        for node, cluster in zip(nx_graph.nodes, cluster_labels_initial):
            if nx_graph.nodes[node].get('node_type') == 'disease':  # Only consider disease nodes for clustering
                if cluster not in first_node_disease_in_cluster_initial:
                    # Store the first disease node in the cluster
                    first_node_disease_in_cluster_initial[cluster] = node
                    # Store the corresponding embedding
                    first_node_embedding_in_cluster_initial[cluster] = node_embeddings_initial[node_to_index_initial[node]]

    return disease_dic_initial, first_node_disease_in_cluster_initial, first_node_embedding_in_cluster_initial

def populate_miRNA_dic_significance(graph, nx_graph, node_embeddings, node_to_index, cluster_labels):
    # Initialize dictionaries and lists to store the results
    miRNA_dic = {}
    significant_miRNAs = []
    first_node_miRNA_in_cluster = {}
    first_node_embedding_in_cluster = {}
    
    cluster_miRNA_dict = {}  # Dictionary to store clusters and corresponding miRNAs
    clusters_with_significant_miRNA = {}  # Dictionary to store clusters and corresponding significant miRNAs
    clusters_node_info = {}  # Dictionary to store node info for each cluster
    
    # Populate miRNA_dic with node miRNAs mapped to embeddings, and check for significance
    for node in nx_graph.nodes:
        # Ensure node type is 'miRNA' and process its embedding
        if nx_graph.nodes[node].get('node_type') == 'miRNA':
            miRNA = nx_graph.nodes[node].get('miRNA', node)  # Use node as fallback key if 'miRNA' is missing
            # Ensure node exists in node_to_index
            if node in node_to_index:
                index = node_to_index[node]
                if index < len(node_embeddings):
                    miRNA_dic[miRNA] = node_embeddings[index]
                else:
                    print(f"Index out of bounds: {index} for node {node}")
            else:
                print(f"Node {node} not found in node_to_index")
            
            # Check if the node's significance is 'significant'
            if nx_graph.nodes[node].get('significance') == 'significant':
                significant_miRNAs.append(miRNA)

    # Assign the first miRNA in each cluster
    for node, cluster in zip(nx_graph.nodes, cluster_labels):
        if nx_graph.nodes[node].get('node_type') == 'miRNA':  # Only consider miRNA nodes for clustering
            miRNA = nx_graph.nodes[node].get('miRNA', node)  # Use node as fallback key if 'miRNA' is missing
            if cluster not in first_node_miRNA_in_cluster:
                # Store the first miRNA node in the cluster
                first_node_miRNA_in_cluster[cluster] = miRNA
                # Store the corresponding embedding
                if node in node_to_index:
                    index = node_to_index[node]
                    if index < len(node_embeddings):
                        first_node_embedding_in_cluster[cluster] = node_embeddings[index]
                    else:
                        print(f"Index out of bounds: {index} for node {node}")
                else:
                    print(f"Node {node} not found in node_to_index")

            # Populate cluster_miRNA_dict
            if cluster not in cluster_miRNA_dict:
                cluster_miRNA_dict[cluster] = []
            cluster_miRNA_dict[cluster].append(miRNA)

            # Populate clusters_with_significant_miRNA
            if cluster not in clusters_with_significant_miRNA:
                clusters_with_significant_miRNA[cluster] = []
            if miRNA in significant_miRNAs:
                clusters_with_significant_miRNA[cluster].append(miRNA)
            
            # Populate clusters_node_info with node information for each cluster
            if cluster not in clusters_node_info:
                clusters_node_info[cluster] = []
            node_info = {
                'miRNA': miRNA,
                'significance': nx_graph.nodes[node].get('significance', 'unknown'),
                'other_info': nx_graph.nodes[node]  # Add other relevant info if necessary
            }
            clusters_node_info[cluster].append(node_info)
    
    return miRNA_dic, significant_miRNAs, first_node_miRNA_in_cluster, first_node_embedding_in_cluster

def populate_disease_dic_significance(graph, nx_graph, node_embeddings, node_to_index, cluster_labels):
    # Initialize dictionaries and lists to store the results
    disease_dic = {}
    significant_diseases = []
    first_node_disease_in_cluster = {}
    first_node_embedding_in_cluster = {}
    
    cluster_disease_dict = {}  # Dictionary to store clusters and corresponding diseases
    clusters_with_significant_disease = {}  # Dictionary to store clusters and corresponding significant diseases
    clusters_node_info = {}  # Dictionary to store node info for each cluster
    
    # Populate disease_dic with node diseases mapped to embeddings, and check for significance
    for node in nx_graph.nodes:
        # Ensure node type is 'disease' and process its embedding
        if nx_graph.nodes[node].get('node_type') == 'disease':
            disease = nx_graph.nodes[node].get('disease', node)  # Use node as fallback key if 'disease' is missing
            # Ensure node exists in node_to_index
            if node in node_to_index:
                index = node_to_index[node]
                if index < len(node_embeddings):
                    disease_dic[disease] = node_embeddings[index]
                else:
                    print(f"Index out of bounds: {index} for node {node}")
            else:
                print(f"Node {node} not found in node_to_index")
            
            # Check if the node's significance is 'significant'
            if nx_graph.nodes[node].get('significance') == 'significant':
                significant_diseases.append(disease)

    # Assign the first disease in each cluster
    for node, cluster in zip(nx_graph.nodes, cluster_labels):
        if nx_graph.nodes[node].get('node_type') == 'disease':  # Only consider disease nodes for clustering
            disease = nx_graph.nodes[node].get('disease', node)  # Use node as fallback key if 'disease' is missing
            if cluster not in first_node_disease_in_cluster:
                # Store the first disease node in the cluster
                first_node_disease_in_cluster[cluster] = disease
                # Store the corresponding embedding
                if node in node_to_index:
                    index = node_to_index[node]
                    if index < len(node_embeddings):
                        first_node_embedding_in_cluster[cluster] = node_embeddings[index]
                    else:
                        print(f"Index out of bounds: {index} for node {node}")
                else:
                    print(f"Node {node} not found in node_to_index")

            # Populate cluster_disease_dict
            if cluster not in cluster_disease_dict:
                cluster_disease_dict[cluster] = []
            cluster_disease_dict[cluster].append(disease)

            # Populate clusters_with_significant_disease
            if cluster not in clusters_with_significant_disease:
                clusters_with_significant_disease[cluster] = []
            if disease in significant_diseases:
                clusters_with_significant_disease[cluster].append(disease)
            
            # Populate clusters_node_info with node information for each cluster
            if cluster not in clusters_node_info:
                clusters_node_info[cluster] = []
            node_info = {
                'disease': disease,
                'significance': nx_graph.nodes[node].get('significance', 'unknown'),
                'other_info': nx_graph.nodes[node]  # Add other relevant info if necessary
            }
            clusters_node_info[cluster].append(node_info)
    
    return disease_dic, significant_diseases, first_node_disease_in_cluster, first_node_embedding_in_cluster


def _0_1_plot_cosine_similarity_matrix_for_clusters_with_values(embeddings, miRNAs, save_path):
    # Compute cosine similarity matrix
    cos_sim = np.dot(embeddings, np.array(embeddings).T)
    norms = np.linalg.norm(embeddings, axis=1)
    cos_sim /= np.outer(norms, norms)

    # Normalize the cosine similarity matrix to the range [0, 1]
    cos_sim_min = cos_sim.min()
    cos_sim_max = cos_sim.max()
    cos_sim_normalized = (cos_sim - cos_sim_min) / (cos_sim_max - cos_sim_min)

    # Set diagonal cells to 1.00 (since cosine similarity of a vector with itself is 1)
    np.fill_diagonal(cos_sim_normalized, 1.0)

    plt.figure(figsize=(10, 8))

    # Set vmin and vmax for the normalized matrix
    vmin = 0  # The minimum value after normalization
    vmax = 1  # The maximum value after normalization

    # Create a heatmap with annotations, using normalized values
    ax = sns.heatmap(cos_sim_normalized, cmap="Spectral", annot=True, fmt=".4f", annot_kws={"size": 9},
                     xticklabels=miRNAs, yticklabels=miRNAs,
                     vmin=vmin, vmax=vmax,  # Use normalized range [0, 1]
                     cbar_kws={"shrink": 0.2, "aspect": 15, "ticks": [vmin, vmax]})

    # Highlight the diagonal squares with value 1 by setting their background color to black
    '''for i in range(len(miRNAs)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color='black', alpha=0.5, zorder=3))'''

    ax.xaxis.tick_top()  # Move x-axis labels to the top
    ax.xaxis.set_label_position('top')  # Set x-axis label position to top
    plt.xticks(rotation=-30, fontsize=10, ha='right')  # Rotate x-axis labels, set font size, and align to the right
    plt.yticks(rotation=0, fontsize=10)  # Set font size and rotation for y-axis labels

    # Draw short lines on the left of the leftmost column
    for i in range(len(miRNAs)):
        plt.plot([-0.5, -0.45], [i + 0.5, i + 0.5], color='black', linestyle='-', linewidth=1)

    # Set the title below the plot
    ax.text(x=0.5, y=-0.03, s="miRNA-miRNA similarities", fontsize=16, ha='center', va='top', transform=ax.transAxes)

    # Adjust the plot and color bar position to move it slightly to the bottom-right
    box = ax.get_position()
    ax.set_position([box.x0 + 0.14, box.y0, box.width * 0.9, box.height * 0.9])

    # Move the color bar as well
    cbar = ax.collections[0].colorbar
    cbar.ax.set_position([box.x1 + 0.1, box.y0 - 0.03, 0.01, box.height])

    # Color bar labels for the normalized range
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def _negative_plot_cosine_similarity_matrix_for_clusters_with_values(embeddings, miRNAs, save_path):
    cos_sim = np.dot(embeddings, np.array(embeddings).T)
    norms = np.linalg.norm(embeddings, axis=1)
    cos_sim /= np.outer(norms, norms)

    plt.figure(figsize=(10, 8))
    
    vmin = cos_sim.min()
    vmax = cos_sim.max()

    ax = sns.heatmap(cos_sim, cmap="Spectral", annot=True, fmt=".4f", annot_kws={"size": 9},
                     xticklabels=miRNAs, yticklabels=miRNAs,
                     cbar_kws={"shrink": 0.2, "aspect": 15, "ticks": [vmin, vmax]})

    # Highlight the diagonal squares with value 1 by setting their background color to black
    for i in range(len(miRNAs)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color='black', alpha=0.5, zorder=3))
        
    ax.xaxis.tick_top()  # Move x-axis labels to the top
    ax.xaxis.set_label_position('top')  # Set x-axis label position to top
    plt.xticks(rotation=-30, fontsize=10, ha='right')  # Rotate x-axis labels, set font size, and align to the right
    plt.yticks(rotation=0, fontsize=10)  # Set font size and rotation for y-axis labels

    # Draw short lines on the left of the leftmost column
    for i in range(len(miRNAs)):
        plt.plot([-0.5, -0.45], [i + 0.5, i + 0.5], color='black', linestyle='-', linewidth=1)

    # Set the title below the plot
    ax.text(x=0.5, y=-0.03, s="miRNA-miRNA similarities", fontsize=16, ha='center', va='top', transform=ax.transAxes)

    # Adjust the plot and color bar position to move it slightly to the bottom-right
    box = ax.get_position()
    ax.set_position([box.x0 + 0.14, box.y0, box.width * 0.9, box.height * 0.9])

    # Move the color bar as well
    cbar = ax.collections[0].colorbar
    cbar.ax.set_position([box.x1 + 0.1, box.y0 - 0.03, 0.01, box.height])

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def _plot_cosine_similarity_matrix_for_clusters_with_values(embeddings, diseases, save_path):
    cos_sim = np.dot(embeddings, np.array(embeddings).T)
    norms = np.linalg.norm(embeddings, axis=1)
    cos_sim /= np.outer(norms, norms)

    plt.figure(figsize=(10, 8))
    
    vmin = cos_sim.min()
    vmax = cos_sim.max()
    # Create the heatmap with a custom color bar
    ##sns.heatmap(data, cmap='cividis')
    ##sns.heatmap(data, cmap='Blues') 'Greens' sns.heatmap(data, cmap='Spectral') 'coolwarm') 'YlGnBu') viridis cubehelix inferno

    ax = sns.heatmap(cos_sim, cmap="Spectral", annot=True, fmt=".3f", annot_kws={"size": 6},
                     xticklabels=diseases, yticklabels=diseases,
                     cbar_kws={"shrink": 0.2, "aspect": 15, "ticks": [vmin, vmax]})

    # Highlight the diagonal squares with value 1 by setting their background color to black
    for i in range(len(diseases)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color='black', alpha=0.5, zorder=3))
        
    ax.xaxis.tick_top()  # Move x-axis labels to the top
    ax.xaxis.set_label_position('top')  # Set x-axis label position to top
    plt.xticks(rotation=-30, fontsize=8, ha='right')  # Rotate x-axis labels, set font size, and align to the right
    plt.yticks(fontsize=8)  # Set font size for y-axis labels

    # Set the title below the plot
    ax.text(x=0.5, y=-0.03, s="Pathway-pathway similarities", fontsize=12, ha='center', va='top', transform=ax.transAxes)

    plt.savefig(save_path, bbox_inches='tight')
    ##plt.show()
    plt.close()
       
def read_gene_names(file_path):
    """
    Reads the gene names from a CSV file and returns a dictionary mapping gene IDs to gene names.

    Parameters:
    file_path (str): Path to the gene names CSV file.

    Returns:
    dict: A dictionary mapping gene IDs to gene names.
    """
    gene_id_to_name_mapping = {}
    gene_id_to_symbol_mapping = {}

    # Read the gene names CSV file and populate the dictionary
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            gene_id = row['NCBI_Gene_ID']
            gene_name = row['Name']
            gene_symbol = row['Approved symbol']
            gene_id_to_name_mapping[gene_id] = gene_name
            gene_id_to_symbol_mapping[gene_id] = gene_symbol

    return gene_id_to_name_mapping, gene_id_to_symbol_mapping

def _plot_cosine_similarity_matrix_for_clusters_with_values(embeddings, miRNAs, save_path):
    cos_sim = np.dot(embeddings, np.array(embeddings).T)
    norms = np.linalg.norm(embeddings, axis=1)
    cos_sim /= np.outer(norms, norms)

    plt.figure(figsize=(10, 8))
    
    vmin = cos_sim.min()
    vmax = cos_sim.max()
    # Create the heatmap with a custom color bar
    ##sns.heatmap(data, cmap='cividis')
    ##sns.heatmap(data, cmap='Blues') 'Greens' sns.heatmap(data, cmap='Spectral') 'coolwarm') 'YlGnBu') viridis cubehelix inferno

    ax = sns.heatmap(cos_sim, cmap="Spectral", annot=True, fmt=".3f", annot_kws={"size": 6},
                     xticklabels=miRNAs, yticklabels=miRNAs,
                     cbar_kws={"shrink": 0.2, "aspect": 15, "ticks": [vmin, vmax]})

    # Highlight the diagonal squares with value 1 by setting their background color to black
    for i in range(len(miRNAs)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color='black', alpha=0.5, zorder=3))
        
    ax.xaxis.tick_top()  # Move x-axis labels to the top
    ax.xaxis.set_label_position('top')  # Set x-axis label position to top
    plt.xticks(rotation=-30, fontsize=8, ha='right')  # Rotate x-axis labels, set font size, and align to the right
    plt.yticks(fontsize=8)  # Set font size for y-axis labels

    # Set the title below the plot
    ax.text(x=0.5, y=-0.03, s="miRNA-miRNA similarities", fontsize=12, ha='center', va='top', transform=ax.transAxes)

    plt.savefig(save_path, bbox_inches='tight')
    ##plt.show()
    plt.close()
    
def create_pathway_map(reactome_file, output_file):
    """
    Extracts gene IDs with the same pathway miRNA and saves them to a new CSV file.

    Parameters:
    reactome_file (str): Path to the NCBI2Reactome.csv file.
    output_file (str): Path to save the output CSV file.
    """
    pathway_map = {}  # Dictionary to store gene IDs for each pathway miRNA

    # Read the NCBI2Reactome.csv file and populate the pathway_map
    with open(reactome_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            gene_id = row[0]
            pathway_miRNA = row[1]
            pathway_map.setdefault(pathway_miRNA, []).append(gene_id)

    # Write the pathway_map to the output CSV file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Pathway miRNA", "Gene IDs"])  # Write header
        for pathway_miRNA, gene_ids in pathway_map.items():
            writer.writerow([pathway_miRNA, ",".join(gene_ids)])
    
    return pathway_map
        
def save_to_neo4j(graph, miRNA_dic, miRNA_mapping, pathway_map, gene_id_to_name_mapping, gene_id_to_symbol_mapping, uri, user, password):
    from neo4j import GraphDatabase

    # Connect to Neo4j
    driver = GraphDatabase.driver(uri, auth=(user, password))
    session = driver.session()

    # Clean the database
    session.run("MATCH (n) DETACH DELETE n")

    try:
        # Create nodes with embeddings and additional attributes
        for node_id in miRNA_dic:
            embedding = miRNA_dic[node_id].tolist()  
            miRNA = miRNA_mapping[node_id]  # Access miRNA based on node_id
            name = graph.graph_nx.nodes[node_id]['name']
            weight = graph.graph_nx.nodes[node_id]['weight']
            significance = graph.graph_nx.nodes[node_id]['significance']
            session.run(
                "CREATE (n:Pathway {embedding: $embedding, miRNA: $miRNA, name: $name, weight: $weight, significance: $significance})",
                embedding=embedding, miRNA=miRNA, name=name, weight=weight, significance=significance
            )

            # Create gene nodes and relationships
            ##genes = get_genes_by_pathway_miRNA(node_id, reactome_file, gene_names_file)
            genes = pathway_map.get(node_id, [])


            ##print('miRNA_to_gene_info=========================-----------------------------\n', genes)
    
            # Create gene nodes and relationships
            for gene_id in genes:
                gene_name = gene_id_to_name_mapping.get(gene_id, None)
                gene_symbol = gene_id_to_symbol_mapping.get(gene_id, None)
                if gene_name:  # Only create node if gene name exists
                    session.run(
                        "MERGE (g:Gene {id: $gene_id, name: $gene_name, symbol: $gene_symbol})",
                        gene_id=gene_id, gene_name=gene_name, gene_symbol = gene_symbol
                    )
                    session.run(
                        "MATCH (p:Pathway {miRNA: $miRNA}), (g:Gene {id: $gene_id}) "
                        "MERGE (p)-[:INVOLVES]->(g)",
                        miRNA=miRNA, gene_id=gene_id
                    )
                
                session.run(
                    "MATCH (p:Pathway {miRNA: $miRNA}), (g:Gene {id: $gene_id}) "
                    "MERGE (p)-[:INVOLVES]->(g)",
                    miRNA=miRNA, gene_id=gene_id
                )
                
        # Create relationships using the miRNA mapping
        for source, target in graph.graph_nx.edges():
            source_miRNA = miRNA_mapping[source]
            target_miRNA = miRNA_mapping[target]
            session.run(
                "MATCH (a {miRNA: $source_miRNA}), (b {miRNA: $target_miRNA}) "
                "CREATE (a)-[:CONNECTED]->(b)",
                source_miRNA=source_miRNA, target_miRNA=target_miRNA
            )

    finally:
        session.close()
        driver.close()

def read_gene_names(file_path):
    """
    Reads the gene names from a CSV file and returns a dictionary mapping gene IDs to gene names.

    Parameters:
    file_path (str): Path to the gene names CSV file.

    Returns:
    dict: A dictionary mapping gene IDs to gene names.
    """
    gene_id_to_name_mapping = {}
    gene_id_to_symbol_mapping = {}

    # Read the gene names CSV file and populate the dictionary
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            gene_id = row['NCBI_Gene_ID']
            gene_name = row['Name']
            gene_symbol = row['Approved symbol']
            gene_id_to_name_mapping[gene_id] = gene_name
            gene_id_to_symbol_mapping[gene_id] = gene_symbol

    return gene_id_to_name_mapping, gene_id_to_symbol_mapping

def _25_most_labels_create_heatmap_with_miRNA(embedding_list, miRNA_list, save_path):
    # Convert the embedding list to a DataFrame
    heatmap_data = pd.DataFrame(embedding_list, index=miRNA_list)
    
    # Create a clustermap
    ax = sns.clustermap(heatmap_data, cmap='tab20', standard_scale=1, figsize=(10, 10))
    # Set smaller font sizes for various elements
    ax.ax_heatmap.tick_params(axis='both', which='both', labelsize=8)  # Tick labels
    ax.ax_heatmap.set_xlabel(ax.ax_heatmap.get_xlabel(), fontsize=8)  # X-axis label
    ax.ax_heatmap.set_ylabel(ax.ax_heatmap.get_ylabel(), fontsize=8)  # Y-axis label
    ax.ax_heatmap.collections[0].colorbar.ax.tick_params(labelsize=8)  # Color bar labels
    
    # Save the clustermap to a file
    plt.savefig(save_path)

    plt.close()

def _create_heatmap_with_miRNA(embedding_list, miRNA_list, save_path):
    # Create DataFrame
    heatmap_data = pd.DataFrame(embedding_list, index=miRNA_list)
    
    # Transpose DataFrame to rotate 90 degrees clockwise
    heatmap_data = heatmap_data.T

    # Generate distinct colors using Seaborn color palette
    num_colors = len(miRNA_list)
    palette = sns.color_palette('dark', num_colors)  # Using 'dark' palette for distinct colors
    color_dict = {miRNA: palette[i] for i, miRNA in enumerate(miRNA_list)}

    plt.figure(figsize=(10, 10))  # Square figure size
    ax = sns.heatmap(heatmap_data, cmap='tab20', cbar_kws={'label': 'Mean embedding value', 'orientation': 'horizontal', 'fraction': 0.046, 'pad': 0.04})
    
    plt.xlabel('miRNAs')
    plt.ylabel('Dimension of the embeddings', labelpad=0) 
    
    # Customize the color bar to be small and on top
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(size=0, labelsize=8)
    cbar.set_label('Mean embedding value', size=10)
    cbar.ax.set_position([0.2, 0.92, 0.6, 0.03])  # [left, bottom, width, height]

    # Add custom color patches above each column
    for i, miRNA in enumerate(miRNA_list):
        color = color_dict[miRNA]
        rect = mpatches.Rectangle((i, heatmap_data.shape[0]), 1, 2, color=color, transform=ax.get_xaxis_transform(), clip_on=False)
        ax.add_patch(rect)

    # Adjust the axis limits to make space for the patches
    ax.set_xlim(0, len(miRNA_list))
    ax.set_ylim(0, heatmap_data.shape[0] + 1.5)

    # Custom x-axis labels with shorter ticks
    ax.set_xticks(np.arange(len(miRNA_list)) + 0.5)
    ax.set_xticklabels(miRNA_list, rotation=45, fontsize=8, ha='right')
    ax.tick_params(axis='x', length=5)  # Shorten the x-axis ticks

    # Custom y-axis labels, only plot the first and last dimension numbers
    y_tick_labels = [heatmap_data.index[0], heatmap_data.index[-1]]
    y_ticks = [0.5, len(heatmap_data.index) - 0.5]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=8)
    ax.tick_params(axis='y', length=0)  # Remove y-axis ticks

    plt.savefig(save_path)
    plt.close()

def create_heatmap_with_miRNA_(embedding_list, miRNA_list, save_path):
    # Create DataFrame
    heatmap_data = pd.DataFrame(embedding_list, index=miRNA_list)
    
    # Transpose DataFrame to rotate 90 degrees clockwise
    heatmap_data = heatmap_data.T

    # Generate distinct colors using Seaborn color palette
    num_colors = len(miRNA_list)
    palette = sns.color_palette('dark', num_colors)  # Using 'dark' palette for distinct colors
    color_dict = {miRNA: palette[i] for i, miRNA in enumerate(miRNA_list)}

    plt.figure(figsize=(10, 10))  # Square figure size
    ax = sns.heatmap(heatmap_data, cmap='tab20', cbar_kws={'label': 'Mean embedding value', 'orientation': 'horizontal', 'fraction': 0.046, 'pad': 0.04})
    
    plt.xlabel('Human pathways')
    plt.ylabel('Dimension of the embeddings', labelpad=0) 
    
    # Customize the color bar to be small and on top
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(size=0, labelsize=8)
    cbar.set_label('Mean embedding value', size=10)
    cbar.ax.set_position([0.2, 0.92, 0.6, 0.03])  # [left, bottom, width, height]

    # Add custom color patches above each column
    for i, miRNA in enumerate(miRNA_list):
        color = color_dict[miRNA]
        rect = mpatches.Rectangle((i, heatmap_data.shape[0]), 1, 2, color=color, transform=ax.get_xaxis_transform(), clip_on=False)
        ax.add_patch(rect)

    # Adjust the axis limits to make space for the patches
    ax.set_xlim(0, len(miRNA_list))
    ax.set_ylim(0, heatmap_data.shape[0] + 1.5)

    # Custom x-axis labels with shorter ticks
    ax.set_xticks(np.arange(len(miRNA_list)) + 1.0)
    ax.set_xticklabels(miRNA_list, rotation=30, fontsize=8, ha='right')
    ax.tick_params(axis='x', length=5)  # Shorten the x-axis ticks

    # Custom y-axis labels, only plot the first and last dimension numbers
    y_tick_labels = [heatmap_data.index[0], heatmap_data.index[-1]]
    y_ticks = [0.5, len(heatmap_data.index) - 0.5]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=8)
    ax.tick_params(axis='y', length=0)  # Remove y-axis ticks

    plt.savefig(save_path)
    plt.close()   

def calculate_cluster_labels_(net, dataloader, device, num_clusters=9):
    all_embeddings = []
    net.eval()
    with torch.no_grad():
        for data in dataloader:
            graph, name = data
            embeddings = net.get_node_embeddings(graph.to(device))
            ##all_embeddings.append(embeddings)
    ##all_embeddings = np.concatenate(all_embeddings, axis=0)
    first_item = name[0].split('.')[0]
    print('graph-------------\n',graph)
    print('first_item-------------\n',first_item)
    # Use KMeans clustering to assign cluster labels
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return embeddings, cluster_labels, first_item

def _calculate_cluster_labels_miRNA(net, dataloader, device, num_clusters=9):
    all_embeddings = []
    all_names = []
    
    net.eval()
    with torch.no_grad():
        for data in dataloader:
            graph, name = data
            print('graph==================\n', graph)
            
            # Get node embeddings
            embeddings = net.get_node_embeddings(graph.to(device))
            
            # Debug: Print available node attributes
            print("Available node attributes:", graph.ndata.keys())
            
            # Check if 'node_type' exists in the graph
            if 'node_type' not in graph.ndata:
                raise KeyError("The node attribute 'node_type' is not present in the graph.")
            
            # Extract node types and filter for 'miRNA'
            node_types = graph.ndata['node_type']  # Assuming 'node_type' contains node type information
            miRNA_mask = node_types == 'miRNA'  # Create a mask for 'miRNA' nodes
            
            # Filter embeddings and names for 'miRNA' nodes
            miRNA_embeddings = embeddings[miRNA_mask]
            miRNA_names = [n for i, n in enumerate(name) if node_types[i] == 'miRNA']
            
            all_embeddings.append(miRNA_embeddings)
            all_names.extend(miRNA_names)
    
    all_embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()
    
    # Use KMeans clustering to assign cluster labels
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(all_embeddings)
    
    return all_embeddings, cluster_labels, all_names

def calculate_cluster_labels_miRNA_(net, dataloader, device, num_clusters=9):
    all_embeddings = []
    all_names = []
    
    net.eval()
    with torch.no_grad():
        for data in dataloader:
            graph, name = data
            print('graph==================\n',graph)
            embeddings = net.get_node_embeddings(graph.to(device))
            
            # Extract node types and filter for 'miRNA'
            node_types = graph.ndata['node_type']  # Assuming 'node_type' contains node type information
            miRNA_mask = node_types == 'miRNA'  # Create a mask for 'miRNA' nodes
            
            # Filter embeddings and names for 'miRNA' nodes
            miRNA_embeddings = embeddings[miRNA_mask]
            miRNA_names = [n for i, n in enumerate(name) if node_types[i] == 'miRNA']
            
            all_embeddings.append(miRNA_embeddings)
            all_names.extend(miRNA_names)
    
    all_embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()
    
    # Use KMeans clustering to assign cluster labels
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(all_embeddings)
    
    return all_embeddings, cluster_labels, all_names

def _calculate_cluster_labels(net, dataloader, device, num_clusters=9):
    all_embeddings = []
    net.eval()
    with torch.no_grad():
        for data in dataloader:
            graph, _ = data
            print('graph-------------\n',graph)
            embeddings = net.get_node_embeddings(graph.to(device))
            all_embeddings.append(embeddings)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Use KMeans clustering to assign cluster labels
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(all_embeddings)
    return all_embeddings, cluster_labels

def visualize_embeddings_pca(embeddings, cluster_labels, miRNA_list, save_path):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))  # Square figure

    # Set the style
    sns.set(style="whitegrid")

    # Define unique clusters and sort them
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)  # Sort the clusters

    # Define a color palette
    palette = sns.color_palette("viridis", len(sorted_clusters))

    # Create a scatter plot with a continuous colormap
    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{miRNA_list[cluster]}', s=20, color=palette[i], edgecolor='k')

    # Add labels and title
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Embeddings')

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    # Ensure the plot is square
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend with dot shapes and miRNA labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8, label=miRNA_list[cluster]) for i, cluster in enumerate(sorted_clusters)]
    plt.legend(handles=handles, title='Label', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., fontsize='small', handlelength=0.5, handletextpad=0.5)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
def visualize_embeddings_pca_ori(embeddings, cluster_labels, miRNA_list, save_path):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))  # Square figure

    # Set the style
    sns.set(style="whitegrid")

    # Define unique clusters and sort them
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)  # Sort the clusters

    # Define a color palette
    palette = sns.color_palette("viridis", len(sorted_clusters))

    # Create a scatter plot with a continuous colormap
    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{miRNA_list[cluster]}', s=20, color=palette[i])

    # Add labels and title
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Embeddings')

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    # Ensure the plot is square
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend with dot shapes and miRNA labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8, label=miRNA_list[cluster]) for i, cluster in enumerate(sorted_clusters)]
    plt.legend(handles=handles, title='Label', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., fontsize='small', handlelength=0.5, handletextpad=0.5)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    
def _visualize_embeddings_tsne(embeddings, cluster_labels, miRNA_list, save_path):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))  # Square figure

    # Set the style
    sns.set(style="whitegrid")

    # Define unique clusters and sort them
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)  # Sort the clusters

    # Define a color palette
    palette = sns.color_palette("viridis", len(sorted_clusters))
    
    # Create a scatter plot with a continuous colormap
    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{miRNA_list[cluster]}', s=20, color=palette[i], edgecolor='k')

    # Add labels and title
    plt.xlabel('dim_1')
    plt.ylabel('dim_2')
    plt.title('T-SNE of Embeddings')

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    # Ensure the plot is square
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend with dot shapes and miRNA labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8, label=miRNA_list[cluster]) for i, cluster in enumerate(sorted_clusters)]
    plt.legend(handles=handles, title='Label', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., fontsize='small', handlelength=0.5, handletextpad=0.5)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def draw_loss_plot(train_loss, valid_loss, save_path):
    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='validation')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility
    
    plt.savefig(f'{save_path}')
    plt.close()

def draw_max_f1_plot(max_train_f1, max_valid_f1, save_path):
    plt.figure()
    plt.plot(max_train_f1, label='train')
    plt.plot(max_valid_f1, label='validation')
    plt.title('Max F1-score over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()
    plt.savefig(f'{save_path}')
    plt.close()

def draw_f1_plot(train_f1, valid_f1, save_path):
    plt.figure()
    plt.plot(train_f1, label='train')
    plt.plot(valid_f1, label='validation')
    plt.title('F1-score over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    plt.savefig(f'{save_path}')
    plt.close()
    
def create_heatmap_with_disease(embedding_list, disease_list, save_path):
    # Convert the embedding list to a DataFrame
    heatmap_data = pd.DataFrame(embedding_list, index=disease_list)
    
    # Create a clustermap
    ax = sns.clustermap(heatmap_data, cmap='tab20', standard_scale=1, figsize=(13, 10))
    
    # Set smaller font sizes for various elements
    ax.ax_heatmap.tick_params(axis='both', which='both', labelsize=8)  # Tick labels
    ax.ax_heatmap.set_xlabel(ax.ax_heatmap.get_xlabel(), fontsize=8)  # X-axis label
    ax.ax_heatmap.set_ylabel(ax.ax_heatmap.get_ylabel(), fontsize=8)  # Y-axis label
    ax.ax_heatmap.collections[0].colorbar.ax.tick_params(labelsize=8)  # Color bar labels
    
    # Limit the number of y-tick labels to 50
    max_labels = 30
    y_labels = heatmap_data.index
    if len(y_labels) > max_labels:
        step = len(y_labels) // max_labels
        y_ticks = range(0, len(y_labels), step)
        ax.ax_heatmap.set_yticks(y_ticks)
        ax.ax_heatmap.set_yticklabels([y_labels[i] for i in y_ticks], fontsize=8)
    else:
        ax.ax_heatmap.set_yticks(range(len(y_labels)))
        ax.ax_heatmap.set_yticklabels(y_labels, fontsize=8)
    
    # Save the clustermap to a file
    plt.savefig(save_path)
    plt.close()
 
def visualize_embeddings_pca(embeddings, cluster_labels, disease_list, save_path):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))  # Square figure

    # Set the style
    sns.set(style="whitegrid")

    # Define unique clusters and sort them
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)  # Sort the clusters

    # Define a color palette
    palette = sns.color_palette("viridis", len(sorted_clusters))

    # Create a scatter plot with a continuous colormap
    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{disease_list[cluster]}', s=20, color=palette[i], edgecolor='k')

    # Add labels and title
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Embeddings')

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    # Ensure the plot is square
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend with dot shapes and disease labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8, label=disease_list[cluster]) for i, cluster in enumerate(sorted_clusters)]
    plt.legend(handles=handles, title='Label', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., fontsize='small', handlelength=0.5, handletextpad=0.5)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
def visualize_embeddings_pca_ori(embeddings, cluster_labels, disease_list, save_path):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))  # Square figure

    # Set the style
    sns.set(style="whitegrid")

    # Define unique clusters and sort them
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)  # Sort the clusters

    # Define a color palette
    palette = sns.color_palette("viridis", len(sorted_clusters))

    # Create a scatter plot with a continuous colormap
    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{disease_list[cluster]}', s=20, color=palette[i])

    # Add labels and title
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Embeddings')

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    # Ensure the plot is square
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend with dot shapes and disease labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8, label=disease_list[cluster]) for i, cluster in enumerate(sorted_clusters)]
    plt.legend(handles=handles, title='Label', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., fontsize='small', handlelength=0.5, handletextpad=0.5)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def visualize_embeddings_tsne(embeddings, cluster_labels, miRNA_list, save_path):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 8))  # Square figure

    # Set the style
    sns.set(style="whitegrid", rc={'axes.facecolor': 'white'})

    # Define unique clusters and sort them
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)  # Sort the clusters

    # Create a colormap
    cmap = plt.get_cmap('tab20')

    # Create a scatter plot with larger dot sizes and colors from the colormap
    for i, cluster in enumerate(sorted_clusters):
        # Ensure that the cluster index is within the bounds of miRNA_list
        if cluster < len(miRNA_list):
            cluster_points = embeddings_2d[cluster_labels == cluster]
            color = cmap(i / len(sorted_clusters))  # Normalize the index to get color from cmap
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{miRNA_list[cluster]}', s=40, color=color, edgecolor='k')
        else:
            print(f"Cluster index {cluster} is out of bounds for miRNA_list.")

    # Add labels and title
    plt.xlabel('dim_1')
    plt.ylabel('dim_2')
    plt.title('T-SNE of Embeddings')

    # Customize the grid and background
    ax = plt.gca()
    ax.grid(False)  # Remove grid lines

    # Ensure the plot is square
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend with dot shapes and miRNA labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i / len(sorted_clusters)), markersize=10, label=miRNA_list[cluster])
               for i, cluster in enumerate(sorted_clusters) if cluster < len(miRNA_list)]
    
    plt.legend(handles=handles, title='Label', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., fontsize='small', handlelength=0.5, handletextpad=0.5)

    # Save the figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def index_out_of_range_visualize_embeddings_tsne(embeddings, cluster_labels, miRNA_list, save_path):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 8))  # Square figure

    # Set the style
    sns.set(style="whitegrid", rc={'axes.facecolor':'white'})

    # Define unique clusters and sort them
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)  # Sort the clusters

    # Create a colormap
    cmap = plt.get_cmap('tab20')

    # Create a scatter plot with larger dot sizes and colors from the colormap
    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]
        color = cmap(i / len(sorted_clusters))  # Normalize the index to get color from cmap
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{miRNA_list[cluster]}', s=40, color=color, edgecolor='k')

    # Add labels and title
    plt.xlabel('dim_1')
    plt.ylabel('dim_2')
    plt.title('T-SNE of Embeddings')

    # Customize the grid and background
    ax = plt.gca()
    ax.grid(False)  # Remove grid lines

    # Ensure the plot is square
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend with dot shapes and miRNA labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i / len(sorted_clusters)), markersize=10, label=miRNA_list[cluster]) for i, cluster in enumerate(sorted_clusters)]
    plt.legend(handles=handles, title='Label', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., fontsize='small', handlelength=0.5, handletextpad=0.5)

    # Save the figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def _ori_visualize_embeddings_tsne(embeddings, cluster_labels, disease_list, save_path):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))  # Square figure

    # Set the style
    sns.set(style="white")

    # Define unique clusters and sort them
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)  # Sort the clusters

    # Define a color palette
    palette = sns.color_palette("viridis", len(sorted_clusters))
    
    # Create a scatter plot with a continuous colormap
    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{disease_list[cluster]}', s=20, color=palette[i], edgecolor='k')

    # Add labels and title
    plt.xlabel('dim_1')
    plt.ylabel('dim_2')
    plt.title('T-SNE of Embeddings')

    # Customize the background and remove grid
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.grid(False)  # Remove grid

    # Ensure the plot is square
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend with dot shapes and disease labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8, label=disease_list[cluster]) for i, cluster in enumerate(sorted_clusters)]
    plt.legend(handles=handles, title='Label', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., fontsize='small', handlelength=0.5, handletextpad=0.5)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
   
def grid_visualize_embeddings_tsne(embeddings, cluster_labels, disease_list, save_path):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))  # Square figure

    # Set the style
    sns.set(style="whitegrid")

    # Define unique clusters and sort them
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)  # Sort the clusters

    # Define a color palette
    palette = sns.color_palette("viridis", len(sorted_clusters))
    
    # Create a scatter plot with a continuous colormap
    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{disease_list[cluster]}', s=20, color=palette[i], edgecolor='k')

    # Add labels and title
    plt.xlabel('dim_1')
    plt.ylabel('dim_2')
    plt.title('T-SNE of Embeddings')

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    # Ensure the plot is square
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend with dot shapes and disease labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8, label=disease_list[cluster]) for i, cluster in enumerate(sorted_clusters)]
    plt.legend(handles=handles, title='Label', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., fontsize='small', handlelength=0.5, handletextpad=0.5)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def draw_loss_plot(train_loss, valid_loss, save_path):
    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='validation')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Customize the background and remove grid
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.grid(False)  # Remove grid

    plt.savefig(f'{save_path}')
    plt.close()

def draw_accuracy_plot(train_acc, valid_acc, save_path):
    plt.figure()
    plt.plot(train_acc, label='train')
    plt.plot(valid_acc, label='validation')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Customize the background and remove grid
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.grid(False)
    
    plt.savefig(f'{save_path}')
    plt.close()
   
def draw_f1_plot(train_f1, valid_f1, save_path):
    plt.figure()
    plt.plot(train_f1, label='train')
    plt.plot(valid_f1, label='validation')
    plt.title('F1 Score over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    # Customize the background and remove grid
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.grid(False)  # Remove grid

    plt.savefig(f'{save_path}')
    plt.close()

def draw_loss_plot_grid(train_loss, valid_loss, save_path):
    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='validation')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility
    
    plt.savefig(f'{save_path}')
    plt.close()

def draw_max_f1_plot(max_train_f1, max_valid_f1, save_path):
    plt.figure()
    plt.plot(max_train_f1, label='train')
    plt.plot(max_valid_f1, label='validation')
    plt.title('Max F1-score over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()
    plt.savefig(f'{save_path}')
    plt.close()

def draw_f1_plot_grid(train_f1, valid_f1, save_path):
    plt.figure()
    plt.plot(train_f1, label='train')
    plt.plot(valid_f1, label='validation')
    plt.title('F1-score over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    plt.savefig(f'{save_path}')
    plt.close()