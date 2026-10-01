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

        # Extract source and target names
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
        '''if 'cancer' in target_name:
            print('source_name-----------------------------------------/n',source_name)
            print('target_name-----------------------------------------/n',target_name)'''

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


