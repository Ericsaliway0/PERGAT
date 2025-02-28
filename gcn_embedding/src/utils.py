import os
import pickle
import urllib.request
import json
from collections import defaultdict, namedtuple
from datetime import datetime
import networkx as nx
import pandas as pd
from py2neo import Graph, Node, Relationship
import torch
from torch import nn
from tqdm import tqdm
from src.dataset import Dataset
from src.network import Network
from src.model import GATModel
from sklearn.model_selection import train_test_split

import torch.nn.functional as F
from dgl.nn import GraphConv

import dgl

class GCNModel(nn.Module):
    def __init__(self, dim_latent: int, num_layers: int, do_train=False):
        super().__init__()
        self.do_train = do_train
        self.linear = nn.Linear(1, dim_latent)
        self.conv_0 = GraphConv(dim_latent, dim_latent, allow_zero_in_degree=True)
        self.relu = nn.LeakyReLU()
        self.layers = nn.ModuleList([GraphConv(dim_latent, dim_latent, allow_zero_in_degree=True)
                                     for _ in range(num_layers - 1)])
        self.predict = nn.Linear(dim_latent, 1)

    def forward(self, graph):
        weights = graph.ndata['weight'].unsqueeze(-1)
        features = self.linear(weights)
        graph = dgl.add_self_loop(graph)
        embedding = self.conv_0(graph, features)

        for conv in self.layers:
            embedding = self.relu(embedding)
            embedding = conv(graph, embedding)
        
        if not self.do_train:
            return embedding.detach()
        
        logits = self.predict(embedding)
        return logits

    def get_node_embeddings(self, graph):
        weights = graph.ndata['weight'].unsqueeze(-1)
        features = self.linear(weights)
        graph = dgl.add_self_loop(graph)
        embedding = self.conv_0(graph, features)

        for conv in self.layers:
            embedding = self.relu(embedding)
            embedding = conv(graph, embedding)

        return embedding

def get_miRNA_mapping(graph):
    miRNA_mapping = {}  # Mapping of node_id to miRNA
    for node_id, data in graph.graph_nx.nodes(data=True):
        miRNA = data['miRNA']
        miRNA_mapping[node_id] = miRNA  # Store the mapping
    return miRNA_mapping  # Return the miRNA mapping

def save_graph_to_neo4j(graph):
    from py2neo import Graph, Node, Relationship

    neo4j_url = "neo4j+s://7ffb183d.databases.neo4j.io"
    user = "neo4j"
    password = "BGc2jKUI44h_awhU5gEp8NScyuyx-iSSkTbFHEHJRpY"
    
    neo4j_graph = Graph(neo4j_url, auth=(user, password))

    # Clear the existing graph
    neo4j_graph.delete_all()

    # Create nodes
    nodes = {}
    for node_id, data in graph.graph_nx.nodes(data=True):
        miRNA = data['miRNA']
        node = Node("Pathway", miRNA=miRNA, name=data['name'], weight=data['weight'], significance=data['significance'])
        nodes[node_id] = node
        neo4j_graph.create(node)

    # Create relationships
    for source, target in graph.graph_nx.edges():
        relationship = Relationship(nodes[source], "parent-child", nodes[target])
        neo4j_graph.create(relationship)

def create_network(path_miRNA_csv, kge):
    graph = Network(path_miRNA_csv, kge)
    return graph


def save_to_disk(graph, save_dir):
    print('save_dir=======================================', save_dir)
    assert os.path.isdir(save_dir), 'Directory does not exist!'
    save_path = os.path.join(save_dir, graph.kge + '.pkl')
    pickle.dump(graph.graph_nx, open(save_path, 'wb'))

def save_miRNA_to_csv(graph, save_dir):
    assert os.path.isdir(save_dir), 'Directory does not exist!'
    miRNA_data = {'miRNA': [node['miRNA'] for node in graph.graph_nx.nodes.values()]}
    df = pd.DataFrame(miRNA_data)
    csv_path = os.path.join(save_dir, 'miRNA_nodes.csv')
    df.to_csv(csv_path, index=False)

def create_graphs_50(save=True, data_dir='data/emb'):
    num_graphs = 20
    graphs = {}

    # Define the save directory
    save_dir = os.path.join(data_dir, 'raw')
    os.makedirs(save_dir, exist_ok=True)

    for i in range(1, num_graphs + 1):
        # Construct the filename for each part
        file_path = os.path.join('data/split_mirna_files_50', f'mirna_p_value_results_part_{i}.csv')

        # Create a network graph from the CSV file
        graph = create_network(file_path, str(i))  # Use index as the name

        # Store the graph with its index as the key
        graphs[i] = graph

        if save:
            # Save each graph to the defined directory with the index as part of the filename
            save_to_disk(graph, os.path.join(save_dir, f'{i}.pkl'))
    
    return graphs

def create_graphs_(save=True, data_dir='data/emb'):
    num_graphs = 20
    graphs = {}

    # Define the save directory
    save_dir = os.path.join(data_dir, 'raw')
    os.makedirs(save_dir, exist_ok=True)

    for i in range(1, num_graphs + 1):
        # Construct the filename for each part
        file_path = os.path.join('data/split_mirna_files', f'mirna_p_value_results_part_{i}.csv')
        emb_name = f'emb_{i}'

        # Create a network graph from the CSV file
        graph = create_network(file_path, emb_name)
        graphs[i] = graph

        if save:
            # Save each graph to the defined directory
            save_to_disk(graph, os.path.join(save_dir, f'{emb_name}.pkl'))
    
    return graphs

def x_create_graphs(save=True, data_dir='data/emb'):
    num_graphs = 20
    graphs = {}

    for i in range(1, num_graphs + 1):
        # Construct the filename for each part
        file_path = os.path.join('data/split_mirna_files', f'mirna_p_value_results_part_{i}.csv')
        emb_name = f'emb_{i}'

        # Create a network graph from the CSV file
        graph = create_network(file_path, emb_name)
        graphs[i] = graph

        if save:
            # Define the save directory for the graphs
            save_dir = os.path.join(data_dir, 'raw')
            os.makedirs(save_dir, exist_ok=True)
            
            # Save each graph
            save_to_disk(graph, os.path.join(save_dir, emb_name))
    
    return graphs

def create_graphs(save=True, data_dir='data/emb'):
    graph_train = create_network('data/mirna_p_value_results_dbDEMC_train.csv', 'emb_train')
    graph_test = create_network('data/mirna_p_value_results_dbDEMC_test.csv', 'emb_test')

    print('creating graph=======================\n')
    if save:
        save_dir = os.path.join(data_dir, 'raw')
        save_to_disk(graph_train, save_dir)
        save_to_disk(graph_test, save_dir)

    return graph_train, graph_test

def create_embeddings_gcn(load_model=True, save=True, data_dir='data/emb', hyperparams=None, plot=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = Dataset(data_dir)
    emb_dir = os.path.abspath(os.path.join(data_dir, 'embeddings'))
    os.makedirs(emb_dir, exist_ok=True)

    in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams.get('num_heads', 2)  # Default to 2 heads if not specified
    feat_drop = hyperparams['feat_drop']
    attn_drop = hyperparams['attn_drop']

    net = GATModel(
            in_feats=in_feats,  
            out_feats=out_feats, 
            num_layers=num_layers, 
            num_heads=num_heads, 
            feat_drop=feat_drop, 
            attn_drop=attn_drop, 
            do_train=True
        ).to(device)

    if load_model:
        model_path = os.path.abspath(os.path.join(data_dir, 'models/model.pth'))
        net.load_state_dict(torch.load(model_path))
    else:
        from src.train import train
        model_path = train(hyperparams=hyperparams, data_path=data_dir, plot=plot)
        net.load_state_dict(torch.load(model_path))

    embedding_dict = {}
    
    for idx in tqdm(range(len(data))):
        graph, name = data[idx]
        graph = graph.to(device)  # Move graph to the same device as net
        
        with torch.no_grad():
            embedding = net(graph)
        embedding_dict[name] = embedding.cpu()
        if save:
            emb_path = os.path.join(emb_dir, f'{name[:-4]}.pth')
            torch.save(embedding.cpu(), emb_path)

    return embedding_dict

def create_embeddings(load_model=True, save=True, data_dir='data/emb', hyperparams=None, plot=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = Dataset(data_dir)
    emb_dir = os.path.abspath(os.path.join(data_dir, 'embeddings'))
    if not os.path.isdir(emb_dir):
        os.mkdir(emb_dir)

    dim_latent = hyperparams['out_feats']
    # dim_latent = hyperparams['dim_latent']
    num_layers = hyperparams['num_layers']
    
    net = GCNModel(dim_latent=dim_latent, num_layers=num_layers).to(device)

    if load_model:
        model_path = os.path.abspath(os.path.join(data_dir, 'models/model.pth'))
        net.load_state_dict(torch.load(model_path))
    else:
        from src.train import train
        model_path = train(hyperparams=hyperparams, data_path=data_dir, plot=plot)
        net.load_state_dict(torch.load(model_path))

    embedding_dict = {}
    
    for idx in range(len(data)):
        graph, name = data[idx]
        graph = graph.to(device)  # Move graph to the same device as net
        
        with torch.no_grad():
            embedding = net(graph)
        embedding_dict[name] = embedding
        if save:
            emb_path = os.path.join(emb_dir, f'{name[:-4]}.pth')
            torch.save(embedding.cpu(), emb_path)

    return embedding_dict

