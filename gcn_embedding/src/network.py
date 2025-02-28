import math
import json
import urllib.request
from collections import defaultdict, namedtuple
from datetime import datetime
import networkx as nx
import pandas as pd
from py2neo import Graph, Node, Relationship
from networkx.algorithms.traversal.depth_first_search import dfs_tree

import networkx as nx
import pandas as pd

class Network:
    def __init__(self, csv_file_path, kge=None):
        # Set KGE or default to timestamp
        self.kge = kge if kge is not None else datetime.now().strftime('%Y-%b-%d-%H-%M')
        
        # Load data from the CSV file
        self.data = self.load_csv(csv_file_path)
        
        # Create the NetworkX graph
        self.graph_nx = self.to_networkx()

    def load_csv(self, csv_file_path):
        """Load the CSV data into a pandas DataFrame."""
        df = pd.read_csv(csv_file_path)
        return df

    def set_node_attributes(self):
        """Set node attributes for miRNAs and diseases."""
        miRNAs = {}
        diseases = {}
        weights = {}
        significances = {}

        for _, row in self.data.iterrows():
            miRNA = row['miRNA']
            disease = row['disease']
            p_value = row['adjPvalue']
            significance = row['significance']

            miRNAs[miRNA] = miRNA
            diseases[disease] = disease
            weights[miRNA] = p_value
            weights[disease] = p_value
            significances[miRNA] = significance
            significances[disease] = significance

        return miRNAs, diseases, weights, significances

    def to_networkx(self):
        """Convert the data to a NetworkX DiGraph."""
        graph_nx = nx.DiGraph()

        # Add nodes and edges with attributes
        for _, row in self.data.iterrows():
            miRNA = row['miRNA']
            disease = row['disease']
            p_value = row['adjPvalue']
            significance = row['significance']

            # Add nodes with type attribute
            graph_nx.add_node(miRNA, p_value=p_value, significance=significance, node_type='miRNA')
            graph_nx.add_node(disease, p_value=p_value, significance=significance, node_type='disease')
        
            
            # Add an edge from miRNA to disease
            graph_nx.add_edge(miRNA, disease, weight=p_value, significance=significance)

        return graph_nx

class Network_:
    def __init__(self, csv_file_path, kge=None):
        # Set KGE or default to timestamp
        self.kge = kge if kge is not None else datetime.now().strftime('%Y-%b-%d-%H-%M')
        
        # Load data from the CSV file
        self.data = self.load_csv(csv_file_path)
        
        # Create the NetworkX graph
        self.graph_nx = self.to_networkx()

    def load_csv(self, csv_file_path):
        """Load the CSV data into a pandas DataFrame."""
        df = pd.read_csv(csv_file_path)
        return df

    def set_node_attributes(self):
        """Set node attributes for miRNAs and diseases."""
        miRNAs = {}
        diseases = {}
        weights = {}
        significances = {}

        for _, row in self.data.iterrows():
            miRNA = row['miRNA']
            disease = row['disease']
            p_value = row['adjPvalue']
            significance = row['significance']

            miRNAs[miRNA] = miRNA
            diseases[disease] = disease
            weights[miRNA] = p_value
            weights[disease] = p_value
            significances[miRNA] = significance
            significances[disease] = significance

        return miRNAs, diseases, weights, significances


    def to_networkx(self):
        """Convert the data to a NetworkX DiGraph."""
        graph_nx = nx.DiGraph()

        # Add nodes and edges with attributes
        for _, row in self.data.iterrows():
            miRNA = row['miRNA']
            disease = row['disease']
            p_value = row['adjPvalue']
            significance = row['significance']

            # Add nodes
            graph_nx.add_node(miRNA, p_value=p_value, significance=significance)
            graph_nx.add_node(disease, p_value=p_value, significance=significance)
            
            # Add an edge from miRNA to disease
            graph_nx.add_edge(miRNA, disease, weight=p_value, significance=significance)

        # Set node attributes for the graph
        miRNA_nodes, disease_nodes, weights, significances = self.set_node_attributes()

        # Update node type attributes
        nx.set_node_attributes(graph_nx, miRNA_nodes, 'node_type')
        nx.set_node_attributes(graph_nx, disease_nodes, 'node_type')
        nx.set_node_attributes(graph_nx, weights, 'weight')
        nx.set_node_attributes(graph_nx, significances, 'significance')

        return graph_nx


