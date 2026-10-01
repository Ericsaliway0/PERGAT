import json
import os
from matplotlib import pyplot as plt
import torch
import itertools
import dgl
import numpy as np
import scipy.sparse as sp
import pandas as pd  # Added import for pandas
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

def train_and_evaluate(args, G_dgl, node_features):
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
    val_neg_u, val_neg_v = neg_u[neg_eids[test_size:test_size + val_size]], neg_v[neg_eids[test_size + val_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size + val_size:]], neg_v[neg_eids[test_size + val_size:]]

    train_g = dgl.remove_edges(G_dgl, eids[:test_size + val_size])

    def create_graph(u, v, num_nodes):
        assert len(u) == len(v), "Source and destination nodes must have the same length"
        return dgl.graph((u, v), num_nodes=num_nodes)

    train_pos_g = create_graph(train_pos_u, train_pos_v, G_dgl.number_of_nodes())
    train_neg_g = create_graph(train_neg_u, train_neg_v, G_dgl.number_of_nodes())
    val_pos_g = create_graph(val_pos_u, val_pos_v, G_dgl.number_of_nodes())
    val_neg_g = create_graph(val_neg_u, val_neg_v, G_dgl.number_of_nodes())
    