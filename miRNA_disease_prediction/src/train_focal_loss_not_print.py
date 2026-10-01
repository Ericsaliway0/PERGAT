import json
import os
from matplotlib import pyplot as plt
import torch
import itertools
import dgl
import numpy as np
import scipy.sparse as sp
from torch import nn
import torch.nn.functional as F
from scipy.stats import sem
from .models import LinkPredictor, GATModel, MLPPredictor
from .utils import (
    compute_loss, compute_hits_k, compute_auc, compute_f1, compute_accuracy,
    compute_precision, compute_recall, compute_map,
    compute_auc_with_symmetrical_confidence, compute_f1_with_symmetrical_confidence,
    compute_accuracy_with_symmetrical_confidence, compute_precision_with_symmetrical_confidence,
    compute_recall_with_symmetrical_confidence, compute_map_with_symmetrical_confidence
)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss  

def compute_focalloss_with_symmetrical_confidence(predictions, targets, alpha=1, gamma=2):
    BCE_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
    pt = torch.exp(-BCE_loss)
    focal_loss = alpha * (1 - pt) ** gamma * BCE_loss
    
    focal_loss_value = focal_loss.mean().item()
    focal_loss_confidence_interval = sem(focal_loss.cpu().numpy()) * 1.96  # 95% confidence interval
    
    return focal_loss_value, focal_loss_confidence_interval

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

    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(G_dgl.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), G_dgl.number_of_edges(), replace=False)
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

    output_path = './link_prediction_gat/results/'
    os.makedirs(output_path, exist_ok=True)
    
    train_f1_scores = []
    val_f1_scores = []
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

    train_focal_losses = []
    val_focal_losses = []

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

            train_focal_loss, train_focal_loss_err = compute_focalloss_with_symmetrical_confidence(train_pos_score, train_neg_score)
            train_focal_losses.append(train_focal_loss)

            h_val = model(train_g, train_g.ndata['feat'])
            val_pos_score = pred(val_pos_g, h_val)
            val_neg_score = pred(val_neg_g, h_val)
            val_f1 = compute_f1(val_pos_score, val_neg_score)
            val_f1_scores.append(val_f1.item())
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

            val_focal_loss, val_focal_loss_err = compute_focalloss_with_symmetrical_confidence(val_pos_score, val_neg_score)
            val_focal_losses.append(val_focal_loss)

    epochs = range(args.epochs)
        
    def plot_scores(epochs, train_f1_scores, val_f1_scores, train_auc_scores, val_auc_scores, 
                    train_map_scores, val_map_scores, train_recall_scores, val_recall_scores,
                    train_acc_scores, val_acc_scores, train_precision_scores, val_precision_scores,
                    train_focal_losses, val_focal_losses, output_path, args):

        # Ensure the output directory exists
        os.makedirs(output_path, exist_ok=True)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_f1_scores, label='Training F1 Score')
        plt.plot(epochs, val_f1_scores, label='Validation F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.title('Training and Validation F1 Scores over Epochs')
        plt.legend()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.savefig(os.path.join(output_path, f'f1_head{args.num_heads}_out{args.out_feats}.png'))
        plt.close()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_auc_scores, label='Training AUC Score')
        plt.plot(epochs, val_auc_scores, label='Validation AUC Score')
        plt.xlabel('Epochs')
        plt.ylabel('AUC Score')
        plt.title('Training and Validation AUC Scores over Epochs')
        plt.legend()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.savefig(os.path.join(output_path, f'auc_head{args.num_heads}_out{args.out_feats}.png'))
        plt.close()

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_map_scores, label='Training MAP Score')
        plt.plot(epochs, val_map_scores, label='Validation MAP Score')
        plt.xlabel('Epochs')
        plt.ylabel('MAP Score')
        plt.title('Training and Validation MAP Scores over Epochs')
        plt.legend()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.savefig(os.path.join(output_path, f'map_head{args.num_heads}_out{args.out_feats}.png'))
        plt.close()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_recall_scores, label='Training Recall Score')
        plt.plot(epochs, val_recall_scores, label='Validation Recall Score')
        plt.xlabel('Epochs')
        plt.ylabel('Recall Score')
        plt.title('Training and Validation Recall Scores over Epochs')
        plt.legend()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.savefig(os.path.join(output_path, f'recall_head{args.num_heads}_out{args.out_feats}.png'))
        plt.close()

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_acc_scores, label='Training Accuracy Score')
        plt.plot(epochs, val_acc_scores, label='Validation Accuracy Score')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy Score')
        plt.title('Training and Validation Accuracy Scores over Epochs')
        plt.legend()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.savefig(os.path.join(output_path, f'acc_head{args.num_heads}_out{args.out_feats}.png'))
        plt.close()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_precision_scores, label='Training Precision Score')
        plt.plot(epochs, val_precision_scores, label='Validation Precision Score')
        plt.xlabel('Epochs')
        plt.ylabel('Precision Score')
        plt.title('Training and Validation Precision Scores over Epochs')
        plt.legend()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.savefig(os.path.join(output_path, f'precision_head{args.num_heads}_out{args.out_feats}.png'))
        plt.close()

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_focal_losses, label='Training Focal Loss')
        plt.plot(epochs, val_focal_losses, label='Validation Focal Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Focal Loss')
        plt.title('Training and Validation Focal Loss over Epochs')
        plt.legend()
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.savefig(os.path.join(output_path, f'focal_loss_head{args.num_heads}_out{args.out_feats}.png'))
        plt.close()

    plot_scores(epochs, train_f1_scores, val_f1_scores, train_auc_scores, val_auc_scores, 
                train_map_scores, val_map_scores, train_recall_scores, val_recall_scores,
                train_acc_scores, val_acc_scores, train_precision_scores, val_precision_scores,
                train_focal_losses, val_focal_losses, output_path, args)

    with open(f'./link_prediction_gat/results/head{args.num_heads}_out{args.out_feats}_training_metrics.json', 'w') as f:
        json.dump({
            "train_f1_scores": train_f1_scores,
            "val_f1_scores": val_f1_scores,
            "train_auc_scores": train_auc_scores,
            "val_auc_scores": val_auc_scores,
            "train_map_scores": train_map_scores,
            "val_map_scores": val_map_scores,
            "train_recall_scores": train_recall_scores,
            "val_recall_scores": val_recall_scores,
            "train_acc_scores": train_acc_scores,
            "val_acc_scores": val_acc_scores,
            "train_precision_scores": train_precision_scores,
            "val_precision_scores": val_precision_scores,
            "train_focal_losses": train_focal_losses,
            "val_focal_losses": val_focal_losses
        }, f)
