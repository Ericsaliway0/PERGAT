import json
import os
from matplotlib import pyplot as plt
import torch
import itertools
import dgl
import numpy as np
import scipy.sparse as sp
from .models import LinkPredictor, GATModel
from .utils import compute_loss, compute_hits_k, compute_auc, compute_f1, compute_precision, compute_recall, compute_map, compute_auc_with_symmetrical_confidence, compute_f1_with_symmetrical_confidence, compute_precision_with_symmetrical_confidence, compute_recall_with_symmetrical_confidence, compute_map_with_symmetrical_confidence


def _train_and_evaluate(args, G_dgl, node_features):
    u, v = G_dgl.edges()
    eids = np.arange(G_dgl.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    train_size = G_dgl.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(G_dgl.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), G_dgl.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    train_g = dgl.remove_edges(G_dgl, eids[:test_size])

    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=G_dgl.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=G_dgl.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=G_dgl.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=G_dgl.number_of_nodes())

    pred = LinkPredictor(args.input_size, args.hidden_size)
    ##model = GraphSAGE(in_feats=node_features.size(1), out_feats=16, num_layers=2, do_train=True)
    model = GATModel(node_features.shape[1], out_feats=args.out_feats, num_layers=args.num_layers, num_heads=args.num_heads, do_train=True)
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.lr)
    


    for e in range(args.epochs):
        model.train()
        logits = model(train_g, train_g.ndata['feat'])
        logits.requires_grad_(True)
        h = model(train_g, train_g.ndata['feat'])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if e % 5 == 0:
            print(f'In epoch {e}, loss: {loss.item()}')



 
    '''
    model.eval()
    with torch.no_grad():
        h = model(train_g, train_g.ndata['feat'])
        pos_score = pred(test_pos_g, h)
        neg_score = pred(test_neg_g, h)
        auc = compute_auc(pos_score, neg_score)
        f1 = compute_f1(pos_score, neg_score)
        precision = compute_precision(pos_score, neg_score)
        recall = compute_recall(pos_score, neg_score)
        hits_k = compute_hits_k(pos_score, neg_score, k=20)
        mean_avg_precision = compute_map(pos_score, neg_score)
        
        print('AUC:', auc)
        print('F1 Score:', f1)
        print('Precision:', precision)
        print('Recall:', recall)
        print('Hits@20:', hits_k)
        print('mAP:', mean_avg_precision)

    auc = auc.item()
    f1 = f1.item()
    precision = precision.item()
    recall = recall.item()
    hits_k = hits_k.item()
    mean_avg_precision = mean_avg_precision.item()
    
    output = {'AUC': auc, 'F1 Score': f1, 'Precision': precision, 'Recall': recall, 'Hits@20': float(hits_k), 'mAP': float(mean_avg_precision)}
    output_path = './link_prediction_gcn/results/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filename = f'test_results_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.json'
    with open(os.path.join(output_path, filename), 'w') as f:
        json.dump(output, f)
    '''

    # Test the model
    model.eval()
    with torch.no_grad():
        h_test = model(G_dgl, G_dgl.ndata['feat'])
        test_pos_score = pred(test_pos_g, h_test)
        test_neg_score = pred(test_neg_g, h_test)
        test_auc, test_auc_err = compute_auc_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_f1, test_f1_err = compute_f1_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_precision, test_precision_err = compute_precision_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_recall, test_recall_err = compute_recall_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_hits_k = compute_hits_k(test_pos_score, test_neg_score, k=10)  # This needs an error range function
        test_map, test_map_err = compute_map_with_symmetrical_confidence(test_pos_score, test_neg_score)

        print(f'Test AUC: {test_auc:.4f} ± {test_auc_err:.4f} | Test F1: {test_f1:.4f} ± {test_f1_err:.4f} | Test Precision: {test_precision:.4f} ± {test_precision_err:.4f} | Test Recall: {test_recall:.4f} ± {test_recall_err:.4f} | Test mAP: {test_map:.4f} ± {test_map_err:.4f}')

    # Save the model
    model_path = './link_prediction_gat/results/pred_model.pth'
    torch.save(pred.state_dict(), model_path)

    output_path = './link_prediction_gat/results/'
    os.makedirs(output_path, exist_ok=True)


    test_auc = test_auc.item()
    test_f1 = test_f1.item()
    test_precision = test_precision.item()
    test_recall = test_recall.item()
    test_hits_k = test_hits_k.item()
    test_map = test_map.item()

    test_auc_err = test_auc_err.item()
    test_f1_err = test_f1_err.item()
    test_precision_err = test_precision_err.item()
    test_recall_err = test_recall_err.item()
    test_map_err = test_map_err.item()

    # Save output to the results folder in the specified format
    output = {
        'Test AUC': f'{test_auc:.4f} ± {test_auc_err:.4f}',
        'Test F1 Score': f'{test_f1:.4f} ± {test_f1_err:.4f}',
        'Test Precision': f'{test_precision:.4f} ± {test_precision_err:.4f}',
        'Test Recall': f'{test_recall:.4f} ± {test_recall_err:.4f}',
        'Test Hit': f'{test_hits_k:.4f}',  # Assuming no confidence interval for Hits@K
        'Test mAP': f'{test_map:.4f} ± {test_map_err:.4f}'
    }

    with open(os.path.join(output_path, 'test_results.json'), 'w') as f:
        json.dump(output, f)

    # Save the training and validation metrics
    '''metrics = {
        'Epoch': list(range(args.epochs)),
        'Train Loss': train_loss_list,
        'Validation AUC': val_auc_list,
        'Validation F1': val_f1_list,
        'Validation Precision': val_precision_list,
        'Validation Recall': val_recall_list,
        'Validation Hit': val_hit_list,
        'Validation mAP': val_map_list
    }

    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(output_path, 'metrics.csv'), index=False)
    '''



    # Generate the filename based on parameters
    ##filename = f'test_results_lr{args.lr}_lay{args.num_layers}_infeat{args.in_feats}_outfeat{args.out_feats}_epoch{args.epochs}.json'
    filename = f'test_results_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.json'

    # Save the test results to a JSON file in the specified format
    test_results = {
        'Learning Rate': args.lr,
        'Epochs': args.epochs,
        'Input Features': args.input_size,
        'Output Features': args.out_feats,
        'Test AUC': f'{test_auc:.4f} ± {test_auc_err:.4f}',
        'Test F1 Score': f'{test_f1:.4f} ± {test_f1_err:.4f}',
        'Test Precision': f'{test_precision:.4f} ± {test_precision_err:.4f}',
        'Test Recall': f'{test_recall:.4f} ± {test_recall_err:.4f}',
        'Test Hit': f'{test_hits_k:.4f}',
        'Test mAP': f'{test_map:.4f} ± {test_map_err:.4f}'
    }

    with open(os.path.join(output_path, filename), 'w') as f:
        json.dump(test_results, f)


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

    neg_eids = np.random.choice(len(neg_u), G_dgl.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    val_neg_u, val_neg_v = neg_u[neg_eids[test_size:test_size + val_size]], neg_v[neg_eids[test_size:test_size + val_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size + val_size:]], neg_v[neg_eids[test_size + val_size:]]

    train_g = dgl.remove_edges(G_dgl, eids[:test_size + val_size])

    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=G_dgl.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=G_dgl.number_of_nodes())
    val_pos_g = dgl.graph((val_pos_u, val_pos_v), num_nodes=G_dgl.number_of_nodes())
    val_neg_g = dgl.graph((val_neg_u, val_neg_v), num_nodes=G_dgl.number_of_nodes())
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=G_dgl.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=G_dgl.number_of_nodes())

    pred = LinkPredictor(args.input_size, args.hidden_size)
    ## model = GATModel(in_feats=node_features.size(1), hidden_feats=args.hidden_size, out_feats=args.dim_latent, num_layers=args.num_layers)
    model = GATModel(node_features.shape[1], out_feats=args.out_feats, num_layers=args.num_layers, num_heads=args.num_heads, do_train=True)
    ##model = GATModel(node_features.shape[1], out_feats=args.out_feats, hidden_feats=args.hidden_feats, input_size=args.input_size, num_layers=args.num_layers, num_heads=args.num_heads, do_train=True)
    
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.lr)
    
    train_f1_scores = []
    val_f1_scores = []

    for e in range(args.epochs):
        model.train()
        logits = model(train_g, train_g.ndata['feat'])
        logits.requires_grad_(True)
        h = model(train_g, train_g.ndata['feat'])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if e % 5 == 0:
            print(f'In epoch {e}, loss: {loss.item()}')

        model.eval()
        with torch.no_grad():
            h_train = model(train_g, train_g.ndata['feat'])
            train_pos_score = pred(train_pos_g, h_train)
            train_neg_score = pred(train_neg_g, h_train)
            train_f1 = compute_f1(train_pos_score, train_neg_score)
            train_f1_scores.append(train_f1.item())

            h_val = model(train_g, train_g.ndata['feat'])
            val_pos_score = pred(val_pos_g, h_val)
            val_neg_score = pred(val_neg_g, h_val)
            val_f1 = compute_f1(val_pos_score, val_neg_score)
            val_f1_scores.append(val_f1.item())

    # Plot the F1 scores
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_f1_scores)), train_f1_scores, label='Training F1 Score')
    plt.plot(range(len(val_f1_scores)), val_f1_scores, label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Scores over Epochs')
    plt.legend()
    plt.show()

    # Test the model
    model.eval()
    with torch.no_grad():
        h_test = model(G_dgl, G_dgl.ndata['feat'])
        test_pos_score = pred(test_pos_g, h_test)
        test_neg_score = pred(test_neg_g, h_test)
        test_auc, test_auc_err = compute_auc_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_f1, test_f1_err = compute_f1_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_precision, test_precision_err = compute_precision_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_recall, test_recall_err = compute_recall_with_symmetrical_confidence(test_pos_score, test_neg_score)
        test_hits_k = compute_hits_k(test_pos_score, test_neg_score, k=10)  # This needs an error range function
        test_map, test_map_err = compute_map_with_symmetrical_confidence(test_pos_score, test_neg_score)

        print(f'Test AUC: {test_auc:.4f} ± {test_auc_err:.4f} | Test F1: {test_f1:.4f} ± {test_f1_err:.4f} | Test Precision: {test_precision:.4f} ± {test_precision_err:.4f} | Test Recall: {test_recall:.4f} ± {test_recall_err:.4f} | Test mAP: {test_map:.4f} ± {test_map_err:.4f}')

    # Save the model
    model_path = './link_prediction_gat/results/pred_model.pth'
    torch.save(pred.state_dict(), model_path)

    output_path = './link_prediction_gat/results/'
    os.makedirs(output_path, exist_ok=True)


    test_auc = test_auc.item()
    test_f1 = test_f1.item()
    test_precision = test_precision.item()
    test_recall = test_recall.item()
    test_hits_k = test_hits_k.item()
    test_map = test_map.item()

    test_auc_err = test_auc_err.item()
    test_f1_err = test_f1_err.item()
    test_precision_err = test_precision_err.item()
    test_recall_err = test_recall_err.item()
    test_map_err = test_map_err.item()

    # Save output to the results folder in the specified format
    output = {
        'Test AUC': f'{test_auc:.4f} ± {test_auc_err:.4f}',
        'Test F1 Score': f'{test_f1:.4f} ± {test_f1_err:.4f}',
        'Test Precision': f'{test_precision:.4f} ± {test_precision_err:.4f}',
        'Test Recall': f'{test_recall:.4f} ± {test_recall_err:.4f}',
        'Test Hit': f'{test_hits_k:.4f}',  # Assuming no confidence interval for Hits@K
        'Test mAP': f'{test_map:.4f} ± {test_map_err:.4f}'
    }

    with open(os.path.join(output_path, 'test_results.json'), 'w') as f:
        json.dump(output, f)

    # Generate the filename based on parameters
    filename = f'test_results_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.json'

    # Save the test results to a JSON file in the specified format
    test_results = {
        'Learning Rate': args.lr,
        'Epochs': args.epochs,
        'Input Features': args.input_size,
        'Output Features': args.dim_latent,
        'Test AUC': f'{test_auc:.4f} ± {test_auc_err:.4f}',
        'Test F1 Score': f'{test_f1:.4f} ± {test_f1_err:.4f}',
        'Test Precision': f'{test_precision:.4f} ± {test_precision_err:.4f}',
        'Test Recall': f'{test_recall:.4f} ± {test_recall_err:.4f}',
        'Test Hit': f'{test_hits_k:.4f}',
        'Test mAP': f'{test_map:.4f} ± {test_map_err:.4f}'
    }

    with open(os.path.join(output_path, filename), 'w') as f:
        json.dump(test_results, f)
