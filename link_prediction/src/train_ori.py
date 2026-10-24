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


def train_and_evaluate(args, G_dgl, node_features, node_id_to_name, seed=42, output_path="results/"):
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

 