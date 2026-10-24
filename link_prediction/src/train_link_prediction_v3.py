import copy
import itertools
import os
import re

import dgl
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from .models import GATModel, MLPPredictor, FocalLoss
from . import utils as U
from .utils import (plot_training_validation_metrics, plot_roc_pr_curves, compute_hits_k, compute_precision_at_k,
                    compute_auc_with_symmetrical_confidence, compute_f1_with_symmetrical_confidence,
                    compute_focalloss_with_symmetrical_confidence, compute_accuracy_with_symmetrical_confidence,
                    compute_precision_with_symmetrical_confidence, compute_recall_with_symmetrical_confidence,
                    compute_map_with_symmetrical_confidence, compute_f1, compute_accuracy, compute_precision, compute_recall)

MIRNA_PREFIXES = ("rno-", "mmu-", "hsa-", "EBV-", "hcmv-", "mdv1-", "kshv-")
CANCER_WORDS = ("cancer", "neoplasm", "carcinoma", "tumor", "tumour", "leukemia", "leukaemia", "lymphoma", "melanoma",
                "glioma", "glioblastoma", "sarcoma", "myeloma", "blastoma", "adenoma", "mesothelioma", "malignan",
                "metasta", "hepatoma", "seminoma", "teratoma")


def is_mirna(name):
    return any(p in name for p in MIRNA_PREFIXES)


def is_cancer(name):
    return (not is_mirna(name)) and any(w in name.lower() for w in CANCER_WORDS)


def norm_name(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def sample_neg_pairs(src_nodes, dst_nodes, pos_keys, n, N, rng):
    """n UNIQUE negatives from src_nodes x dst_nodes that are not known positives."""
    keys = np.empty(0, dtype=np.int64)
    while len(keys) < n:
        k = rng.choice(src_nodes, 2 * n).astype(np.int64) * N + rng.choice(dst_nodes, 2 * n)
        keys = np.unique(np.concatenate([keys, k[~np.isin(k, pos_keys)]]))
    keys = rng.permutation(keys)[:n]
    return keys // N, keys % N


def reverse_eids(G, eids):
    u, v = G.find_edges(torch.as_tensor(eids))
    has = G.has_edges_between(v, u)
    return G.edge_ids(v[has], u[has]).numpy() if has.any() else np.array([], dtype=np.int64)


def val_f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-12)
    return float(th[int(np.argmax(f))]) if len(th) else 0.5


def pad_mean(curves):
    L = max(len(c) for c in curves)
    arr = np.full((len(curves), L), np.nan)
    for i, c in enumerate(curves): arr[i, :len(c)] = c
    return np.nanmean(arr, axis=0)


def save_results_latex(df, path, scale=100, bold_best=True,
                       caption=("Five-fold cross-validation on balanced test sets (1:1 positive:negative miRNA--disease pairs). "
                                "Per-fold values are mean $\\pm$ half-width of a 95\\% bootstrap interval over test pairs; "
                                "the last row is the mean and the standard deviation across folds. "
                                "Threshold chosen on validation F1. Best score per column in bold."),
                       label="tab:cv_results"):
    cols = [("Test AUC", "Test AUC Err", r"AUC (\%)"), ("Test F1 Score", "Test F1 Score Err", r"F1 (\%)"),
            ("Test Accuracy", "Test Accuracy Err", r"Acc. (\%)"), ("Test Precision", "Test Precision Err", r"Prec. (\%)"),
            ("Test Recall", "Test Recall Err", r"Recall (\%)"), ("Test mAP", "Test mAP Err", r"mAP (\%)")]
    fmt = lambda v, e: f"{v * scale:.2f}$\\pm${e * scale:.2f}"
    best = {v: df[v].idxmax() for v, _, _ in cols}
    L = [r"% requires \usepackage{booktabs,caption}", r"\begin{table}[t]", r"\centering", r"\captionsetup{font=scriptsize}",
         rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
         r"Fold & " + " & ".join(h for _, _, h in cols) + r" \\", r"\midrule"]
    for i, row in df.iterrows():
        cells = [str(int(row["Fold"]))]
        for v, e, _ in cols:
            s = fmt(row[v], row[e])
            cells.append(s.replace(f"{row[v] * scale:.2f}", rf"\textbf{{{row[v] * scale:.2f}}}", 1) if bold_best and i == best[v] else s)
        L.append(" & ".join(cells) + r" \\")
    L.append(r"\midrule")
    L.append(r"Mean$\pm$SD & " + " & ".join(f"{df[v].mean() * scale:.2f}$\\pm${df[v].std(ddof=1) * scale:.2f}" for v, _, _ in cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("LaTeX table saved to:", path)


def train_and_evaluate(args, G_dgl, node_features, node_id_to_name, seed=None, output_path="results/"):
    seed = getattr(args, "seed", 42) if seed is None else seed
    os.makedirs(output_path, exist_ok=True)
    np.random.seed(seed); torch.manual_seed(seed)

    n_raw = G_dgl.number_of_edges()
    G_dgl = dgl.to_simple(G_dgl)                                    # one edge per miRNA-disease pair
    N, E = G_dgl.number_of_nodes(), G_dgl.number_of_edges()
    print(f"edges: raw={n_raw:,} -> unique pairs={E:,}")
    u, v = G_dgl.edges()
    eids = np.arange(E)
    names = [node_id_to_name[i] for i in range(N)]
    tag = f"drop{args.attn_drop}_head{args.num_heads}_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}"

    src_nodes, dst_nodes = np.unique(u.numpy()), np.unique(v.numpy())
    bipartite = len(np.intersect1d(src_nodes, dst_nodes)) == 0
    pos_keys = u.numpy().astype(np.int64) * N + v.numpy()
    if not bipartite:
        src_nodes = dst_nodes = np.arange(N)
        pos_keys = np.unique(np.r_[pos_keys, v.numpy().astype(np.int64) * N + u.numpy(), np.arange(N) * (N + 1)])
    max_neg = len(src_nodes) * len(dst_nodes) - len(pos_keys)
    print(f"bipartite={bipartite} | sources={len(src_nodes):,} destinations={len(dst_nodes):,} | possible negatives={max_neg:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patience = getattr(args, "patience", 30)
    val_frac = getattr(args, "val_frac", 0.1)

    def mp_graph(g):                                                # miRNAs have in-degree 0 -> message passing in both directions
        if bipartite and getattr(args, "bipartite_mp", True):
            g = dgl.add_reverse_edges(g)
        return dgl.add_self_loop(dgl.remove_self_loop(g))

    mk = lambda uu, vv: dgl.graph((torch.as_tensor(uu), torch.as_tensor(vv)), num_nodes=N).to(device)
    all_rows, fold_results, fold_states = [], [], []
    tr_acc, va_acc, tr_loss, va_loss = [], [], [], []
    metrics_txt = os.path.join(output_path, f"test_metrics_5_fold_{tag}.txt")
    open(metrics_txt, "w").close()

    for fold, (trainval_idx, test_idx) in enumerate(KFold(5, shuffle=True, random_state=seed).split(eids)):
        print(f"\nFold {fold + 1}")
        rng = np.random.RandomState(seed + fold)
        trainval_idx = rng.permutation(trainval_idx)
        n_val = int(len(trainval_idx) * val_frac)                   # validation = a SMALL held-out part (was 80% of the training edges)
        val_idx, train_idx = trainval_idx[:n_val], trainval_idx[n_val:]

        neg_u, neg_v = sample_neg_pairs(src_nodes, dst_nodes, pos_keys, min(E, max_neg), N, rng)
        a, b = len(train_idx), len(train_idx) + len(val_idx)
        tn_, vn_, te_n = (neg_u[:a], neg_v[:a]), (neg_u[a:b], neg_v[a:b]), (neg_u[b:b + len(test_idx)], neg_v[b:b + len(test_idx)])

        hold_val = np.unique(np.concatenate([val_idx, reverse_eids(G_dgl, val_idx)]))
        hold_test = np.unique(np.concatenate([test_idx, reverse_eids(G_dgl, test_idx)]))
        train_g = mp_graph(dgl.remove_edges(G_dgl, np.unique(np.concatenate([hold_val, hold_test])).tolist())).to(device)  # train edges only
        test_g = mp_graph(dgl.remove_edges(G_dgl, hold_test.tolist())).to(device)                                         # train+val edges
        tp_g, tn_g = mk(u[train_idx], v[train_idx]), mk(*tn_)
        vp_g, vn_g = mk(u[val_idx], v[val_idx]), mk(*vn_)
        ep_g, en_g = mk(u[test_idx], v[test_idx]), mk(*te_n)

        torch.manual_seed(seed + fold)
        model = GATModel(node_features.shape[1], out_feats=args.out_feats, num_layers=args.num_layers, num_heads=args.num_heads,
                         feat_drop=args.feat_drop, attn_drop=args.attn_drop, do_train=True).to(device)
        pred = MLPPredictor(args.input_size, args.hidden_size).to(device)
        criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
        opt = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.lr)
        sched = StepLR(opt, step_size=1000, gamma=0.1)
        feat_tr, feat_te = train_g.ndata["feat"], test_g.ndata["feat"]

        f_tra, f_vaa, f_trl, f_val = [], [], [], []
        best = dict(auc=-1.0, ep=-1, state=None); bad = 0
        for e in tqdm(range(args.epochs), desc=f"Training Fold {fold + 1}"):
            model.train(); pred.train()
            h = model(train_g, feat_tr)
            ps, ns = pred(tp_g, h), pred(tn_g, h)
            loss = criterion(torch.cat([ps, ns]), torch.cat([torch.ones_like(ps), torch.zeros_like(ns)]))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()

            model.eval(); pred.eval()
            with torch.no_grad():
                h_val = model(train_g, feat_tr)                     # validation sees TRAIN edges only (was the full graph)
                vp, vn = pred(vp_g, h_val), pred(vn_g, h_val)
                v_loss = criterion(torch.cat([vp, vn]), torch.cat([torch.ones_like(vp), torch.zeros_like(vn)]))
                vpp, vnp = torch.sigmoid(vp), torch.sigmoid(vn)     # probabilities everywhere: accuracy thresholds p, not logits
                v_acc = ((vpp >= 0.5).sum().item() + (vnp < 0.5).sum().item()) / (len(vpp) + len(vnp))
                t_acc = ((torch.sigmoid(ps) >= 0.5).sum().item() + (torch.sigmoid(ns) < 0.5).sum().item()) / (len(ps) + len(ns))
                v_auc = roc_auc_score(np.r_[np.ones(len(vpp)), np.zeros(len(vnp))], torch.cat([vpp, vnp]).cpu().numpy())
            f_trl.append(loss.item()); f_val.append(v_loss.item()); f_tra.append(t_acc); f_vaa.append(v_acc)
            if v_auc > best["auc"]:
                best = dict(auc=v_auc, ep=e, state=(copy.deepcopy(model.state_dict()), copy.deepcopy(pred.state_dict()))); bad = 0
            else:
                bad += 1
            if e % 5 == 0:
                print(f"Fold {fold + 1} | Epoch {e} | Loss {loss.item():.4f} | Train acc {t_acc:.4f} | Val acc {v_acc:.4f} | Val AUC {v_auc:.4f}")
            if bad >= patience:
                print(f"early stop at epoch {e} (best epoch {best['ep']}, val AUC {best['auc']:.4f})"); break
        tr_acc.append(f_tra); va_acc.append(f_vaa); tr_loss.append(f_trl); va_loss.append(f_val)

        model.load_state_dict(best["state"][0]); pred.load_state_dict(best["state"][1]); model.eval(); pred.eval()
        with torch.no_grad():
            h = model(train_g, feat_tr)
            vpp, vnp = torch.sigmoid(pred(vp_g, h)).cpu().numpy(), torch.sigmoid(pred(vn_g, h)).cpu().numpy()
            thr = val_f1_threshold(np.r_[np.ones(len(vpp)), np.zeros(len(vnp))], np.r_[vpp, vnp])   # threshold from VALIDATION only
            h_t = model(test_g, feat_te)                            # test edges NOT in the graph
            test_pos, test_neg = torch.sigmoid(pred(ep_g, h_t)).cpu(), torch.sigmoid(pred(en_g, h_t)).cpu()

        U.THRESHOLD = thr
        auc_, auc_e = compute_auc_with_symmetrical_confidence(test_pos, test_neg)
        f1_, f1_e = compute_f1_with_symmetrical_confidence(test_pos, test_neg)
        fl_, fl_e = compute_focalloss_with_symmetrical_confidence(test_pos, test_neg)
        pr_, pr_e = compute_precision_with_symmetrical_confidence(test_pos, test_neg)
        rc_, rc_e = compute_recall_with_symmetrical_confidence(test_pos, test_neg)
        mp_, mp_e = compute_map_with_symmetrical_confidence(test_pos, test_neg)
        ac_, ac_e = compute_accuracy_with_symmetrical_confidence(test_pos, test_neg)
        hits10, hits50 = compute_hits_k(test_pos, test_neg, 10), compute_hits_k(test_pos, test_neg, 50)
        p_at_100 = compute_precision_at_k(test_pos, test_neg, 100)
        U.THRESHOLD = 0.5                                           # same model at the fixed 0.5 cut-off (point estimates)
        fixed = dict(f1=compute_f1(test_pos, test_neg), acc=compute_accuracy(test_pos, test_neg),
                     prec=compute_precision(test_pos, test_neg), rec=compute_recall(test_pos, test_neg))

        line = (f"Fold {fold + 1} (best epoch {best['ep']}, val-thr {thr:.3f}): AUC {auc_:.4f}+/-{auc_e:.4f} | F1 {f1_:.4f}+/-{f1_e:.4f} | "
                f"Acc {ac_:.4f}+/-{ac_e:.4f} | Prec {pr_:.4f}+/-{pr_e:.4f} | Rec {rc_:.4f}+/-{rc_e:.4f} | mAP {mp_:.4f}+/-{mp_e:.4f} | "
                f"Hits@10 {hits10:.4f} Hits@50 {hits50:.4f} P@100 {p_at_100:.4f} | @0.5: F1 {fixed['f1']:.4f} Prec {fixed['prec']:.4f} Rec {fixed['rec']:.4f}")
        print(line)
        with open(metrics_txt, "a") as f: f.write(line + "\n")

        fold_results.append((np.r_[np.ones(len(test_pos)), np.zeros(len(test_neg))], np.r_[test_pos.numpy(), test_neg.numpy()]))
        all_rows.append({"Fold": fold + 1, "best_epoch": best["ep"], "val_auc": best["auc"], "threshold": thr,
                         "Test AUC": auc_, "Test AUC Err": auc_e, "Test F1 Score": f1_, "Test F1 Score Err": f1_e,
                         "Test Precision": pr_, "Test Precision Err": pr_e, "Test Recall": rc_, "Test Recall Err": rc_e,
                         "Test Hit10": hits10, "Test Hit50": hits50, "Test P@100": p_at_100, "Test mAP": mp_, "Test mAP Err": mp_e,
                         "Test FocalLoss": fl_, "Test FocalLoss Err": fl_e, "Test Accuracy": ac_, "Test Accuracy Err": ac_e,
                         "F1@0.5": fixed["f1"], "Acc@0.5": fixed["acc"], "Prec@0.5": fixed["prec"], "Rec@0.5": fixed["rec"]})
        fold_states.append((copy.deepcopy(model.state_dict()), copy.deepcopy(pred.state_dict())))

    # ---------------- results, tables, plots (once, after all folds) ----------------
    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(output_path, "all_fold_results.csv"), index=False)
    print("\n5-fold mean +/- SD across folds"); print(res.drop(columns="Fold").agg(["mean", "std"]).T.round(4).to_string())
    save_results_latex(res, os.path.join(output_path, f"table_5_fold_{tag}.tex"))
    plot_roc_pr_curves(fold_results, os.path.join(output_path, f"roc_pr_curve_5_fold_{tag}.png"))
    plot_training_validation_metrics(tr_acc, pad_mean(tr_acc), va_acc, pad_mean(va_acc), tr_loss, pad_mean(tr_loss),
                                     va_loss, pad_mean(va_loss), output_path, args)

    # ---------------- novel predictions: score CANDIDATE pairs, ensemble of the 5 fold models ----------------
    if bipartite:
        mir_ids = src_nodes
        dis_ids = np.array([i for i in dst_nodes if (not getattr(args, "cancer_only", False)) or is_cancer(names[i])])
    else:
        mir_ids = np.array([i for i, n in enumerate(names) if is_mirna(n)]); dis_ids = np.array([i for i, n in enumerate(names) if not is_mirna(n)])
    cu, cv = np.repeat(mir_ids, len(dis_ids)), np.tile(dis_ids, len(mir_ids))
    known = G_dgl.has_edges_between(torch.as_tensor(cu), torch.as_tensor(cv)).numpy()
    cu, cv = cu[~known], cv[~known]
    print(f"\ncandidate pairs: {len(cu):,} (known edges excluded: {int(known.sum()):,})")
    G_full = mp_graph(G_dgl).to(device); feat_full = G_full.ndata["feat"]
    score_sum = np.zeros(len(cu))
    for ms, ps_ in fold_states:
        model.load_state_dict(ms); pred.load_state_dict(ps_); model.eval(); pred.eval()
        with torch.no_grad():
            h = model(G_full, feat_full)
            for s0 in range(0, len(cu), 500_000):
                sl = slice(s0, s0 + 500_000)
                score_sum[sl] += torch.sigmoid(pred(mk(cu[sl], cv[sl]), h)).cpu().numpy()
    novel = pd.DataFrame({"source": [names[i] for i in cu], "destination": [names[i] for i in cv], "score": score_sum / len(fold_states)})
    novel = novel.sort_values("score", ascending=False).reset_index(drop=True)
    novel["rank"] = np.arange(1, len(novel) + 1); novel["percentile"] = novel["rank"] / len(novel)
    novel.to_csv(os.path.join(output_path, f"top_scores_novel_{tag}.csv"), index=False)

    # evidence: PAIR-level match (miRNA AND disease) on normalised names (old code matched any pair sharing the miRNA)
    df2 = pd.read_csv("data/_combined_hmdd_miR2Disease_miRNA_disease.csv")
    df2["k_m"], df2["k_d"] = df2["miRNA"].map(norm_name), df2["disease"].map(norm_name)
    ref = df2.groupby(["k_m", "k_d"])["reference"].agg(lambda s: ", ".join(sorted(set(map(str, s))))).reset_index()
    novel["k_m"], novel["k_d"] = novel["source"].map(norm_name), novel["destination"].map(norm_name)
    novel = novel.merge(ref, on=["k_m", "k_d"], how="left")
    novel["evidence"] = novel["reference"].fillna("Unconfirmed")
    conf = novel[novel["evidence"] != "Unconfirmed"]
    print(f"novel candidates with pair-level literature evidence: {len(conf):,} of {len(novel):,}; "
          f"median percentile of confirmed pairs: {conf['percentile'].median() if len(conf) else float('nan'):.4f}")
    novel.drop(columns=["k_m", "k_d", "reference"]).to_csv(os.path.join(output_path, f"top_predictions_with_evidence_{tag}.csv"), index=False)
    return res
