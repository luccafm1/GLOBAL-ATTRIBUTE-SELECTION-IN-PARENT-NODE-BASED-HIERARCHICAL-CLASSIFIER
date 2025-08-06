#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hic.py  –  Hierarchical benchmark extended with five literature-backed levers

Features
--------
* Local-Classifier-per-Parent-Node (LCPN) with ExtraTrees (baseline)
* Global flat multi-label MLP with ancestor propagation (baseline)
* HMC-LMLP-Predicted (Cerri et al. 2016)  ← lever L1
* Cost-Sensitive Hierarchical Bayes reconciliation (Valentini 2014) ← L2
* Weighted True-Path-Rule smoothing (Valentini 2014)               ← L3
* Heterogeneous probability blending (mean / geometric)            ← L4
* Depth-wise threshold optimisation                                ← L5

Usage
-----
    python hic.py --data-dir "/path/to/arff"            # full benchmark
    python hic.py --data-dir "/path/to/arff" --demo     # church_GO only
"""

from __future__ import annotations
import time, warnings
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from hic_arff import *

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                   MultiLabelBinarizer)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score

from hiclass import MultiLabelLocalClassifierPerParentNode as LCPN
from hiclass.HierarchicalClassifier import make_leveled
from hiclass.metrics import f1 as hf1, precision as hprec, recall as hrec

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------- HYPERPARAMS ------------------------------------
MINC       = 2           # mínimo de instâncias por folha (após promoção)
PCT_FS     = 0.60        # % de features mantidas (χ² / Gini)
K_MAX      = 300         # limite superior de K no χ² global
MIN_LEAF   = 5           # min_samples_leaf ExtraTrees
SEED       = 4822
MLP_HIDDEN = (128, 64)   # arquitetura padrão dos MLPs
# -----------------------------------------------------------------------------

# =============================== Utilities ====================================
# def _pad_to(arr, max_p, max_l):
#     if arr.shape[1] == max_p and arr.shape[2] == max_l:
#         return arr
#     out = np.full((arr.shape[0], max_p, max_l), '', dtype=object)
#     out[:, :arr.shape[1], :arr.shape[2]] = arr
#     return out

def leaves(Y):
    return {p[-1] for s in Y for p in s}


def auto_k(n_feats: int, pct: float = PCT_FS, k_max: int = K_MAX) -> int:
    return max(5, min(int(n_feats * pct), k_max))

def duplicate_weights(X_df: pd.DataFrame) -> np.ndarray:
    if X_df.empty:                           # sem colunas? peso uniforme
        return np.ones(len(X_df), dtype=float)
    h = pd.util.hash_pandas_object(X_df, index=False)
    freq = h.value_counts()
    return h.map(lambda v: 1.0 / freq[v]).to_numpy()

def flat_micro_aupr(y_true_paths, y_pred_paths):
    true_leaves = [ {p[-1] for p in s} for s in y_true_paths ]
    pred_leaves = [ {p[-1] for p in s} for s in y_pred_paths ]

    mlb = MultiLabelBinarizer()
    Y_true = mlb.fit_transform(true_leaves)
    Y_pred = mlb.transform(pred_leaves)  # binary 0/1 acts as score proxy
    if Y_true.shape[1] == 0:
        return 0.0
    return float(average_precision_score(Y_true, Y_pred, average="micro"))

def hf_samples(y_true_paths, y_pred_paths, root="root"):
    def to_set(sample):
        return {node for path in sample for node in path}

    hfs = []
    for true_sample, pred_sample in zip(y_true_paths, y_pred_paths):
        T = to_set(true_sample)
        P = to_set(pred_sample)

        if root in T: T.remove(root)
        if root in P: P.remove(root)

        inter = len(T & P)
        if not P or not T:
            hfs.append(0.0)      
            continue
        prec = inter / len(P)
        rec  = inter / len(T)
        hfs.append( (2*prec*rec) / (prec+rec) if (prec+rec) else 0.0 )

    return float(np.mean(hfs))

# ------------------- Feature-selection ----------------------------
def fs_parent_chi2(X_df, Y, c2p):
    X_imp = SimpleImputer(strategy="median").fit_transform(X_df)
    X_norm = MinMaxScaler().fit_transform(X_imp)


    m, n = X_norm.shape

    sel = np.zeros(n, dtype=bool)
    k = auto_k(n)

    parent2rows = defaultdict(list)
    for i, sample in enumerate(Y):
        for path in sample:
            for pa, _ in zip(path[:-1], path[1:]):
                parent2rows[pa].append(i)

    for rows in parent2rows.values():
        rows = np.array(rows)
        y_bin = np.zeros(m); y_bin[rows] = 1
        scores, _ = chi2(X_norm, y_bin)
        sel[np.argsort(scores)[-k:]] = True
    return sel


def fs_parent_tree(X_df, Y, c2p):
    X_imp = SimpleImputer(strategy="median").fit_transform(X_df)

    m, n = X_imp.shape
    sel = np.zeros(n, dtype=bool)
    k = auto_k(n)

    parent2rows = defaultdict(set)
    for i, sample in enumerate(Y):
        for path in sample:
            for pa, _ in zip(path[:-1], path[1:]):
                parent2rows[pa].add(i)

    for rows in parent2rows.values():
        rows = list(rows)
        y_bin = np.zeros(m); y_bin[rows] = 1
        et = ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)
        et.fit(X_imp[rows], y_bin[rows])
        sel[np.argsort(et.feature_importances_)[-k:]] = True
    return sel


# ---- helper: determine k ----
def _auto_k(n_features, pct_fs=None, k_max=None):
    # Uses your globals if present; default pct=0.6 and cap by k_max if given
    pct = pct_fs if pct_fs is not None else (globals().get("PCT_FS", 0.60))
    kmx = k_max if k_max is not None else (globals().get("K_MAX", None))
    k = max(1, int(round(pct * n_features)))
    if kmx is not None:
        k = min(k, int(kmx))
    return k

# ---- helper: binarize leaves per sample ----
def _binarize_leaves(Y):
    """
    Y: iterable of samples; each sample is list of paths; each path is list of nodes
    Returns: (Y_bin, mlb, leaf_list)
    """
    leaf_sets = []
    for s in Y:
        leaves = {p[-1] for p in s}
        leaf_sets.append(leaves)
    mlb = MultiLabelBinarizer()
    Y_bin = mlb.fit_transform(leaf_sets)  # shape (n_samples, n_labels)
    return Y_bin, mlb, list(mlb.classes_)

# ---- helper: safe numeric matrix ----
def _prep_X_numeric(X_df, scale_nonneg=False):
    """
    Impute numerics and optionally scale to [0,1] (needed for chi2).
    Returns np.ndarray.
    """
    X = X_df.values if isinstance(X_df, pd.DataFrame) else np.asarray(X_df)
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    if scale_nonneg:
        scaler = MinMaxScaler()
        X_imp = scaler.fit_transform(X_imp)
    return X_imp

# =========================
# GLOBAL FS: ExtraTrees (embedded, supervised)
# =========================
def fs_global_tree(X_df, Y, n_estimators=256, random_state=0, agg="max", pct_fs=None, k_max=None):
    """
    Supervised global FS using One-vs-Rest ExtraTrees feature importances aggregated across labels.
    agg: "max" or "mean" to pool importances across labels.
    Returns boolean mask of length n_features.
    """
    X_imp = _prep_X_numeric(X_df, scale_nonneg=False)
    n = X_imp.shape[1]
    if n == 0:
        return np.zeros(0, dtype=bool)

    Y_bin, mlb, labels = _binarize_leaves(Y)
    # Optionally limit to top-L frequent labels to reduce cost (comment out if not needed)
    freq = np.asarray(Y_bin.sum(axis=0)).ravel()
    order = np.argsort(-freq)
    # You can cap to top L labels if extremely high-dimensional label space:
    # L_cap = min(len(order), 512)
    # order = order[:L_cap]

    agg_scores = np.zeros(n, dtype=float)
    counts = 0

    # Fit OvR ExtraTrees for each label
    for j in order:
        yj = Y_bin[:, j]
        if yj.sum() < 2:  # skip degenerate labels
            continue
        clf = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            random_state=(random_state + int(j)),
            n_jobs=-1,
            bootstrap=False
        )
        clf.fit(X_imp, yj)
        fi = getattr(clf, "feature_importances_", None)
        if fi is None:
            continue
        if agg == "max":
            agg_scores = np.maximum(agg_scores, fi)
        else:
            agg_scores += fi
        counts += 1

    if counts == 0:
        # Fallback: uniform
        agg_scores = np.zeros(n)
    elif agg != "max":
        agg_scores /= max(1, counts)

    k = _auto_k(n, pct_fs=pct_fs, k_max=k_max)
    idx = np.argsort(agg_scores)[-k:]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask

# =========================
# GLOBAL FS: Mutual Information (filter, supervised)
# =========================
def fs_global_mi(X_df, Y, discrete_features="auto", n_neighbors=3, random_state=0, agg="max", pct_fs=None, k_max=None):
    """
    Supervised global FS using mutual information aggregated across labels (OvR).
    Aggregation: "max" (default) or "mean".
    """
    X_imp = _prep_X_numeric(X_df, scale_nonneg=False)
    n = X_imp.shape[1]
    if n == 0:
        return np.zeros(0, dtype=bool)

    Y_bin, mlb, labels = _binarize_leaves(Y)

    agg_scores = np.zeros(n, dtype=float)
    counts = 0
    for j in range(Y_bin.shape[1]):
        yj = Y_bin[:, j]
        if yj.sum() < 2 or yj.sum() == len(yj):  # skip degenerate
            continue
        mi = mutual_info_classif(
            X_imp, yj,
            discrete_features=discrete_features,
            n_neighbors=n_neighbors,
            random_state=random_state
        )  # shape (n_features,)
        if agg == "max":
            agg_scores = np.maximum(agg_scores, mi)
        else:
            agg_scores += mi
        counts += 1

    if counts == 0:
        agg_scores = np.zeros(n)
    elif agg != "max":
        agg_scores /= max(1, counts)

    k = _auto_k(n, pct_fs=pct_fs, k_max=k_max)
    idx = np.argsort(agg_scores)[-k:]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask

# =========================
# GLOBAL FS: Chi-square (filter, supervised)
# =========================
def fs_global_chi2(X_df, Y, agg="max", pct_fs=None, k_max=None):
    """
    Supervised global FS using chi-square aggregated across labels (OvR).
    NOTE: Requires non-negative features → scale to [0,1].
    """
    X_nonneg = _prep_X_numeric(X_df, scale_nonneg=True)
    n = X_nonneg.shape[1]
    if n == 0:
        return np.zeros(0, dtype=bool)

    Y_bin, mlb, labels = _binarize_leaves(Y)

    agg_scores = np.zeros(n, dtype=float)
    counts = 0
    for j in range(Y_bin.shape[1]):
        yj = Y_bin[:, j]
        if yj.sum() < 2 or yj.sum() == len(yj):
            continue
        s, _ = chi2(X_nonneg, yj)  # shape (n_features,)
        if agg == "max":
            agg_scores = np.maximum(agg_scores, s)
        else:
            agg_scores += s
        counts += 1

    if counts == 0:
        agg_scores = np.zeros(n)
    elif agg != "max":
        agg_scores /= max(1, counts)

    k = _auto_k(n, pct_fs=pct_fs, k_max=k_max)
    idx = np.argsort(agg_scores)[-k:]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


# ----------------------- Rare-leaf promotion ----------------------------------
# def promote_iter(ytr, yva, minc=MINC):
#     total_leaves_before = len({p[-1] for s in np.concatenate([ytr, yva])
#                                          for p in s})
#     inst_before = len(ytr)
#     promoted_total = 0
#     removed_labels = set()

#     while True:
#         cnt = Counter([p[-1] for s in np.concatenate([ytr, yva]) for p in s])
#         rare = {l for l, c in cnt.items() if c < minc}
#         if not rare:
#             break
#         promoted_total += sum(cnt[l] for l in rare)
#         removed_labels |= rare

#         def promote(sample):
#             out = []
#             for path in sample:
#                 while path and path[-1] in rare:
#                     path = path[:-1]
#                 if path:
#                     out.append(path)
#             return out

#         ytr = np.array([promote(s) for s in ytr], dtype=object)
#         yva = np.array([promote(s) for s in yva], dtype=object)

#     total_leaves_after = len({p[-1] for s in np.concatenate([ytr, yva])
#                                         for p in s})

#     stats = {
#         "leaves_before": total_leaves_before,
#         "leaves_after":  total_leaves_after,
#         "pct_leaves_removed": (
#             100 * (total_leaves_before - total_leaves_after)
#             / total_leaves_before
#         ),
#         "instances_promoted": promoted_total,
#         "pct_inst_promoted": 100 * promoted_total / inst_before,
#     }
#     print(f"   → removed {stats['pct_leaves_removed']:.1f}% of leaf labels "
#             f"({total_leaves_before - total_leaves_after} / "
#             f"{total_leaves_before})")
#     print(f"   → promoted {stats['pct_inst_promoted']:.1f}% of training "
#             f"instances ({promoted_total} / {inst_before})")
#     return ytr, yva
def promote_iter(ytr, yva, minc=MINC):
    def _leaf_counter(samples):
        return Counter([p[-1] for s in samples for p in s])

    total_leaves_before = len({p[-1] for s in ytr for p in s})
    inst_before = len(ytr)
    promoted_total = 0

    while True:
        cnt_tr = _leaf_counter(ytr)
        rare = {l for l, c in cnt_tr.items() if c < minc}
        if not rare:
            break

        def _promote_sample(sample_paths):
            out = []
            for path in sample_paths:
                p = list(path)
                while len(p) > 1 and p[-1] in rare:
                    p.pop()
                if p:
                    out.append(p)
            return out

        promoted_total += sum(cnt_tr[l] for l in rare if l in cnt_tr)
        ytr = np.array([_promote_sample(s) for s in ytr], dtype=object)
        yva = np.array([_promote_sample(s) for s in yva], dtype=object)

    total_leaves_after = len({p[-1] for s in ytr for p in s})
    pct_removed = 0.0 if total_leaves_before == 0 else \
                  100.0 * (total_leaves_before - total_leaves_after) / max(1, total_leaves_before)
    pct_promoted = 0.0 if inst_before == 0 else 100.0 * (promoted_total / inst_before)
    print(f"   → removed {pct_removed:.1f}% of TRAIN leaf labels "
          f"({total_leaves_before - total_leaves_after} / {total_leaves_before})")
    print(f"   → promoted {pct_promoted:.1f}% of training instances "
          f"({promoted_total} / {inst_before})")
    return ytr, yva

# --------------------------- Helpers ------------------------------------------
def expand_leaves_to_paths(pred_leaves, c2p, root="root"):
    def to_path(leaf):
        path = [leaf]
        while path[0] in c2p:
            path.insert(0, c2p[path[0]])
        if path[0] != root:
            path.insert(0, root)
        return path
    return np.array([[to_path(l) for l in sample] for sample in pred_leaves],
                    dtype=object)


def build_preproc(n_features):
    return ColumnTransformer([
        ('num', Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('std', StandardScaler(with_mean=False))
        ]), list(range(n_features)))
    ])


def timeit(fn, *args, **kwargs):
    start = time.perf_counter()
    res = fn(*args, **kwargs)
    return res, time.perf_counter() - start


def pad_pair(y_true, y_pred):
    y_pred = [s if len(s) else [["root"]] for s in y_pred]

    yt = [make_leveled(s) for s in y_true]
    yp = [make_leveled(s) for s in y_pred]
    mp = max(max(a.shape[0] for a in yt), max(a.shape[0] for a in yp))
    ml = max(max(a.shape[1] for a in yt), max(a.shape[1] for a in yp))

    def pad(b):
        out = np.full((len(b), mp, ml), '', dtype=object)
        for i, a in enumerate(b):
            out[i, :a.shape[0], :a.shape[1]] = a
        return out
    return pad(yt), pad(yp)

# ============================= Baselines ======================================
def train_lcpn(Xtr_p, ytr, Xva_p, yva, w_train=None):
    base = ExtraTreesClassifier(
        n_estimators=500,         
        max_depth=20,             
        min_samples_split=10,     
        min_samples_leaf=MIN_LEAF,
        max_features='sqrt',      
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=SEED
    )
    lcpn = LCPN(local_classifier=deepcopy(base),
                n_jobs=-1, tolerance=0)

    fit_args = (Xtr_p, ytr)
    fit_kwargs = {"sample_weight": w_train} if w_train is not None else {}
    _, t_fit = timeit(lcpn.fit, *fit_args, **fit_kwargs)

    pr, t_pr = timeit(lcpn.predict, Xva_p)
    yt, yp = pad_pair(yva, pr)
    return yt, yp, t_fit, t_pr

def train_lcpn_mlp(Xtr_p, ytr, Xva_p, yva):
    base = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-7,
        batch_size="auto",
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=1000,
        shuffle=True,
        random_state=SEED,
        early_stopping=False,
        n_iter_no_change=10,
    )
    lcpn = LCPN(local_classifier=deepcopy(base), n_jobs=-1, tolerance=0.05)
    _, t_fit = timeit(lcpn.fit, Xtr_p, ytr)       
    pr, t_pr = timeit(lcpn.predict, Xva_p)
    yt, yp = pad_pair(yva, pr)
    return yt, yp, t_fit, t_pr


def train_hmc_global(Xtr_p, ytr, Xva_p, yva, c2p):
    mlb = MultiLabelBinarizer()
    Ytr_bin = mlb.fit_transform([leaves([s]) for s in ytr])

    tree = DecisionTreeClassifier(min_samples_leaf=MIN_LEAF, random_state=SEED)
    _, t_fit = timeit(tree.fit, Xtr_p, Ytr_bin)
    Y_pred_bin, t_pr = timeit(tree.predict, Xva_p)

    pred_leaves = []
    for row in Y_pred_bin:
        lbls = [mlb.classes_[i] for i in np.flatnonzero(row)]
        pred_leaves.append(lbls)
    pred_paths = expand_leaves_to_paths(pred_leaves, c2p)

    yt, yp = pad_pair(yva, pred_paths)
    return yt, yp, t_fit, t_pr

# ================================= MAIN =======================================

def merge_arffs(files, out_path):
    if not files:
        raise ValueError("lista vazia")

    union_attrs, per_file = set(), []
    for f in files:
        txt = f.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"(?i)^[ \t]*@data\b", txt, re.M)
        if not m:
            raise RuntimeError(f"@data ausente em {f}")

        header, data = txt[:m.start()], txt[m.end():]
        attrs = [
            ln.split()[1]          # segundo token é o nome
            for ln in header.splitlines()
            if ln.lower().startswith("@attribute")
        ]
        union_attrs.update(attrs)
        per_file.append((attrs, data))

    # ordem canônica = ordem em que apareceu na primeira base
    ordered = list(per_file[0][0]) + [a for a in union_attrs if a not in per_file[0][0]]

    dfs = []
    for attrs, data in per_file:
        df = pd.read_csv(io.StringIO(data), header=None,
                         comment="%", na_values="?")
        # mapa coluna‑origem → posição no union
        colmap = {a: ordered.index(a) for a in attrs}
        new = pd.DataFrame(np.nan, index=df.index, columns=range(len(ordered)))
        for j_src, a in enumerate(attrs):
            new.iloc[:, colmap[a]] = df.iloc[:, j_src]
        dfs.append(new)

    merged = pd.concat(dfs, ignore_index=True)

    # reconstruir cabeçalho
    first_header = files[0].read_text(encoding="utf-8", errors="ignore")
    hdr_lines = [
        ln for ln in first_header.splitlines()
        if not ln.lower().startswith("@attribute")
    ]
    for a in ordered:
        hdr_lines.append(f"@attribute {a} numeric")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fw:
        fw.write("\n".join(hdr_lines).rstrip() + "\n@data\n")
        merged.to_csv(fw, header=False, index=False)



# ================================= IO helpers ===========================
OUTPUT_COLS = [
    "dataset", "fold", "scenario", "model",
    "hF", "hF1(mac)", "hF1(mic)", "Prec", "Rec", "t_fit", "t_pred"
]
def _append_rows(out_path: Path, rows: list[dict]):
    """Append rows to CSV, creating header if file does not exist."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exists = out_path.exists()
    df = pd.DataFrame(rows)[OUTPUT_COLS]
    # mode='a' and header=not exists ensure one header only
    df.to_csv(out_path, mode='a', header=(not exists), index=False)

def _load_done_keys(out_path: Path) -> set[tuple[str,int,str]]:
    """Read existing (dataset, fold, scenario) keys to avoid duplicates on resume."""
    if not out_path.exists():
        return set()
    try:
        df = pd.read_csv(out_path, usecols=["dataset","fold","scenario"])
        return set((r["dataset"], int(r["fold"]), r["scenario"]) for _, r in df.iterrows())
    except Exception:
        # If the file is partially written or being synced, be permissive.
        return set()

# ================================= MAIN =======================================
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
import multiprocessing as mp
import numpy as np

def load_full_arff(ds_name: str, c2p, data_dir: Path):
    X_tr, y_tr = load_split(ds_name, "train", c2p, data_dir)
    X_va, y_va = load_split(ds_name, "valid", c2p, data_dir)
    X_full = pd.concat([X_tr, X_va], ignore_index=True)
    Y_full = np.concatenate([y_tr, y_va])
    return X_full.reset_index(drop=True), Y_full

def primary_leaf(sample):
    return sample[0][-1] if sample and sample[0] else "root"

# >>> run_one_fold now writes incrementally and returns counters instead of rows
def run_one_fold(ds, fold_id, train_idx, valid_idx,
                 X_all, Y_all, c2p, scenarios, MINC, out_path: Path):
    appended = 0
    skipped  = 0
    failed   = 0
    try:
        # Load de-dupe keys ONCE per fold process (resume-safe across restarts)
        done = _load_done_keys(out_path)

        Xtr = X_all.iloc[train_idx].reset_index(drop=True)
        ytr = Y_all[train_idx]
        Xva = X_all.iloc[valid_idx].reset_index(drop=True)
        yva = Y_all[valid_idx]

        ytr, yva = promote_iter(ytr, yva, minc=MINC)

        for scen, fs_fun in scenarios.items():
            key = (ds, int(fold_id), scen)
            if key in done:
                skipped += 1
                continue

            try:
                mask = fs_fun(Xtr, ytr)
                Xtr_fs, Xva_fs = Xtr.loc[:, mask], Xva.loc[:, mask]

                prep = build_preproc(Xtr_fs.shape[1])
                Xtr_p = prep.fit_transform(Xtr_fs)
                Xva_p = prep.transform(Xva_fs)

                if scen.startswith("LCPN") and "MLP" in scen:
                    yt, yp, tf, tp = train_lcpn_mlp(Xtr_p, ytr, Xva_p, yva)
                    model_name = "LCPN-MLP"
                elif scen.startswith("LCPN"):
                    yt, yp, tf, tp = train_lcpn(Xtr_p, ytr, Xva_p, yva, duplicate_weights(Xtr_fs))
                    model_name = "LCPN"
                else:
                    yt, yp, tf, tp = train_hmc_global(Xtr_p, ytr, Xva_p, yva, c2p)
                    model_name = "GLOBAL"

                # Compute metrics
                row = {
                    "dataset": ds,
                    "fold":    int(fold_id),
                    "scenario": scen,
                    "model":   model_name,
                    "hF1(mac)":round(hf1(yt, yp, "macro"), 3),
                    "hF1(mic)":round(hf1(yt, yp, "micro"), 3),
                    "hF":      round(hf_samples(yt, yp), 3),
                    "Prec":    round(hprec(yt, yp, "micro"), 3),
                    "Rec":     round(hrec(yt, yp, "micro"), 3),
                    "t_fit":   round(tf, 2),
                    "t_pred":  round(tp, 2),
                    # "AU(PRC)": round(flat_micro_aupr(yt, yp), 2),  # not written
                }

                # Append immediately (crash-resilient)
                _append_rows(out_path, [row])
                appended += 1
                done.add(key)

            except Exception as e:
                failed += 1
                # Keep going with next scenario, but make it visible in the log
                print(f"[WARN] {ds} fold {fold_id} scenario {scen} failed: {e}")

    except Exception as e:
        # If the whole fold hits a fatal error, abandon this fold only.
        print(f"[ERROR] Abandoning {ds} fold {fold_id}: {e}")

    return appended, skipped, failed

def main():
    RAW = Path("datasets/raw")
    N_FOLDS = 5
    N_CORES = min(N_FOLDS, mp.cpu_count())

    datasets = [f.stem.rsplit(".", 1)[0]
                for f in RAW.glob("*_GO.train.arff")]
    
    for ds in datasets:
        print(f"\n=== {ds} (5-fold CV) ===")
        c2p = child_parent_map(RAW / f"{ds}.train.arff")

        scenarios = {
            "LCPN-BASE":        lambda X, _: np.ones(X.shape[1], bool),
            "LCPN-MLP":         lambda X, _: np.ones(X.shape[1], bool),

            "GLOB-CHI2":        lambda X, Y: fs_global_chi2(X, Y),
            "LCPN-CHI2":        lambda X, Y: fs_parent_chi2(X, Y, c2p),
            "LCPN-TREEFS":      lambda X, Y: fs_parent_tree(X, Y, c2p),
            "LCPN-GCHI2":       lambda X, Y: fs_global_chi2(X, Y),
            "LCPN-GTREE":       lambda X, Y: fs_global_tree(X, Y),
            "LCPN-GMI":         lambda X, Y: fs_global_mi(X, Y),
            "LCPN-CHI2-MLP":    lambda X, Y: fs_parent_chi2(X, Y, c2p),
            "LCPN-TREEFS-MLP":  lambda X, Y: fs_parent_tree(X, Y, c2p),
            "LCPN-GCHI2-MLP":   lambda X, Y: fs_global_chi2(X, Y),
            "LCPN-GTREE-MLP":   lambda X, Y: fs_global_tree(X, Y),
            "LCPN-GMI-MLP":     lambda X, Y: fs_global_mi(X, Y),
        }

        X_all, Y_all = load_full_arff(ds, c2p, RAW)
        strata = np.array([primary_leaf(s) for s in Y_all])
        skf    = StratifiedKFold(N_FOLDS, shuffle=True, random_state=42)

        # >>> Prepare per-fold output paths now (constant across datasets)
        out_paths = {
            fold_id+1: RAW.parent / f"results_fold{fold_id+1}_MINC{MINC}.csv"
            for fold_id in range(N_FOLDS)
        }

        stats = Parallel(n_jobs=N_CORES, backend="loky")(
            delayed(run_one_fold)(
                ds, fold_id + 1, train_idx, valid_idx,
                X_all, Y_all, c2p, scenarios, MINC, out_paths[fold_id + 1]
            )
            for fold_id, (train_idx, valid_idx)
            in enumerate(skf.split(X_all, strata))
        )

        # >>> Report per-dataset progress; files have already been appended
        for fidx, (app, skp, fail) in enumerate(stats, start=1):
            print(f"→ {ds} fold{fidx}: +{app} appended, {skp} skipped (resume), {fail} failed")

# -------------------------------- CLI -----------------------------------------
if __name__ == "__main__":
    main()