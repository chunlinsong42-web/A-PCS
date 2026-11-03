# -*- coding: utf-8 -*-
"""Quick gamma sweep for APCS2w on T1 (5 folds x 1 seed) to pick final gamma grid."""
import numpy as np
import sys, os, time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apcs_core import (qc_markers, standardize, pcc, ld_precompute, apcs2,
                       tune_lambda_val, fit_predict_gblup, corr_abs)
from experiments import make_folds, LAMS

d = np.load("../data/sim_2026.npz", allow_pickle=True)
X = d["X"]
y = d["y_T1_block_oligo"]
pos, chrom = d["pos"], d["chrom"]
Xq, keep_idx = qc_markers(X)
pos, chrom = pos[keep_idx], chrom[keep_idx]
y_std = (y - y.mean()) / y.std()

rows = []
folds = make_folds(len(y_std), 5, 1)
for fi, te in enumerate(folds):
    tr = np.setdiff1d(np.arange(len(y_std)), te)
    rng = np.random.default_rng(1000 + fi)
    perm = rng.permutation(tr)
    n_val = int(0.25 * len(tr))
    val, fit = perm[:n_val], perm[n_val:]
    Zfit, mu_f, sd_f = standardize(Xq[fit])
    Zval = (Xq[val] - mu_f) / sd_f
    Ztr, mu_t, sd_t = standardize(Xq[tr])
    Zte = (Xq[te] - mu_t) / sd_t
    yfit, yval, ytr, yte = y_std[fit], y_std[val], y_std[tr], y_std[te]
    pre_fit = ld_precompute(Zfit, pos, chrom, 120e3, 10)
    pre_tr = ld_precompute(Ztr, pos, chrom, 120e3, 10)
    for q in (1e-3, 3e-3):
        for gamma in (6.0, 8.0, 12.0, 20.0):
            cfg = dict(W=120e3, q=q, b=0.7, t_col_min=0.7, t_col_max=0.95, gamma=gamma)
            keep_f, _ = apcs2(Zfit, yfit, pos, chrom, 120e3, q, 0.7, 0.7, 0.95, pre_fit)
            w_f = np.where(keep_f, np.power(np.maximum(corr_abs(Zfit, yfit), 1e-6), gamma), 0.0)[keep_f]
            lam, _ = tune_lambda_val(Zfit, yfit, Zval, yval, keep_f, LAMS, w_f=w_f)
            keep, _ = apcs2(Ztr, ytr, pos, chrom, 120e3, q, 0.7, 0.7, 0.95, pre_tr)
            w_kept = np.where(keep, np.power(np.maximum(corr_abs(Ztr, ytr), 1e-6), gamma), 0.0)[keep]
            pred = fit_predict_gblup(Ztr, ytr, Zte, keep, lam, w_kept=w_kept)
            rows.append(dict(q=q, gamma=gamma, fold=fi, pcc=pcc(pred, yte), n=int(keep.sum())))
            print(rows[-1])
pd.DataFrame(rows).to_csv("../results/gamma_sweep2.csv", index=False)
print(pd.DataFrame(rows).groupby(["q", "gamma"])[["pcc", "n"]].mean().round(4).to_string())
