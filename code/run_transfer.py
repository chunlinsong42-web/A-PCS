# -*- coding: utf-8 -*-
"""
Cross-dataset hyper-parameter transfer: apply the (q, b, t_col_min, t_col_max)
threshold geometry tuned on the SIMULATION dataset directly to the REAL mice
dataset without re-tuning (W is re-scaled to marker density, which is a data
protocol parameter, not a threshold semantic). Compare against per-dataset
tuned counterparts from mice_v3.csv.
"""
import numpy as np
import os
import sys
import pandas as pd
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apcs_core import (qc_markers, standardize, pcc, ld_precompute,
                       fit_predict_gblup, tune_lambda_val)
from experiments import make_folds, LAMS, run_selection

TRAIT_LIST = ["Obesity.BMI", "Obesity.BodyLength", "Obesity.EndNormalBW",
              "Biochem.Glucose", "Biochem.HDL", "Biochem.Tot.Cholesterol"]


def best_configs(sim_csv, methods=("APCS2w", "PCS")):
    """Most frequently selected config per method across the simulation full runs
    (mode of inner-validation choices = the 'tuned-on-sim' threshold geometry)."""
    df = pd.read_csv(sim_csv)
    out = {}
    for m in methods:
        cfgs = df[df.method == m]["config"]
        if len(cfgs) == 0:
            continue
        cnt = Counter(cfgs)
        out[m] = eval(cnt.most_common(1)[0][0])
    return out


def run_transfer(X_raw, traits, pos, chrom, methods_cfg, W_default, out_csv,
                 min_markers=12, seeds=(1, 2), n_folds=5, log=print):
    rows = []
    Xq, keep_idx = qc_markers(X_raw)
    pos = np.asarray(pos, dtype=np.float64)[keep_idx]
    chrom = np.asarray(chrom)[keep_idx]
    for tname, y_raw in traits.items():
        y = np.asarray(y_raw, dtype=np.float64)
        ok = np.isfinite(y)
        Xq_t, y_t = Xq[ok], y[ok]
        y_std = (y_t - y_t.mean()) / y_t.std()
        for seed in seeds:
            folds = make_folds(len(y_std), n_folds, seed)
            for fi, te in enumerate(folds):
                tr = np.setdiff1d(np.arange(len(y_std)), te)
                rng = np.random.default_rng(1000 * seed + fi)
                perm = rng.permutation(tr)
                n_val = max(40, int(0.25 * len(tr)))
                val, fit = perm[:n_val], perm[n_val:]
                Zfit, mu_f, sd_f = standardize(Xq_t[fit])
                Zval = (Xq_t[val] - mu_f) / sd_f
                Ztr, mu_t, sd_t = standardize(Xq_t[tr])
                Zte = (Xq_t[te] - mu_t) / sd_t
                yfit, yval, ytr, yte = y_std[fit], y_std[val], y_std[tr], y_std[te]
                all_W = sorted({c.get("W", W_default) for c in methods_cfg.values()} | {W_default})
                pre_fit = {W: ld_precompute(Zfit, pos, chrom, W, min_markers) for W in all_W}
                pre_tr = {W: ld_precompute(Ztr, pos, chrom, W, min_markers) for W in all_W}
                for method, cfg in methods_cfg.items():
                    t1 = time.time()
                    W = W_default  # protocol parameter: target dataset marker density
                    keep_f = run_selection(method, cfg, Zfit, yfit, pos, chrom, W, pre_fit)
                    lam, _ = tune_lambda_val(Zfit, yfit, Zval, yval, keep_f, LAMS)
                    keep = run_selection(method, cfg, Ztr, ytr, pos, chrom, W, pre_tr)
                    pred = fit_predict_gblup(Ztr, ytr, Zte, keep, lam)
                    pc = pcc(pred, yte)
                    rows.append(dict(dataset="mice_transfer", trait=tname, method=method + "_tr",
                                     seed=seed, fold=fi, pcc=pc,
                                     rmse=float(np.sqrt(np.mean((pred - yte) ** 2))),
                                     n_sel=int(keep.sum()), lam=float(lam),
                                     runtime_s=round(time.time() - t1, 1), config=str(cfg)))
                    log(f"[transfer] {tname} s{seed} f{fi} {method}_tr: PCC={pc:.4f} "
                        f"n={int(keep.sum())}")
                    pd.DataFrame(rows).to_csv(out_csv, index=False)


if __name__ == "__main__":
    import ast
    import sys as _sys
    sim_csv = "../results/sim_full_v5.csv"
    out_csv = os.environ.get("TRANSFER_OUT", "../results/transfer_mice.csv")
    only_trait = os.environ.get("TRANSFER_TRAIT")
    cfgs = best_configs(sim_csv)
    # rescale W: threshold semantics transfer, window width follows marker density
    for m, c in cfgs.items():
        cfgs[m] = dict(c)  # threshold semantics transfer; W is overridden per dataset
    print("transferred configs:", cfgs)
    d = np.load("../data/mice.npz", allow_pickle=True)
    names = list(d["trait_names"])
    Y = d["Y"]
    tl = [only_trait] if only_trait else TRAIT_LIST
    traits = {t: Y[names.index(t)] for t in tl}
    run_transfer(d["X"], traits, d["pos"], d["chrom"], cfgs, 4e6, out_csv)
    print("done ->", out_csv)
