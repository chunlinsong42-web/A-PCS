# -*- coding: utf-8 -*-
"""
Nonlinear downstream-model check: replace GBLUP with Histogram Gradient Boosting,
keeping each screening method's own (GBLUP-tuned) feature set.

Per fold: tune PCS / APCS configs + lambda via inner validation with GBLUP (cheap),
transfer the winning feature sets to HGB. HGB-All serves as the no-screening control.
"""
import argparse
import numpy as np
import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apcs_core import (qc_markers, standardize, pcc, ld_precompute,
                       tune_lambda_val)
from experiments import make_folds, LAMS, run_selection
from sklearn.ensemble import HistGradientBoostingRegressor

HGB_PARAMS = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                  min_samples_leaf=20, l2_regularization=1.0,
                  early_stopping=True, validation_fraction=0.12, random_state=0)


def hgb_fit_predict(Ztr, ytr, Zte):
    m = HistGradientBoostingRegressor(**HGB_PARAMS)
    m.fit(Ztr, ytr)
    return m.predict(Zte)


def best_config_and_mask(method, cfgs, Zfit, yfit, Zval, yval, pos, chrom, W_default, pre_fit):
    best = None
    for cfg in cfgs:
        W = cfg.get("W", W_default)
        keep = run_selection(method, cfg, Zfit, yfit, pos, chrom, W, pre_fit)
        if keep is None or keep.sum() < 5:
            continue
        lam, pv = tune_lambda_val(Zfit, yfit, Zval, yval, keep, LAMS)
        if best is None or pv > best["pv"]:
            best = dict(cfg=cfg, pv=pv, lam=lam, keep=keep)
    return best


def run_hgb(X_raw, traits, pos, chrom, methods_cfg, W_default, seed=1, n_folds=5,
            out_csv="../results/hgb.csv", tag="sim", min_markers=10, log=print):
    rows = []
    Xq, keep_idx = qc_markers(X_raw)
    pos = np.asarray(pos, dtype=np.float64)[keep_idx]
    chrom = np.asarray(chrom)[keep_idx]
    for tname, y_raw in traits.items():
        y = np.asarray(y_raw, dtype=np.float64)
        ok = np.isfinite(y)
        Xq_t, y_t = Xq[ok], y[ok]
        y_std = (y_t - y_t.mean()) / y_t.std()
        folds = make_folds(len(y_std), n_folds, seed)
        for fi, te in enumerate(folds):
            tr = np.setdiff1d(np.arange(len(y_std)), te)
            rng = np.random.default_rng(1000 * seed + fi)
            perm = rng.permutation(tr)
            n_val = max(50, int(0.25 * len(tr)))
            val, fit = perm[:n_val], perm[n_val:]
            Zfit, mu_f, sd_f = standardize(Xq_t[fit])
            Zval = (Xq_t[val] - mu_f) / sd_f
            Ztr, mu_t, sd_t = standardize(Xq_t[tr])
            Zte = (Xq_t[te] - mu_t) / sd_t
            yfit, yval, ytr, yte = y_std[fit], y_std[val], y_std[tr], y_std[te]
            all_W = sorted({c.get("W", W_default) for cfgs in methods_cfg.values()
                            for c in cfgs} | {W_default})
            pre_fit = {W: ld_precompute(Zfit, pos, chrom, W, min_markers) for W in all_W}
            for method, cfgs in methods_cfg.items():
                t1 = time.time()
                if method == "All":
                    keep = np.ones(Ztr.shape[1], dtype=bool)
                else:
                    b = best_config_and_mask(method, cfgs, Zfit, yfit, Zval, yval,
                                             pos, chrom, W_default, pre_fit)
                    cfg, lam = b["cfg"], b["lam"]
                    W = cfg.get("W", W_default)
                    pre_tr = ld_precompute(Ztr, pos, chrom, W, min_markers)
                    keep = run_selection(method, cfg, Ztr, ytr, pos, chrom, W,
                                         {W: pre_tr})
                pred = hgb_fit_predict(Ztr[:, keep], ytr, Zte[:, keep])
                pc = pcc(pred, yte)
                rm = float(np.sqrt(np.mean((pred - yte) ** 2)))
                rows.append(dict(dataset=tag, trait=tname, method=method + "_HGB", seed=seed,
                                 fold=fi, pcc=pc, rmse=rm, n_sel=int(keep.sum()),
                                 runtime_s=round(time.time() - t1, 1)))
                log(f"[{tag}-HGB] {tname} f{fi} {method}: PCC={pc:.4f} n={int(keep.sum())}")
                pd.DataFrame(rows).to_csv(out_csv, index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["sim", "mice"], default="sim")
    ap.add_argument("--version", default="v3")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--trait", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from run_sim import sim_methods
    methods = {k: v for k, v in sim_methods(args.version).items()
               if k in ("PCS", "APCS", "APCS2", "APCS2w", "All")}
    if args.source == "sim":
        d = np.load("../data/sim_2026.npz", allow_pickle=True)
        X, meta = d["X"], {k: d[k] for k in ("pos", "chrom")}
        traits = {k: d[f"y_{k}"] for k in d["trait_names"]}
        if args.trait:
            traits = {args.trait: traits[args.trait]}
        out = args.out or "../results/hgb_sim.csv"
        run_hgb(X, traits, meta["pos"], meta["chrom"], methods, 120e3,
                out_csv=out, tag="sim", min_markers=10)
    else:
        d = np.load("../data/mice.npz", allow_pickle=True)
        names = list(d["trait_names"])
        Y = d["Y"]
        keep_t = ["Obesity.BMI", "Biochem.Glucose", "Biochem.HDL"]
        if args.trait:
            keep_t = [args.trait]
        traits = {t: Y[names.index(t)] for t in keep_t}
        out = args.out or "../results/hgb_mice.csv"
        run_hgb(d["X"], traits, d["pos"], d["chrom"], methods, 4e6,
                out_csv=out, tag="mice", min_markers=12)
    print("done")
