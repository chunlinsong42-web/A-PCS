# -*- coding: utf-8 -*-
"""
Cross-population generalization: leave-one-cluster-out (LOCO) validation.

Clusters = genotype-based population structure (KMeans on top-20 genotype PCs).
For each held-out cluster: screening + GBLUP fit on the remaining clusters,
predict the held-out one. Methods: All / PCS / APCS (each tuned on an inner
validation split of the training clusters).
"""
import argparse
import numpy as np
import os
import sys
import pandas as pd
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apcs_core import (qc_markers, standardize, pcc, ld_precompute,
                       fit_predict_gblup, tune_lambda_val)
from experiments import LAMS, run_selection
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def loco_eval(X_raw, traits, pos, chrom, methods_cfg, W_default, out_csv, tag,
              n_clusters=5, seeds=(1, 2, 3), min_markers=10, log=print,
              subpop=None):
    rows = []
    Xq, keep_idx = qc_markers(X_raw)
    pos = np.asarray(pos, dtype=np.float64)[keep_idx]
    chrom = np.asarray(chrom)[keep_idx]
    Xq = Xq
    for tname, y_raw in traits.items():
        y = np.asarray(y_raw, dtype=np.float64)
        ok = np.isfinite(y)
        Xq_t, y_t = Xq[ok], y[ok]
        y_std = (y_t - y_t.mean()) / y_t.std()
        # cluster assignment from genotype PCs (fixed across seeds)
        Zall, _, _ = standardize(Xq_t)
        n_pc = min(20, Zall.shape[0] - 1)
        if subpop is not None:
            clusters = np.asarray(subpop)[ok]
        else:
            pcs = PCA(n_components=n_pc, random_state=0).fit_transform(Zall)
            clusters = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(pcs)
        uclusters = np.unique(clusters)
        for seed in seeds:
            for held in uclusters:
                te = np.where(clusters == held)[0]
                tr = np.where(clusters != held)[0]
                rng = np.random.default_rng(1000 * seed + int(held))
                perm = rng.permutation(tr)
                n_val = max(40, int(0.2 * len(tr)))
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
                    if cfgs:
                        best = None
                        for cfg in cfgs:
                            W = cfg.get("W", W_default)
                            keep_f = run_selection(method, cfg, Zfit, yfit, pos,
                                                   chrom, W, pre_fit)
                            if keep_f is None or keep_f.sum() < 5:
                                continue
                            lam_c, pv = tune_lambda_val(Zfit, yfit, Zval, yval,
                                                        keep_f, LAMS)
                            if best is None or pv > best["pv"]:
                                best = dict(cfg=cfg, pv=pv, lam=lam_c)
                        cfg, lam = best["cfg"], best["lam"]
                    else:
                        cfg, lam = {}, None
                    W = cfg.get("W", W_default)
                    if method == "All":
                        keep = np.ones(Ztr.shape[1], dtype=bool)
                    else:
                        pre_tr = ld_precompute(Ztr, pos, chrom, W, min_markers)
                        keep = run_selection(method, cfg, Ztr, ytr, pos, chrom, W,
                                             {W: pre_tr})
                    if lam is None:
                        keep_f = run_selection(method, {}, Zfit, yfit, pos, chrom,
                                               W, pre_fit) if method != "All" \
                            else np.ones(Zfit.shape[1], dtype=bool)
                        lam, _ = tune_lambda_val(Zfit, yfit, Zval, yval,
                                                 keep_f, LAMS)
                    pred = fit_predict_gblup(Ztr, ytr, Zte, keep, lam)
                    pc = pcc(pred, yte)
                    rm = float(np.sqrt(np.mean((pred - yte) ** 2)))
                    rows.append(dict(dataset=tag, trait=tname, method=method, seed=seed,
                                     fold=int(held), pcc=pc, rmse=rm, n_sel=int(keep.sum()),
                                     runtime_s=round(time.time() - t1, 1)))
                    log(f"[{tag}-LOCO] {tname} held{held} s{seed} {method}: "
                        f"PCC={pc:.4f} n={int(keep.sum())}")
                    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["sim", "mice"], default="mice")
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
        X = d["X"]
        meta = {k: d[k] for k in ("pos", "chrom", "subpop")}
        traits = {k: d[f"y_{k}"] for k in d["trait_names"]}
        if args.trait:
            traits = {args.trait: traits[args.trait]}
        out = args.out or "../results/loco_sim.csv"
        loco_eval(X, traits, meta["pos"], meta["chrom"], methods, 120e3,
                  out, "sim", subpop=meta["subpop"], min_markers=10)
    else:
        d = np.load("../data/mice.npz", allow_pickle=True)
        names = list(d["trait_names"])
        Y = d["Y"]
        keep_t = ["Obesity.BMI", "Obesity.BodyLength", "Biochem.Glucose",
                  "Biochem.HDL", "Biochem.LDL", "Biochem.Tot.Cholesterol"]
        if args.trait:
            keep_t = [args.trait]
        traits = {t: Y[names.index(t)] for t in keep_t}
        out = args.out or "../results/loco_mice.csv"
        loco_eval(d["X"], traits, d["pos"], d["chrom"], methods, 4e6,
                  out, "mice", n_clusters=5, min_markers=12)
    print("done")
