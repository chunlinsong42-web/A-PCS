# -*- coding: utf-8 -*-
"""
Leakage-free cross-validation harness comparing A-PCS vs baselines.

Protocol (per trait x seed x fold):
  5-fold train/test split; within train an inner validation split (25%) used ONLY
  for hyper-parameter (threshold grid + GBLUP lambda) selection; final selection +
  fit on the full training fold. Feature selection never sees test data.
Metrics: PCC / RMSE on standardized y. Results appended incrementally to CSV.
"""
import numpy as np
import pandas as pd
import time

from apcs_core import (qc_markers, standardize, pcc, ld_precompute, pcs_fixed,
                       apcs, apcs_fix_sig, apcs_fix_col, ld_prune, apcs2,
                       apcs2_fix_sig, apcs2_fix_col,
                       gblup_eigh, gblup_predict_from_eigh, tune_lambda_on_val,
                       tune_lambda_val, fit_predict_gblup, method_weights)

LAMS = np.logspace(-4, 1, 8)


def make_folds(n, k, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [np.sort(f) for f in np.array_split(perm, k)]


def run_selection(method, cfg, Z, y, pos, chrom, W, pre_by_W):
    """Boolean keep-mask for one method config on standardized Z, y."""
    if method == "All":
        return np.ones(Z.shape[1], dtype=bool)
    if method == "LDPrune":
        return ld_prune(Z, pos, chrom, W, r2_max=0.8)
    if method == "PCS":
        kept, _ = pcs_fixed(Z, y, pos, chrom, cfg["t_sig"], cfg["t_col"], W)
        return kept
    if method == "APCS":
        pre = pre_by_W[cfg["W"]]
        kept, _ = apcs(Z, y, pos, chrom, cfg["W"], cfg["t_sig_min"], cfg["t_sig_max"],
                       cfg["t_col_min"], cfg["t_col_max"], pre)
        return kept
    if method == "APCS_fixSig":
        pre = pre_by_W[cfg["W"]]
        kept, _ = apcs_fix_sig(Z, y, pos, chrom, cfg["W"], cfg["t_sig"],
                               cfg["t_col_min"], cfg["t_col_max"], pre)
        return kept
    if method == "APCS_fixCol":
        pre = pre_by_W[cfg["W"]]
        kept, _ = apcs_fix_col(Z, y, pos, chrom, cfg["W"], cfg["t_sig_min"],
                               cfg["t_sig_max"], cfg["t_col"], pre)
        return kept
    if method in ("APCS2", "APCS2w"):
        pre = pre_by_W[cfg["W"]]
        kept, _ = apcs2(Z, y, pos, chrom, cfg["W"], cfg["q"], cfg["b"],
                        cfg["t_col_min"], cfg["t_col_max"], pre)
        return kept
    if method == "APCS2_fixSig":
        pre = pre_by_W[cfg["W"]]
        kept, _ = apcs2_fix_sig(Z, y, pos, chrom, cfg["W"], cfg["q"],
                                cfg["t_col_min"], cfg["t_col_max"], pre)
        return kept
    if method == "APCS2_fixCol":
        pre = pre_by_W[cfg["W"]]
        kept, _ = apcs2_fix_col(Z, y, pos, chrom, cfg["W"], cfg["q"], cfg["b"],
                                cfg["t_col"], pre)
        return kept
    if method == "Random":
        return None  # handled by caller
    raise ValueError(method)


def run_experiment(X_raw, traits, pos, chrom, methods_cfg, W_default,
                   seeds=(1, 2, 3), n_folds=5, out_csv="results.csv", tag="sim",
                   random_match="APCS", min_markers=10, log=print):
    """traits: dict name -> raw y. methods_cfg: dict method -> list of config dicts
    (empty list = untuned)."""
    rows = []
    Xq, keep_idx = qc_markers(X_raw)
    pos_q = np.asarray(pos, dtype=np.float64)[keep_idx]
    chrom_q = np.asarray(chrom)[keep_idx]
    log(f"[{tag}] QC: {X_raw.shape[1]} -> {Xq.shape[1]} markers")
    all_W = sorted({c.get("W", W_default) for cfgs in methods_cfg.values()
                    for c in cfgs} | {W_default})
    for tname, y_raw in traits.items():
        y = np.asarray(y_raw, dtype=np.float64)
        ok = np.isfinite(y)
        Xq_t, y_t = Xq[ok], y[ok]
        pos_t, chrom_t = pos_q, chrom_q  # QC drops markers, not individuals
        y_std = (y_t - y_t.mean()) / y_t.std()
        for seed in seeds:
            folds = make_folds(len(y_std), n_folds, seed)
            for fi, te in enumerate(folds):
                tr = np.setdiff1d(np.arange(len(y_std)), te)
                t0 = time.time()
                rng = np.random.default_rng(1000 * seed + fi)
                perm = rng.permutation(tr)
                n_val = max(50, int(0.25 * len(tr)))
                val, fit = perm[:n_val], perm[n_val:]
                Zfit, mu_f, sd_f = standardize(Xq_t[fit])
                Zval = (Xq_t[val] - mu_f) / sd_f
                yfit, yval = y_std[fit], y_std[val]
                Ztr, mu_t, sd_t = standardize(Xq_t[tr])
                Zte = (Xq_t[te] - mu_t) / sd_t
                ytr, yte = y_std[tr], y_std[te]
                pre_fit, pre_tr = {}, {}
                for W in all_W:
                    pre_fit[W] = ld_precompute(Zfit, pos_t, chrom_t, W, min_markers)
                    pre_tr[W] = ld_precompute(Ztr, pos_t, chrom_t, W, min_markers)
                t_prep = time.time() - t0

                n_sel_random = None
                for method, cfgs in methods_cfg.items():
                    t1 = time.time()
                    if cfgs:  # tuned method: config + lambda on inner validation split
                        best = None
                        for cfg in cfgs:
                            W = cfg.get("W", W_default)
                            keep_f = run_selection(method, cfg, Zfit, yfit,
                                                   pos_t, chrom_t, W, pre_fit)
                            if keep_f is None or keep_f.sum() < 5:
                                continue
                            w_f = method_weights(method, cfg, Zfit, yfit, keep_f)
                            lam_c, pv = tune_lambda_val(Zfit, yfit, Zval, yval,
                                                        keep_f, LAMS, w_f=w_f)
                            if best is None or pv > best["pv"]:
                                best = dict(cfg=cfg, pv=pv, lam=lam_c)
                        cfg, lam = best["cfg"], best["lam"]
                    else:
                        cfg, lam = {}, None
                    W = cfg.get("W", W_default)
                    # final selection on the full training fold
                    if method == "Random":
                        k = n_sel_random if n_sel_random else 100
                        keep = np.zeros(Ztr.shape[1], dtype=bool)
                        rng2 = np.random.default_rng(777)
                        keep[rng2.choice(Ztr.shape[1], size=k, replace=False)] = True
                    else:
                        keep = run_selection(method, cfg, Ztr, ytr, pos_t,
                                             chrom_t, W, pre_tr)
                    if lam is None:  # untuned method: lambda still via inner validation
                        if method == "Random":
                            keep_f = np.zeros(Zfit.shape[1], dtype=bool)
                            rng2 = np.random.default_rng(778)
                            keep_f[rng2.choice(Zfit.shape[1], size=max(5, int(keep.sum())),
                                               replace=False)] = True
                        else:
                            keep_f = run_selection(method, cfg, Zfit, yfit, pos_t,
                                                   chrom_t, W, pre_fit)
                        lam, _ = tune_lambda_val(Zfit, yfit, Zval, yval, keep_f, LAMS)
                    w_kept = method_weights(method, cfg, Ztr, ytr, keep)
                    pred = fit_predict_gblup(Ztr, ytr, Zte, keep, lam, w_kept=w_kept)
                    pcc_te = pcc(pred, yte)
                    rmse_te = float(np.sqrt(np.mean((pred - yte) ** 2)))
                    dt = time.time() - t1
                    rows.append(dict(dataset=tag, trait=tname, method=method, seed=seed,
                                     fold=fi, pcc=pcc_te, rmse=rmse_te,
                                     n_sel=int(keep.sum()), lam=float(lam),
                                     runtime_s=round(dt, 2), prep_s=round(t_prep, 2),
                                     config=str(cfg)))
                    if method == random_match:
                        n_sel_random = int(keep.sum())
                    log(f"[{tag}] {tname} s{seed} f{fi} {method}: "
                        f"PCC={pcc_te:.4f} n={int(keep.sum())} ({dt:.1f}s)")
                pd.DataFrame(rows).to_csv(out_csv, index=False)
    return pd.DataFrame(rows)
