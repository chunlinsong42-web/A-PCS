# -*- coding: utf-8 -*-
"""
Diagnostics for the report:
  1. LD-decay profiles per regime (simulation) -> Fig 1 data
  2. LD landscape L(pos) + adaptive thresholds along a chromosome -> Fig 2 data
  3. QTL recall of screening methods (direct + LD-proxy), by regime -> Table
  4. mice per-window half-decay profile -> real-data heterogeneity figure
All diagnostics fit screening on the FULL data (no prediction involved).
"""
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apcs_core import qc_markers, standardize, ld_precompute, pcs_fixed, apcs, apcs2

OUT = "../results"
os.makedirs(OUT, exist_ok=True)


def ld_decay_profile(Z, pos, regime, max_kb=400.0, n_bins=12, spacing=6e3):
    prof = {}
    bins = np.linspace(0.5 * spacing, max_kb * 1e3, n_bins + 1)
    idx_all = np.arange(len(pos))
    # subsample markers to keep pair computation manageable
    step = max(1, len(pos) // 2000)
    sub = idx_all[::step]
    Zs = np.ascontiguousarray(Z[:, sub])
    R = (Zs.T @ Zs) / Zs.shape[0]
    d = np.abs(pos[sub][None, :] - pos[sub][:, None])
    iu = np.triu_indices(len(sub), 1)
    dv, r2v = d[iu], R[iu] ** 2
    for reg in np.unique(regime):
        mask = regime[sub][iu[0]] == regime[sub][iu[1]]
        prof_reg = []
        for b in range(n_bins):
            m = mask & (dv >= bins[b]) & (dv < bins[b + 1])
            prof_reg.append(float(r2v[m].mean()) if m.sum() > 10 else None)
        prof[str(reg)] = dict(bin_edges=[float(x / 1e3) for x in bins],
                              r2=prof_reg)
    return prof


def recall_stats(keep, causal, regime, pos, proxy_r2_min=0.5, proxy_span=500e3, Z=None):
    """Primary metric: LD coverage = per causal site, max r^2 between the site and
    any retained marker (0 if none within proxy_span); averaged per regime.
    Secondary: binary retention (site itself or a proxy with r^2>=proxy_r2_min kept)."""

    def coverage(idx):
        lo = np.searchsorted(pos, pos[idx] - proxy_span)
        hi = np.searchsorted(pos, pos[idx] + proxy_span)
        cand = np.arange(lo, hi)
        cand = cand[keep[cand]]
        if len(cand) == 0:
            return 0.0
        zi = Z[:, idx]
        rr = (Z[:, cand].T @ zi) / Z.shape[0]
        return float((rr * rr).max())

    def binary(idx):
        return coverage(idx) >= proxy_r2_min or keep[idx]

    res = {}
    for name, sel in [("all", np.ones(len(causal), bool)),
                      ("hotspot", regime[causal] == "hotspot"),
                      ("moderate", regime[causal] == "moderate"),
                      ("block", regime[causal] == "block")]:
        idxs = causal[sel]
        if len(idxs) == 0:
            continue
        covs = [coverage(i) for i in idxs]
        res[name] = dict(n=int(len(idxs)),
                         coverage=float(np.mean(covs)),
                         binary=float(np.mean([binary(i) for i in idxs])))
    return res


def run_sim_diagnostics(seed=2026):
    d = np.load(f"../data/sim_{seed}.npz", allow_pickle=True)
    X, meta = d["X"], {k: d[k] for k in ("pos", "chrom", "regime")}
    pos, chrom, regime = meta["pos"], meta["chrom"], meta["regime"].astype(str)
    Xq, keep_idx = qc_markers(X)
    pos, chrom, regime = pos[keep_idx], chrom[keep_idx], regime[keep_idx]
    Z, _, _ = standardize(Xq)
    out = {}
    # 1. LD decay profiles
    out["ld_profile"] = ld_decay_profile(Z, pos, regime)
    # 2/3. landscape + recall for each trait
    # fit on full data with representative configs
    pcs_cfg = dict(t_sig=0.08, t_col=0.8)
    apcs_cfg = dict(W=120e3, t_sig_min=0.04, t_sig_max=0.12, t_col_min=0.5, t_col_max=0.95)
    apcs2_cfg = dict(W=120e3, q=0.3, b=0.7, t_col_min=0.7, t_col_max=0.95)
    pre = ld_precompute(Z, pos, chrom, 120e3, 10)
    L, dh_m = pre[0], pre[1]
    out["landscape"] = dict(L=L.tolist(), dh=dh_m.tolist(), pos=pos.tolist(),
                            chrom=chrom.tolist())
    for tname in d["trait_names"]:
        y = d[f"y_{tname}"]
        ys = (y - y.mean()) / y.std()
        causal = d[f"causal_{tname}"]
        # causal indices refer to pre-QC columns -> map through keep_idx
        causal_q = np.searchsorted(keep_idx, causal)
        keep_pcs, _ = pcs_fixed(Z, ys, pos, chrom, pcs_cfg["t_sig"],
                                pcs_cfg["t_col"], 120e3)
        keep_apcs, _ = apcs(Z, ys, pos, chrom, apcs_cfg["W"],
                            apcs_cfg["t_sig_min"], apcs_cfg["t_sig_max"],
                            apcs_cfg["t_col_min"], apcs_cfg["t_col_max"], pre)
        keep_apcs2, _ = apcs2(Z, ys, pos, chrom, apcs2_cfg["W"], apcs2_cfg["q"],
                              apcs2_cfg["b"], apcs2_cfg["t_col_min"],
                              apcs2_cfg["t_col_max"], pre)
        out[f"recall_{tname}"] = {
            "PCS": recall_stats(keep_pcs, causal_q, regime, pos, Z=Z),
            "APCS": recall_stats(keep_apcs, causal_q, regime, pos, Z=Z),
            "APCS2": recall_stats(keep_apcs2, causal_q, regime, pos, Z=Z),
            "n_selected": {"PCS": int(keep_pcs.sum()), "APCS": int(keep_apcs.sum()),
                           "APCS2": int(keep_apcs2.sum())},
        }
    # thresholds along genome for fig2
    t_sig = apcs_cfg["t_sig_min"] + (apcs_cfg["t_sig_max"] - apcs_cfg["t_sig_min"]) * L
    t_col = apcs_cfg["t_col_max"] - (apcs_cfg["t_col_max"] - apcs_cfg["t_col_min"]) * L
    out["landscape"]["t_sig"] = t_sig.tolist()
    out["landscape"]["t_col"] = t_col.tolist()
    with open(f"{OUT}/diag_sim.json", "w") as f:
        json.dump(out, f)
    print("saved diag_sim.json")


def run_mice_ld():
    d = np.load("../data/mice.npz", allow_pickle=True)
    X, chrom, pos = d["X"], d["chrom"], d["pos"]
    Xq, keep_idx = qc_markers(X)
    pos, chrom = pos[keep_idx], chrom[keep_idx]
    Z, _, _ = standardize(Xq)
    pre = ld_precompute(Z, pos, chrom, 4e6, 12)
    # per-window d_half
    from apcs_core import build_windows, window_dhalf
    wins = build_windows(pos, chrom, 4e6, 12)
    dh = window_dhalf(Z, pos, wins)
    centers = np.array([0.5 * (pos[i0] + pos[i1 - 1]) for (_, i0, i1) in wins])
    chrw = np.array([c for (c, _, _) in wins])
    np.savez(f"{OUT}/diag_mice_ld.npz", centers=centers, chrw=chrw, dh=dh,
             L_marker=pre[0], pos_marker=pos, chrom_marker=chrom)
    print("saved diag_mice_ld.npz;", len(wins), "windows; d_half Mb percentiles:",
          np.round(np.nanpercentile(dh / 1e6, [10, 50, 90]), 2))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    if what in ("sim", "both"):
        run_sim_diagnostics()
    if what in ("mice", "both"):
        run_mice_ld()
