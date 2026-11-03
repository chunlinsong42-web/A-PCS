# -*- coding: utf-8 -*-
"""Run simulation experiments (dev iteration or full protocol)."""
import argparse
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate import simulate_genome, simulate_traits
from experiments import run_experiment


def sim_methods(version, dev=False):
    """Method configuration grids per algorithm version."""
    PCS = [dict(t_sig=t, t_col=c) for t in (0.05, 0.08, 0.11, 0.14)
           for c in (0.6, 0.8, 0.95)]
    if version == "v1":
        APCS = [dict(W=W, t_sig_min=a, t_sig_max=b, t_col_min=c0, t_col_max=c1)
                for W in (120e3, 240e3)
                for (a, b) in ((0.04, 0.12), (0.06, 0.15))
                for (c0, c1) in ((0.50, 0.95), (0.65, 0.85))]
        FSIG = [dict(W=120e3, t_sig=t, t_col_min=0.50, t_col_max=0.95)
                for t in (0.06, 0.09, 0.12)]
        FCOL = [dict(W=120e3, t_sig_min=0.04, t_sig_max=0.12, t_col=c)
                for c in (0.6, 0.8, 0.95)]
        return {"All": [], "LDPrune": [], "PCS": PCS, "APCS": APCS,
                "APCS_fixSig": FSIG, "APCS_fixCol": FCOL, "Random": []}
    if version == "v2":
        # head-to-head dev: v1 APCS (subset grid) vs v2 APCS2
        APCS = [dict(W=120e3, t_sig_min=a, t_sig_max=b, t_col_min=c0, t_col_max=c1)
                for (a, b) in ((0.04, 0.12), (0.06, 0.15))
                for (c0, c1) in ((0.50, 0.95), (0.65, 0.85))]
        APCS2 = [dict(W=120e3, q=q, b=b, t_col_min=c0, t_col_max=c1)
                 for q in (1e-3, 1e-2) for b in (0.5, 0.7)
                 for (c0, c1) in ((0.50, 0.90), (0.60, 0.95))]
        F2SIG = [dict(W=120e3, q=1e-2, t_col_min=c0, t_col_max=c1)
                 for (c0, c1) in ((0.50, 0.90), (0.60, 0.95))]
        F2COL = [dict(W=120e3, q=1e-2, b=0.7, t_col=c) for c in (0.6, 0.8)]
        return {"All": [], "LDPrune": [], "PCS": PCS, "APCS": APCS, "APCS2": APCS2,
                "APCS2_fixSig": F2SIG, "APCS2_fixCol": F2COL, "Random": []}
    if version == "v3":
        # final: softer block dedup + weighted-kernel flagship
        APCS2 = [dict(W=120e3, q=q, b=b, t_col_min=0.7, t_col_max=0.95)
                 for q in (1e-3, 3e-3, 1e-2) for b in (0.5, 0.7)]
        APCS2w = [dict(W=120e3, q=q, b=0.7, t_col_min=0.7, t_col_max=0.95, gamma=g)
                  for q in (1e-3, 3e-3, 1e-2) for g in (1.0, 2.0)]
        F2SIG = [dict(W=120e3, q=1e-2, t_col_min=0.7, t_col_max=0.95)]
        F2COL = [dict(W=120e3, q=1e-2, b=0.7, t_col=0.8)]
        return {"All": [], "LDPrune": [], "PCS": PCS, "APCS2": APCS2, "APCS2w": APCS2w,
                "APCS2_fixSig": F2SIG, "APCS2_fixCol": F2COL, "Random": []}
    if version == "v5":
        # q upper bound lifted again (mice inner validation saturated q=0.1);
        # gamma=0 makes the weighted family a superset of the unweighted kernel
        APCS2 = [dict(W=120e3, q=q, b=b, t_col_min=0.7, t_col_max=0.95)
                 for q in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1) for b in (0.5, 0.7)]
        APCS2w = [dict(W=120e3, q=q, b=0.7, t_col_min=0.7, t_col_max=0.95, gamma=g)
                  for q in (1e-2, 3e-2, 1e-1, 3e-1) for g in (0.0, 1.0, 4.0, 8.0)]
        F2SIG = [dict(W=120e3, q=1e-2, t_col_min=0.7, t_col_max=0.95)]
        F2COL = [dict(W=120e3, q=1e-2, b=0.7, t_col=0.8)]
        return {"All": [], "LDPrune": [], "PCS": PCS, "APCS2": APCS2, "APCS2w": APCS2w,
                "APCS2_fixSig": F2SIG, "APCS2_fixCol": F2COL, "Random": []}
    if version == "v4":
        # widened q grid: lets inner validation pick loose admission on small-p
        # datasets (mice/wheat) while keeping tight admission available for p>>n
        APCS2 = [dict(W=120e3, q=q, b=b, t_col_min=0.7, t_col_max=0.95)
                 for q in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1) for b in (0.5, 0.7)]
        APCS2w = [dict(W=120e3, q=q, b=0.7, t_col_min=0.7, t_col_max=0.95, gamma=g)
                  for q in (3e-3, 1e-2, 3e-2) for g in (1.0, 4.0, 8.0)]
        F2SIG = [dict(W=120e3, q=1e-2, t_col_min=0.7, t_col_max=0.95)]
        F2COL = [dict(W=120e3, q=1e-2, b=0.7, t_col=0.8)]
        return {"All": [], "LDPrune": [], "PCS": PCS, "APCS2": APCS2, "APCS2w": APCS2w,
                "APCS2_fixSig": F2SIG, "APCS2_fixCol": F2COL, "Random": []}
    # v3 full (final protocol): gamma-swept weighted kernel flagship
    APCS2 = [dict(W=120e3, q=q, b=b, t_col_min=0.7, t_col_max=0.95)
             for q in (1e-3, 3e-3, 1e-2) for b in (0.5, 0.7)]
    APCS2w = [dict(W=120e3, q=q, b=0.7, t_col_min=0.7, t_col_max=0.95, gamma=g)
              for q in (1e-3, 3e-3) for g in (1.0, 4.0, 8.0)]
    F2SIG = [dict(W=120e3, q=1e-2, t_col_min=0.7, t_col_max=0.95)]
    F2COL = [dict(W=120e3, q=1e-2, b=0.7, t_col=0.8)]
    return {"All": [], "LDPrune": [], "PCS": PCS, "APCS2": APCS2, "APCS2w": APCS2w,
            "APCS2_fixSig": F2SIG, "APCS2_fixCol": F2COL, "Random": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "dev", "full"], default="dev")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    if args.mode == "smoke":
        data_path = f"../data/sim_smoke.npz"
    else:
        data_path = f"../data/sim_{args.seed}.npz"
    if os.path.exists(data_path):
        d = np.load(data_path, allow_pickle=True)
        X, meta = d["X"], {k: d[k] for k in
                           ("pos", "chrom", "regime", "subpop", "maf", "spacing")}
        traits_raw = {k: d[f"y_{k}"] for k in d["trait_names"]}
        causal = {k: d[f"causal_{k}"] for k in d["trait_names"]}
    else:
        n_ind = 160 if args.mode == "smoke" else 1600
        X, meta = simulate_genome(n_ind=n_ind, seed=args.seed)
        tr = simulate_traits(X, meta, seed=7)
        traits_raw = {k: v["y"] for k, v in tr.items()}
        causal = {k: v["causal"] for k, v in tr.items()}
        np.savez_compressed(data_path, X=X, pos=meta["pos"], chrom=meta["chrom"],
                            regime=meta["regime"], subpop=meta["subpop"],
                            maf=meta["maf"], spacing=meta["spacing"],
                            trait_names=np.array(list(traits_raw.keys())),
                            **{f"y_{k}": v for k, v in traits_raw.items()},
                            **{f"causal_{k}": v for k, v in causal.items()})
    if args.mode == "smoke":
        traits = {list(traits_raw)[0]: traits_raw[list(traits_raw)[0]]}
        seeds = (1,)
        methods = {"All": [], "PCS": [dict(t_sig=0.08, t_col=0.8)],
                   "APCS2": [dict(W=120e3, q=1e-2, b=0.7, t_col_min=0.5,
                                  t_col_max=0.95)], "Random": []}
        out = f"../results/sim_smoke.csv"
    elif args.mode == "dev":
        keep = ["T1_block_oligo", "T2_polygenic"]
        traits = {k: traits_raw[k] for k in keep}
        seeds = (1, 2)
        methods = sim_methods(args.version, dev=True)
        out = f"../results/sim_dev_{args.version}.csv"
    else:
        traits = traits_raw
        seeds = (1, 2, 3)
        methods = sim_methods("v3f" if args.version == "v3" else args.version)
        out = f"../results/sim_full_{args.version}.csv"
    os.makedirs("../results", exist_ok=True)
    match = "APCS2w" if "APCS2w" in methods else ("APCS2" if "APCS2" in methods else "APCS")
    run_experiment(X, traits, meta["pos"], meta["chrom"], methods,
                   W_default=120e3, seeds=seeds, n_folds=5, out_csv=out,
                   tag=f"sim_{args.mode}", min_markers=10, random_match=match)
    print("done ->", out)


if __name__ == "__main__":
    main()
