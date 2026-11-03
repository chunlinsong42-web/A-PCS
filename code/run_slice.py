# -*- coding: utf-8 -*-
"""Parallel driver: run one (dataset, trait[, seed]) slice of the full experiment.
Usage: python run_slice.py --dataset sim --trait T1_block_oligo --seeds 1,2,3 --out ../results/sim_full_v3__T1.csv
"""
import argparse
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import run_experiment
from run_sim import sim_methods


def load_dataset(name):
    if name == "sim":
        d = np.load("../data/sim_2026.npz", allow_pickle=True)
        return d["X"], {k: d[f"y_{k}"] for k in d["trait_names"]}, d["pos"], d["chrom"], 120e3, 10
    if name == "mice":
        d = np.load("../data/mice.npz", allow_pickle=True)
        names = list(d["trait_names"])
        traits = {t: d["Y"][i] for i, t in enumerate(names)}
        return d["X"], traits, d["pos"], d["chrom"], 4e6, 12
    if name == "wheat":
        d = np.load("../data/wheat.npz")
        Y = d["Y"]
        traits = {f"wheat_env{c}": Y[:, j] for j, c in enumerate([1, 2, 4, 5])}
        pos = np.arange(d["X"].shape[1], dtype=np.float64)
        chrom = np.ones(d["X"].shape[1], dtype=int)
        return d["X"], traits, pos, chrom, 40.0, 8
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--trait", required=True)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--version", default="v3f")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    X, traits, pos, chrom, W, mm = load_dataset(args.dataset)
    y = {args.trait: traits[args.trait]}
    methods = sim_methods(args.version if args.version != "v3f" else "v3full")
    match = "APCS2w" if "APCS2w" in methods else "APCS2"
    run_experiment(X, y, pos, chrom, methods, W_default=W,
                   seeds=tuple(int(s) for s in args.seeds.split(",")),
                   n_folds=5, out_csv=args.out, tag=f"{args.dataset}",
                   min_markers=mm, random_match=match,
                   log=lambda *a: print(*a, flush=True))


if __name__ == "__main__":
    main()
