# -*- coding: utf-8 -*-
"""Run mice (real, n=1814, 10346 SNPs, physical map) experiments."""
import argparse
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import run_experiment
from run_sim import sim_methods

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v3")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--traits",
                    default="Obesity.BMI;Obesity.BodyLength;Obesity.EndNormalBW;"
                            "Biochem.Glucose;Biochem.HDL;Biochem.Tot.Cholesterol")
    args = ap.parse_args()

    d = np.load("../data/mice.npz", allow_pickle=True)
    X, chrom, pos = d["X"], d["chrom"], d["pos"]
    names = list(d["trait_names"])
    Y = d["Y"]
    if args.traits == "all":
        tnames = names
    else:
        tnames = args.traits.split(";")
    traits = {t: Y[names.index(t)] for t in tnames}
    methods = sim_methods(args.version)
    os.makedirs("../results", exist_ok=True)
    out = f"../results/mice_{args.version}.csv"
    run_experiment(X, traits, pos, chrom, methods, W_default=4e6,
                   seeds=tuple(int(s) for s in args.seeds.split(",")),
                   n_folds=5, out_csv=out, tag="mice", min_markers=12)
    print("done ->", out)

if __name__ == "__main__":
    main()
