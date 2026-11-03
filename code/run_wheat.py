# -*- coding: utf-8 -*-
"""Run wheat (real, n=599, 1279 markers, marker-order pseudo-map) experiments."""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import run_experiment
from run_sim import sim_methods

d = np.load("../data/wheat.npz")
X, Y = d["X"], d["Y"]
traits = {f"wheat_env{c}": Y[:, j] for j, c in enumerate([1, 2, 4, 5])}
pos = np.arange(X.shape[1], dtype=np.float64)  # marker-order pseudo positions
chrom = np.ones(X.shape[1], dtype=int)
methods = sim_methods("v3")
os.makedirs("../results", exist_ok=True)
run_experiment(X, traits, pos, chrom, methods, W_default=40.0,
               seeds=(1, 2, 3), n_folds=5, out_csv="../results/wheat_v3.csv",
               tag="wheat", min_markers=8)
print("done -> ../results/wheat_v3.csv")
