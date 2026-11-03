# -*- coding: utf-8 -*-
"""Merge parallel slice CSVs into final result files."""
import glob
import os
import pandas as pd

RES = "../results"

def merge(pattern, out):
    files = sorted(glob.glob(pattern))
    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f)
            if len(d):
                dfs.append(d)
        except Exception as e:
            print("skip", f, e)
    if not dfs:
        print("nothing to merge for", pattern)
        return
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(out, index=False)
    print(f"{out}: {len(df)} rows from {len(dfs)} slices")
    print(df.groupby("method")["pcc"].agg(["mean", "count"]).round(3).to_string())

if __name__ == "__main__":
    which = sys_arg = os.environ.get("MERGE_WHAT", "all")
    if which in ("sim", "all"):
        merge(f"{RES}/parts/sim_*.csv", f"{RES}/sim_full_v3.csv")
    if which in ("mice", "all"):
        merge(f"{RES}/parts/mice_*.csv", f"{RES}/mice_v3.csv")
    if which in ("wheat", "all"):
        merge(f"{RES}/parts/wheat_*.csv", f"{RES}/wheat_v3.csv")
