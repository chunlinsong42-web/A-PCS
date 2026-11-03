# -*- coding: utf-8 -*-
"""Prepare real datasets (BGLR mice & wheat) into npz for fast reuse."""
import numpy as np
import pandas as pd
import pyreadr

np.random.seed(0)

# ------------------------------------------------------------------------ mice
r = pyreadr.read_r("../data/mice.RData")
X = r["mice.X"].to_numpy(dtype=np.float32)
mp = r["mice.map"]
ph = r["mice.pheno"]
print("mice.X", X.shape, "values:", np.unique(X[:200])[:6], "NaN frac:", np.isnan(X).mean())
print("chr values:", sorted(mp["chr"].astype(str).unique()))

chr_str = mp["chr"].astype(str).to_numpy()
mbp = pd.to_numeric(mp["mbp"], errors="coerce").to_numpy()
chr_num = np.full(len(chr_str), -1)
for i, c in enumerate(chr_str):
    try:
        chr_num[i] = int(c)
    except ValueError:
        chr_num[i] = {"X": 20, "Y": 21, "MT": 22, "M": 22}.get(c.upper(), -1)
valid = (chr_num > 0) & np.isfinite(mbp)
Xv, chr_v, pos_v = X[:, valid], chr_num[valid], mbp[valid] * 1e6
# order markers by (chr, pos)
order = np.lexsort((pos_v, chr_v))
Xv, chr_v, pos_v = Xv[:, order], chr_v[order], pos_v[order]
print("mice valid markers:", Xv.shape)

# pick continuous traits with acceptable missingness; drop admin/date columns
EXCLUDE = ("Date", "Month", "Year", "Day", "Season", "Seconds", "Age",
           "Litter", "Cage", "Subject", "CoatColour", "GENDER")
traits = {}
for col in ph.columns:
    if any(k.lower() in col.lower() for k in EXCLUDE):
        continue
    v = pd.to_numeric(ph[col], errors="coerce").to_numpy(dtype=np.float64)
    if np.isfinite(v).sum() < 0.8 * len(v) or np.isfinite(v).sum() < 1000:
        continue
    if np.nanstd(v) < 1e-9:
        continue
    traits[col] = v
print("candidate traits:", {k: int(np.isfinite(v).sum()) for k, v in traits.items()})

np.savez_compressed("../data/mice.npz", X=Xv, chrom=chr_v, pos=pos_v,
                    trait_names=np.array(list(traits.keys())),
                    Y=np.stack([np.nan_to_num(v, nan=np.nan) for v in traits.values()]))
print("saved mice.npz")

# ----------------------------------------------------------------------- wheat
r = pyreadr.read_r("../data/wheat.RData")
Xw = r["wheat.X"].to_numpy(dtype=np.float32)
Yw = r["wheat.Y"].to_numpy(dtype=np.float64)
print("wheat.X", Xw.shape, "NaN:", np.isnan(Xw).mean(), "Y", Yw.shape,
      "NaN per env:", np.isnan(Yw).mean(axis=0))
np.savez_compressed("../data/wheat.npz", X=Xw, Y=Yw)
print("saved wheat.npz")
