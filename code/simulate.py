# -*- coding: utf-8 -*-
"""
Crop-like genome simulator with heterogeneous LD landscape.

Genome: n_chr chromosomes, uniform marker spacing.
LD regimes per segment: 'hotspot' (fast decay), 'moderate', 'block' (slow decay).
Latent AR(1) chain per segment -> direct 3-level discretization to 0/1/2 dosage
with per-marker MAF q (probs (1-q)^2, 2q(1-q), q^2).
Subpopulations: allele-frequency shifts (population structure).
Traits: architectures with known causal markers (ground truth for recall analysis).
"""
import numpy as np
from scipy.stats import norm

REGIMES = ("hotspot", "moderate", "block")


def simulate_genome(n_ind=1600, n_chr=12, chr_len_mb=30.0, spacing_kb=6.0,
                    maf_range=(0.10, 0.5), seed=2026,
                    seg_frac=(0.35, 0.45, 0.20),
                    decay_kb={"hotspot": 8.0, "moderate": 25.0, "block": 600.0},
                    seg_len_kb={"hotspot": (20, 60), "moderate": (80, 300), "block": (400, 1200)},
                    subpops=((0.40, -0.35), (0.35, 0.0), (0.25, 0.35))):
    rng = np.random.default_rng(seed)
    spacing = spacing_kb * 1e3
    chr_len = chr_len_mb * 1e6
    pos_list, chrom_list, regime_list = [], [], []
    for c in range(1, n_chr + 1):
        p = 0.0
        while p < chr_len:
            r = int(rng.choice(3, p=seg_frac))
            lo, hi = seg_len_kb[REGIMES[r]]
            L = rng.uniform(lo, hi) * 1e3
            p2 = min(p + L, chr_len)
            n_mk = int((p2 - p) // spacing)
            if n_mk >= 3:
                pos_list.append(p + spacing * np.arange(1, n_mk + 1))
                chrom_list.append(np.full(n_mk, c))
                regime_list.append(np.full(n_mk, REGIMES[r]))
            p = p2
    pos = np.concatenate(pos_list)
    chrom = np.concatenate(chrom_list)
    regime = np.concatenate(regime_list)
    order = np.lexsort((pos, chrom))
    pos, chrom, regime = pos[order], chrom[order], regime[order]
    M = len(pos)

    q = rng.uniform(maf_range[0], maf_range[1], M)
    rho_mk = np.exp(-spacing / (np.array([decay_kb[r] for r in regime]) * 1e3))
    # discretization edges per marker for dosage 0/1/2 under latent N(0,1)
    e1 = norm.ppf((1.0 - q) ** 2)
    e2 = norm.ppf((1.0 - q) ** 2 + 2.0 * q * (1.0 - q))

    probs = np.array([s[0] for s in subpops]); probs = probs / probs.sum()
    sub = rng.choice(len(subpops), size=n_ind, p=probs)
    delta = np.array([s[1] for s in subpops])[sub]

    X = np.zeros((n_ind, M), dtype=np.float32)
    for c in np.unique(chrom):
        idx = np.where(chrom == c)[0]
        z = rng.normal(size=n_ind) + delta
        X[:, idx[0]] = ((z >= e1[idx[0]]).astype(np.float32)
                        + (z >= e2[idx[0]]).astype(np.float32))
        for a, b in zip(idx[:-1], idx[1:]):
            if regime[b] != regime[a]:  # segment boundary: recombination reset
                z = rng.normal(size=n_ind) + delta
            else:
                rho = rho_mk[b]
                z = rho * z + np.sqrt(max(1e-12, 1.0 - rho * rho)) * rng.normal(size=n_ind) \
                    + delta * (1.0 - rho)
            X[:, b] = (z >= e1[b]).astype(np.float32) + (z >= e2[b]).astype(np.float32)
    meta = dict(pos=pos, chrom=chrom, regime=regime, subpop=sub, maf=q,
                spacing=spacing, n_chr=n_chr)
    return X, meta


def _sample_causal(rng, regime, M, n_target, regime_filter=None, exclude=None,
                   existing=None, min_gap=20):
    existing = list(existing or [])
    out = []
    tries = 0
    while len(out) < n_target and tries < n_target * 800:
        tries += 1
        i = int(rng.integers(0, M))
        if regime_filter is not None and regime[i] != regime_filter:
            continue
        if exclude is not None and regime[i] == exclude:
            continue
        if existing and np.min(np.abs(np.asarray(existing + out) - i)) < min_gap:
            continue
        out.append(i)
    return out


def simulate_traits(X, meta, seed=7, specs=None):
    """Return dict trait_name -> dict(y, causal, h2, epistatic_pairs)."""
    rng = np.random.default_rng(seed)
    n, M = X.shape
    regime = meta["regime"]
    if specs is None:
        specs = [
            dict(name="T1_block_oligo", n_major=30, n_minor=200, major_regime="block",
                 h2=0.60, epistatic=0),
            dict(name="T2_polygenic", n_major=0, n_minor=500, major_regime=None,
                 hotspot_bias=0.60, h2=0.50, epistatic=0),
            dict(name="T3_mixed_epi", n_major=50, n_minor=120, major_regime=None,
                 h2=0.55, epistatic=12),
        ]
    out = {}
    for sp in specs:
        causal = []
        if sp.get("major_regime"):
            causal += _sample_causal(rng, regime, M, sp["n_major"],
                                     regime_filter=sp["major_regime"])
        if sp.get("hotspot_bias"):
            n_hot = int(round(sp["n_minor"] * sp["hotspot_bias"]))
            causal += _sample_causal(rng, regime, M, n_hot, regime_filter="hotspot",
                                     existing=causal)
            causal += _sample_causal(rng, regime, M, sp["n_minor"] - n_hot,
                                     exclude="hotspot", existing=causal)
        else:
            causal += _sample_causal(rng, regime, M, sp["n_minor"], existing=causal)
        causal = np.array(sorted(causal))
        eff = rng.normal(size=len(causal))
        if sp.get("n_major", 0) > 0:
            eff[:sp["n_major"]] *= rng.uniform(2.0, 4.0, sp["n_major"])
        Xc = X[:, causal].astype(np.float64)
        Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-12)
        g = Xc @ eff
        epi_pairs = []
        for _ in range(sp.get("epistatic", 0)):
            a, b = rng.choice(causal, 2, replace=False)
            za = (X[:, a] - X[:, a].mean()) / (X[:, a].std() + 1e-12)
            zb = (X[:, b] - X[:, b].mean()) / (X[:, b].std() + 1e-12)
            g += rng.normal() * za * zb
            epi_pairs.append((int(a), int(b)))
        g = (g - g.mean()) / (g.std() + 1e-12)
        h2 = sp["h2"]
        y = np.sqrt(h2) * g + np.sqrt(1 - h2) * rng.normal(size=n)
        out[sp["name"]] = dict(y=y, causal=causal, h2=h2, epistatic_pairs=epi_pairs)
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    X, meta = simulate_genome(n_ind=400, seed=1)
    print("genome", X.shape, "in %.1fs" % (time.time() - t0))
    from apcs_core import standardize
    Z, _, _ = standardize(X)
    sp_pos = meta["spacing"]
    for reg in REGIMES:
        idx = np.where(meta["regime"] == reg)[0][:400]
        Zs = Z[:, idx]
        R = (Zs.T @ Zs) / Zs.shape[0]
        d = np.abs(meta["pos"][idx][None, :] - meta["pos"][idx][:, None])
        iu = np.triu_indices(len(idx), 1)
        near = (d[iu] > 0.5 * sp_pos) & (d[iu] <= 2 * sp_pos)
        print(reg, "mean r2 (1-2 spacing):", round(float((R[iu][near] ** 2).mean()), 3))
    # subpop MAF shift check
    for s in np.unique(meta["subpop"]):
        m = meta["subpop"] == s
        print("subpop", s, "n", int(m.sum()), "mean dosage:", round(float(X[m].mean()), 3))
    tr = simulate_traits(X, meta)
    for k, v in tr.items():
        regs = meta["regime"][v["causal"]]
        print(k, "causal", len(v["causal"]),
              "hotspot:", int((regs == "hotspot").sum()),
              "moderate:", int((regs == "moderate").sum()),
              "block:", int((regs == "block").sum()),
              "y sd:", round(float(v["y"].std()), 3))
