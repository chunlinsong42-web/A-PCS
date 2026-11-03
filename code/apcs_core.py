# -*- coding: utf-8 -*-
"""
A-PCS (Adaptive Pearson Collinearity Screening) core library.

Implements:
  - QC & standardization
  - sliding-window LD-decay landscape estimation (d_half -> L(pos))
  - A-PCS adaptive screening (position-adaptive t_sig / t_col / dedup span)
  - Baselines: PCS-fixed (global static thresholds), LD-prune, all-markers, random subset
  - GBLUP (kernel ridge) with lambda selection on an inner validation split
"""
import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import t as _t_dist


# ---------------------------------------------------------------- QC / scaling

def qc_markers(X, maf_min=0.05, miss_max=0.2):
    """MAF + missingness QC; impute NaN with rounded column mean (mode proxy for 0/1/2)."""
    X = np.asarray(X, dtype=np.float64)
    miss = np.isnan(X).mean(axis=0)
    with np.errstate(all="ignore"):
        colmean = np.nanmean(X, axis=0)
    finite = np.isfinite(colmean)
    colmean = np.where(finite, colmean, 0.0)
    maf = np.minimum(colmean, 2.0 - colmean) / 2.0
    keep = (miss <= miss_max) & (maf >= maf_min) & finite
    Xq = X[:, keep].copy()
    fill = np.round(colmean[keep])
    nan_r, nan_c = np.where(np.isnan(Xq))
    if len(nan_r):
        Xq[nan_r, nan_c] = fill[nan_c]
    return Xq, np.where(keep)[0]


def standardize(X, mu=None, sd=None):
    X = np.asarray(X, dtype=np.float32)
    if mu is None:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (X - mu) / sd
    return np.ascontiguousarray(Z, dtype=np.float32), mu, sd


def pcc(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ------------------------------------------------------- windows & LD landscape

def build_windows(pos, chrom, W, min_markers=10):
    """Consecutive windows per chromosome: width <= W and count >= min_markers."""
    wins = []
    for c in np.unique(chrom):
        idx = np.where(chrom == c)[0]
        p = pos[idx]
        start = 0
        while start < len(idx):
            end = start + 1
            while end < len(idx):
                if (p[end] - p[start] > W) and (end - start >= min_markers):
                    break
                end += 1
            wins.append((c, idx[start], idx[end - 1] + 1))  # global marker range [i0, i1)
            start = end
    return wins  # list of (chrom, i0, i1)


def window_dhalf(Z, pos, wins, n_bins=6, min_pairs=5):
    """Per-window LD half-decay distance: first distance where mean r^2 < r2(d_min)/2."""
    dh = np.full(len(wins), np.nan, dtype=np.float64)
    for wi, (c, i0, i1) in enumerate(wins):
        idx = np.arange(i0, i1)
        k = len(idx)
        if k < 6:
            continue
        Zs = np.ascontiguousarray(Z[:, idx])
        R = (Zs.T @ Zs) / Zs.shape[0]
        iu = np.triu_indices(k, 1)
        d = np.abs(pos[idx][iu[1]] - pos[idx][iu[0]])
        r2 = R[iu] ** 2
        if len(d) == 0:
            continue
        nb = max(3, min(n_bins, k // 3))
        edges = np.linspace(0.0, d.max() + 1e-9, nb + 1)
        prof = np.full(nb, np.nan)
        for b in range(nb):
            m = (d >= edges[b]) & (d < edges[b + 1])
            if m.sum() >= min_pairs:
                prof[b] = r2[m].mean()
        base = prof[0] if np.isfinite(prof[0]) else np.nanmax(prof)
        if not np.isfinite(base) or base <= 1e-6:
            dh[wi] = edges[-1]
            continue
        below = np.where(prof < base / 2.0)[0]
        if len(below) == 0:
            dh[wi] = edges[-1]
        elif below[0] == 0:
            dh[wi] = edges[1] * (base / 2.0) / base
        else:
            b = below[0]
            p0, p1 = prof[b - 1], prof[b]
            f = (p0 - base / 2.0) / (p0 - p1) if p0 != p1 else 0.5
            dh[wi] = edges[b - 1] + f * (edges[b] - edges[b - 1])
    return dh


def _smooth_by_chrom(vals, chr_of, k):
    out = vals.copy()
    for c in np.unique(chr_of):
        m = chr_of == c
        out[m] = median_filter(vals[m], size=k, mode="nearest")
    return out


def _interp_by_chrom(centers, vals, chr_of_c, pos, chrom):
    out = np.empty(len(pos), dtype=np.float64)
    for c in np.unique(chrom):
        m = chrom == c
        mc = chr_of_c == c
        if mc.sum() == 0:
            out[m] = vals[mc].mean() if mc.any() else 0.0
            continue
        out[m] = np.interp(pos[m], centers[mc], vals[mc])
    return out


def ld_precompute(Z, pos, chrom, W, min_markers=10, smooth_k=5, w_max_mult=8):
    """Estimate the LD-decay landscape.

    Returns (L_marker, dh_marker): L in [0,1] (0=recombination hotspot, 1=strong-LD block),
    dh_marker = smoothed per-marker half-decay distance (capped at hi percentile).
    """
    wins = build_windows(pos, chrom, W, min_markers)
    dh_w = window_dhalf(Z, pos, wins)
    Wcur = W
    for _ in range(6):
        finite = np.isfinite(dh_w)
        if finite.sum() >= 5 and np.mean(dh_w[finite] >= Wcur * 0.98) <= 0.5:
            break
        Wcur *= 2
        if Wcur > W * w_max_mult:
            break
        wins = build_windows(pos, chrom, Wcur, min_markers)
        dh_w = window_dhalf(Z, pos, wins)
    centers = np.array([0.5 * (pos[i0] + pos[i1 - 1]) for (_, i0, i1) in wins], dtype=np.float64)
    chr_of_w = np.array([c for (c, _, _) in wins])
    dhf = dh_w[np.isfinite(dh_w)]
    if len(dhf) < 5:
        L = np.full(len(pos), 0.5)
        return L, np.full(len(pos), float(W)), centers, chr_of_w, dh_w, Wcur
    lo, hi = np.percentile(dhf, [10, 90])
    if not np.isfinite(lo) or lo <= 0:
        lo = max(1e-9, float(np.min(dhf[dhf > 0])) if np.any(dhf > 0) else 1e-9)
    if not np.isfinite(hi) or hi <= lo * 1.2:
        Lc = np.full(len(centers), 0.5)
    else:
        Lc = np.log(np.clip(dh_w, lo, hi) / lo) / np.log(hi / lo)
        Lc = np.nan_to_num(Lc, nan=0.5)
        Lc = np.clip(Lc, 0.0, 1.0)
    Lc_s = _smooth_by_chrom(Lc, chr_of_w, smooth_k)
    dh_s = _smooth_by_chrom(np.clip(np.nan_to_num(dh_w, nan=hi), lo, hi), chr_of_w, smooth_k)
    L = np.clip(_interp_by_chrom(centers, Lc_s, chr_of_w, pos, chrom), 0.0, 1.0)
    dh_m = np.clip(_interp_by_chrom(centers, dh_s, chr_of_w, pos, chrom), lo, hi)
    return L, dh_m, centers, chr_of_w, dh_w, Wcur


# ------------------------------------------------------------- screening ops

def _corr_y(Z, y):
    return np.abs(Z.T @ y) / len(y)


def corr_abs(Z, y):
    """Public: |Pearson r| of every standardized column with standardized y."""
    return np.abs(Z.T @ y) / len(y)


def tune_lambda_val(Zfit, yfit, Zval, yval, keep_f, lam_grid, w_f=None):
    """Lambda selection on a validation split; supports marker weights."""
    Zk = Zfit[:, keep_f]
    d, U, ym, m_eff, Kw = gblup_eigh(Zk, yfit, w=w_f)
    Zvk = Zval[:, keep_f]
    Zvk_w = Zvk * np.sqrt(w_f)[None, :] if w_f is not None else Zvk
    coef = U.T @ (yfit - ym)
    ZvT = (Zvk_w @ Kw.T) / m_eff
    best_lam, best_pv = lam_grid[0], -2.0
    for lam in lam_grid:
        yhat = ZvT @ (U @ (coef / (d + lam)))
        v = pcc(yhat, yval - np.mean(yval))
        if v > best_pv:
            best_pv, best_lam = v, lam
    return best_lam, best_pv


def fit_predict_gblup(Ztr, ytr, Zte, keep, lam, w_kept=None):
    """Fit GBLUP on selected (optionally weighted) markers and predict test."""
    Zk = Ztr[:, keep]
    d, U, ym, m_eff, Kw = gblup_eigh(Zk, ytr, w=w_kept)
    Ztk = Zte[:, keep]
    Ztk_w = Ztk * np.sqrt(w_kept)[None, :] if w_kept is not None else Ztk
    return gblup_predict_from_eigh(d, U, Zk, ytr - ym, Ztk, lam, ym,
                                   m_eff=m_eff, Ztr_w=Kw, Zte_w=Ztk_w)


def method_weights(method, cfg, Z, y, keep):
    """|PCC|^gamma weights for weighted variants; None otherwise."""
    if cfg.get("gamma"):
        rmy = corr_abs(Z, y)
        w = np.where(keep, np.power(np.maximum(rmy, 1e-6), cfg["gamma"]), 0.0)
        return w[keep]
    return None


def _greedy_dedup(Z, pos, chrom, cand, t_col_of, dnb_of):
    """Greedy: walk candidates (sorted by |r_my| desc), keep if no already-kept
    neighbour within dnb_of[i] has |r| >= t_col_of[i]."""
    n, M = Z.shape
    kept = np.zeros(M, dtype=bool)
    kp, ki = {}, {}
    Zcol_cache = Z
    for i in cand:
        c = chrom[i]
        arr_p = kp.get(c)
        ok = True
        if arr_p:
            P = np.asarray(arr_p)
            nb = np.abs(P - pos[i]) <= dnb_of[i]
            if nb.any():
                idxs = np.asarray(ki[c])[nb]
                rr = (Zcol_cache[:, idxs].T @ Zcol_cache[:, i]) / n
                if np.abs(rr).max() >= t_col_of[i]:
                    ok = False
        if ok:
            kept[i] = True
            kp.setdefault(c, []).append(pos[i])
            ki.setdefault(c, []).append(i)
    return kept


def _admitted_order(rmy, t_sig_of):
    adm = np.where(rmy >= t_sig_of)[0]
    return adm[np.argsort(-rmy[adm])]


def pcs_fixed(Z, y, pos, chrom, t_sig, t_col, W):
    """Classic PCS: global fixed admission threshold + fixed collinearity threshold."""
    rmy = _corr_y(Z, y)
    M = len(rmy)
    t_sig_of = np.full(M, float(t_sig))
    t_col_of = np.full(M, float(t_col))
    dnb_of = np.full(M, float(W))
    cand = _admitted_order(rmy, t_sig_of)
    kept = _greedy_dedup(Z, pos, chrom, cand, t_col_of, dnb_of)
    return kept, rmy


def apcs(Z, y, pos, chrom, W, t_sig_min, t_sig_max, t_col_min, t_col_max, pre):
    """A-PCS: LD-landscape-adaptive thresholds.

    t_sig(pos) = t_sig_min + (t_sig_max - t_sig_min) * L     (strong LD -> strict admission)
    t_col(pos) = t_col_max - (t_col_max - t_col_min) * L     (strong LD -> aggressive dedup)
    dedup span d_nb(pos) = max(2 * d_half(pos), W)
    pre = (L, dh_marker) from ld_precompute.
    """
    L, dh_m = pre[0], pre[1]
    rmy = _corr_y(Z, y)
    t_sig_of = t_sig_min + (t_sig_max - t_sig_min) * L
    t_col_of = t_col_max - (t_col_max - t_col_min) * L
    dnb_of = np.maximum(2.0 * dh_m, float(W))
    cand = _admitted_order(rmy, t_sig_of)
    kept = _greedy_dedup(Z, pos, chrom, cand, t_col_of, dnb_of)
    return kept, rmy


def apcs_fix_sig(Z, y, pos, chrom, W, t_sig, t_col_min, t_col_max, pre):
    """Ablation: fixed admission threshold, adaptive collinearity threshold."""
    L, dh_m = pre[0], pre[1]
    rmy = _corr_y(Z, y)
    t_sig_of = np.full(len(rmy), float(t_sig))
    t_col_of = t_col_max - (t_col_max - t_col_min) * L
    dnb_of = np.maximum(2.0 * dh_m, float(W))
    cand = _admitted_order(rmy, t_sig_of)
    return _greedy_dedup(Z, pos, chrom, cand, t_col_of, dnb_of), rmy


def apcs_fix_col(Z, y, pos, chrom, W, t_sig_min, t_sig_max, t_col, pre):
    """Ablation: adaptive admission threshold, fixed collinearity threshold."""
    L, dh_m = pre[0], pre[1]
    rmy = _corr_y(Z, y)
    t_sig_of = t_sig_min + (t_sig_max - t_sig_min) * L
    t_col_of = np.full(len(rmy), float(t_col))
    dnb_of = np.maximum(2.0 * dh_m, float(W))
    cand = _admitted_order(rmy, t_sig_of)
    return _greedy_dedup(Z, pos, chrom, cand, t_col_of, dnb_of), rmy


# ------------------------------------------------------------------ A-PCS v2
# Scale-free admission: nominal per-marker significance level q in p-value space,
# tightened in strong-LD regions by factor (1 - b*L).  This decouples the admission
# rule from sample size / marker count / trait scale -> transferable across datasets.

def _corr_pvalues(rmy_abs, n):
    """Two-sided t-test p-values for Pearson correlations (vectorized)."""
    from scipy.stats import t as t_dist
    r = np.clip(rmy_abs, 0.0, 1.0 - 1e-12)
    tstat = r * np.sqrt((n - 2) / (1 - r * r))
    return 2.0 * t_dist.sf(tstat, df=n - 2)


def apcs2(Z, y, pos, chrom, W, q, b, t_col_min, t_col_max, pre):
    """A-PCS v2: p-space adaptive admission + LD-adaptive collinearity dedup."""
    L, dh_m = pre[0], pre[1]
    rmy = _corr_y(Z, y)
    n = len(y)
    pv = _corr_pvalues(rmy, n)
    q_eff = q * (1.0 - b * L)
    cand_mask = pv <= q_eff
    cand = np.where(cand_mask)[0]
    cand = cand[np.argsort(-rmy[cand])]
    t_col_of = t_col_max - (t_col_max - t_col_min) * L
    dnb_of = np.maximum(2.0 * dh_m, float(W))
    kept = _greedy_dedup(Z, pos, chrom, cand, t_col_of, dnb_of)
    return kept, rmy


def apcs2_fix_sig(Z, y, pos, chrom, W, q, t_col_min, t_col_max, pre):
    """Ablation of v2: no LD modulation of the admission level (b=0)."""
    L, dh_m = pre[0], pre[1]
    rmy = _corr_y(Z, y)
    n = len(y)
    pv = _corr_pvalues(rmy, n)
    cand = np.where(pv <= q)[0]
    cand = cand[np.argsort(-rmy[cand])]
    t_col_of = t_col_max - (t_col_max - t_col_min) * L
    dnb_of = np.maximum(2.0 * dh_m, float(W))
    kept = _greedy_dedup(Z, pos, chrom, cand, t_col_of, dnb_of)
    return kept, rmy


def apcs2_fix_col(Z, y, pos, chrom, W, q, b, t_col, pre):
    """Ablation of v2: fixed collinearity threshold."""
    L, dh_m = pre[0], pre[1]
    rmy = _corr_y(Z, y)
    n = len(y)
    pv = _corr_pvalues(rmy, n)
    q_eff = q * (1.0 - b * L)
    cand = np.where(pv <= q_eff)[0]
    cand = cand[np.argsort(-rmy[cand])]
    t_col_of = np.full(len(rmy), float(t_col))
    dnb_of = np.maximum(2.0 * dh_m, float(W))
    kept = _greedy_dedup(Z, pos, chrom, cand, t_col_of, dnb_of)
    return kept, rmy


def ld_prune(Z, pos, chrom, W, r2_max=0.8):
    """Classic LD pruning (no phenotype info): left-to-right, drop marker if an
    already-kept marker within W has |r| >= r2_max^(1/2)."""
    n, M = Z.shape
    order = np.argsort(pos, kind="stable")
    t_col_of = np.full(M, float(np.sqrt(r2_max)))
    dnb_of = np.full(M, float(W))
    kept = _greedy_dedup(Z, pos, chrom, order, t_col_of, dnb_of)
    return kept


# --------------------------------------------------------------------- GBLUP

def gblup_eigh(Ztr, ytr, w=None):
    """Eigendecompose the (optionally marker-weighted) genomic relationship matrix.
    w: per-marker weights; K = (Z*sqrt(w))(Z*sqrt(w))' / sum(w)  -> diag(K)=1."""
    if w is None:
        m_eff = Ztr.shape[1]
        Kw = Ztr
    else:
        w = np.asarray(w, dtype=np.float64)
        m_eff = float(w.sum())
        Kw = Ztr * np.sqrt(w)[None, :]
    K = (Kw @ Kw.T) / m_eff
    K64 = np.asarray(K, dtype=np.float64)
    K64 = 0.5 * (K64 + K64.T)
    d, U = np.linalg.eigh(K64)
    return d, U, float(ytr.mean()), m_eff, Kw


def gblup_predict_from_eigh(d, U, Ztr, ytr, Zte, lam, y_mean=0.0, m_eff=None, Ztr_w=None, Zte_w=None):
    if m_eff is None:
        m_eff = Ztr.shape[1]
    Sw = (Ztr_w if Ztr_w is not None else Ztr).T
    Tw = (Zte_w if Zte_w is not None else Zte)
    coef = U.T @ ytr
    b = U @ (coef / (d + lam))
    return (Tw @ (Sw @ b)) / m_eff + y_mean


def tune_lambda_on_val(Zfit, yfit, Zval, yval, lams, w_fit=None, w_val=None):
    """Pick lambda by validation PCC using one eigendecomposition."""
    d, U, ym, m_eff, Zfit_w = gblup_eigh(Zfit, ytr=yfit, w=w_fit)
    Zval_w = Zval * np.sqrt(w_val)[None, :] if w_val is not None else Zval
    yfit_c = yfit - ym
    coef = U.T @ yfit_c
    ZvT = (Zval_w @ Zfit_w.T) / m_eff
    best_lam, best_pcc = lams[0], -2.0
    for lam in lams:
        yhat = ZvT @ (U @ (coef / (d + lam)))
        v = pcc(yhat, yval - np.mean(yval))
        if v > best_pcc:
            best_pcc, best_lam = v, lam
    return best_lam, best_pcc
