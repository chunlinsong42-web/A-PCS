# -*- coding: utf-8 -*-
"""生成 data/sample 示例数据：200 样本 × 2000 标记，2 条染色体、各约 1 Mb。

输出 matrix.csv（行=样本、列=标记、0/1/2）、phenotype.csv（sample_id,phenotype）、
positions.csv（marker,chrom,pos）。

模拟方式：按"单倍型行走"生成基因型——沿染色体扫描标记，遇到重组就重新抽样等位基因，
否则继承前一位点，因此重组率低的区段会形成长 LD 块、热点区段 LD 迅速衰减，
得到强/弱 LD 交替的景观。表型由若干因果位点的加性效应 + 环境噪声组成。

用法：
    python make_sample_data.py            # 写到 ../data/sample/
    python make_sample_data.py --seed 7   # 换随机种子
"""
import argparse
import csv
import os

import numpy as np

N_SAMPLES = 200
MARKERS_PER_CHROM = 1000
CHROM_LEN = 1_000_000

# 每条染色体的重组率分段：低重组=强 LD 区，高重组=弱 LD 区
SEGMENTS = [(350, "S"), (350, "W"), (300, "S")]
RATE_STRONG = 0.0010
RATE_WEAK = 0.0800

NA_WORDS = {"", "na", "nan"}


def _recomb_profile(rng, m):
    rate = np.empty(m, dtype=np.float64)
    start = 0
    for size, kind in SEGMENTS:
        rate[start:start + size] = RATE_STRONG if kind == "S" else RATE_WEAK
        start += size
    if start < m:
        rate[start:] = RATE_WEAK
    return rate


def _positions(rng, m, chrom_len):
    gaps = rng.uniform(0.5, 1.5, size=m)
    p = np.cumsum(gaps)
    p = p / p[-1] * (chrom_len - 1)
    p = np.round(p).astype(np.int64) + 1
    p[0] = 1
    # 保证严格递增（四舍五入可能撞点）
    for i in range(1, m):
        if p[i] <= p[i - 1]:
            p[i] = p[i - 1] + 1
    return p


def _haplotypes(rng, n, m, recomb):
    af = rng.uniform(0.10, 0.50, size=m)
    hap = np.zeros((n, m), dtype=np.int8)
    cur = (rng.random(n) < af[0]).astype(np.int8)
    hap[:, 0] = cur
    for k in range(1, m):
        switch = rng.random(n) < recomb[k]
        fresh = (rng.random(n) < af[k]).astype(np.int8)
        cur = np.where(switch, fresh, cur)
        hap[:, k] = cur
    return hap


def simulate(seed):
    rng = np.random.default_rng(seed)
    n = N_SAMPLES
    m_total = MARKERS_PER_CHROM * 2
    X = np.zeros((n, m_total), dtype=np.int8)
    pos = np.zeros(m_total, dtype=np.int64)
    chrom = np.empty(m_total, dtype=object)

    for ci in (1, 2):
        off = (ci - 1) * MARKERS_PER_CHROM
        m = MARKERS_PER_CHROM
        p = _positions(rng, m, CHROM_LEN)
        recomb = _recomb_profile(rng, m)
        h1 = _haplotypes(rng, n, m, recomb)
        h2 = _haplotypes(rng, n, m, recomb)
        X[:, off:off + m] = h1 + h2
        pos[off:off + m] = p
        chrom[off:off + m] = ci

    # 表型：30 个因果位点，加性效应 + 噪声（遗传力中等）
    causal = rng.choice(m_total, size=30, replace=False)
    beta = rng.normal(0.0, 1.0, size=len(causal))
    beta = beta / np.sqrt(np.sum(beta ** 2)) * 1.7
    g = X[:, causal].astype(np.float64) @ beta
    noise = rng.normal(0.0, 1.0, size=n)
    y = g + noise * (np.std(g) / 0.85)
    y = y - y.mean()

    Xf = X.astype(np.float64)
    # 少量缺失（约 0.2%）
    n_miss = int(0.002 * n * m_total)
    Xf[rng.integers(0, n, n_miss), rng.integers(0, m_total, n_miss)] = np.nan
    # 3 个单态位点 + 3 个低频位点，用于演示质控
    mono = rng.choice(m_total, size=3, replace=False)
    Xf[:, mono] = 0.0
    low = rng.choice(m_total, size=3, replace=False)
    Xf[:, low] = 0.0
    Xf[:5, low] = 1.0

    markers = ["m%04d" % (i + 1) for i in range(m_total)]
    samples = ["S%03d" % (i + 1) for i in range(n)]
    return Xf, pos, chrom, y, markers, samples


def _fmt(v):
    return "" if not np.isfinite(v) else "%d" % int(round(v))


def write_csvs(out_dir, Xf, pos, chrom, y, markers, samples):
    os.makedirs(out_dir, exist_ok=True)
    geno = os.path.join(out_dir, "matrix.csv")
    pheno = os.path.join(out_dir, "phenotype.csv")
    posit = os.path.join(out_dir, "positions.csv")

    with open(geno, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id"] + markers)
        for i, sid in enumerate(samples):
            w.writerow([sid] + [_fmt(v) for v in Xf[i]])

    with open(pheno, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "phenotype"])
        for sid, val in zip(samples, y):
            w.writerow([sid, "%.6f" % float(val)])

    with open(posit, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["marker", "chrom", "pos"])
        for name, c, p in zip(markers, chrom, pos):
            w.writerow([name, str(c), int(p)])
    return geno, pheno, posit


def main():
    ap = argparse.ArgumentParser(description="生成 A-PCS 示例数据")
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample"))
    args = ap.parse_args()

    Xf, pos, chrom, y, markers, samples = simulate(args.seed)
    paths = write_csvs(args.out, Xf, pos, chrom, y, markers, samples)
    total = sum(os.path.getsize(p) for p in paths)
    for p in paths:
        print("%-14s %8.1f KB" % (os.path.basename(p), os.path.getsize(p) / 1024.0))
    print("合计 %.2f MB，%d 样本 × %d 标记，种子 %d" % (total / 1048576.0, Xf.shape[0], Xf.shape[1], args.seed))


if __name__ == "__main__":
    main()
