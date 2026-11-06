# -*- coding: utf-8 -*-
"""最小可复现示例：读取 data/sample/ 的三份文件，跑一遍 A-PCS 主流程。

流程：qc_markers → standardize → ld_precompute(W=100000) → apcs2(q=0.1, b=0.7,
t_col_min=0.5, t_col_max=0.9)，结果写到 docs/example/。

用法：
    python code/example_run.py
    python code/example_run.py --data data/sample --out docs/example
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import apcs_core as core                     # noqa: E402
import plot_landscape                        # noqa: E402

DEFAULTS = dict(W=100000.0, q=0.1, b=0.7, t_col_min=0.5, t_col_max=0.9,
                maf_min=0.05, miss_max=0.2)

NA_WORDS = {"", "na", "nan", "n/a", "null", "none", ".", "?"}


class DataError(Exception):
    pass


def _read_rows(path):
    if not os.path.isfile(path):
        raise DataError("找不到文件：%s" % path)
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        first = fh.readline()
        fh.seek(0)
        delim = "\t" if first.count("\t") > first.count(",") else ","
        for row in csv.reader(fh, delimiter=delim):
            if row and any(c.strip() for c in row):
                rows.append(row)
    if not rows:
        raise DataError("文件为空：%s" % path)
    return rows


def load_genotype(path):
    rows = _read_rows(path)
    if len(rows) < 2 or len(rows[0]) < 2:
        raise DataError("基因型文件需要表头 + 样本行，且至少 1 个标记列")
    markers = [h.strip() for h in rows[0][1:]]
    if len(set(markers)) != len(markers):
        raise DataError("基因型文件存在重复标记名")
    n, m = len(rows) - 1, len(markers)
    samples = []
    X = np.empty((n, m), dtype=np.float64)
    for i, row in enumerate(rows[1:]):
        if len(row) - 1 < m:
            raise DataError("第 %d 行标记数不足（%d < %d）" % (i + 2, len(row) - 1, m))
        sid = row[0].strip()
        if not sid:
            raise DataError("第 %d 行缺少样本编号" % (i + 2))
        samples.append(sid)
        cells = [v if v else "nan" for v in row[1:m + 1]]  # 空串按缺失处理
        try:
            X[i] = list(map(float, cells))
        except ValueError:
            for j, cell in enumerate(cells):
                try:
                    X[i, j] = float(cell)
                except ValueError:
                    if cell.strip().lower() in NA_WORDS:
                        X[i, j] = np.nan
                    else:
                        raise DataError("基因型取值无法解析：样本 %s 标记 %s 得到 %r"
                                        % (sid, markers[j], cell))
    if len(set(samples)) != len(samples):
        raise DataError("基因型文件存在重复样本编号")
    finite = np.isfinite(X)
    bad = finite & (np.abs(X - np.round(X)) > 1e-9)
    bad |= finite & ~np.isin(np.where(finite, np.round(X), 0.0), (0.0, 1.0, 2.0))
    if np.any(bad):
        lines = ["基因型取值必须为 0/1/2："]
        for i, j in np.argwhere(bad)[:5]:
            lines.append("样本 %s 的标记 %s 值 %r" % (samples[i], markers[j], X[i, j]))
        raise DataError("\n".join(lines))
    return X, samples, markers


def load_phenotype(path):
    rows = _read_rows(path)
    start = 1 if len(rows[0]) >= 2 and not _is_float(rows[0][1]) else 0
    ids, vals = [], []
    for r in rows[start:]:
        if len(r) < 2:
            raise DataError("表型文件每行需要 sample_id,phenotype 两列")
        ids.append(r[0].strip())
        vals.append(float(r[1]))
    if len(set(ids)) != len(ids):
        raise DataError("表型文件存在重复样本编号")
    return ids, np.asarray(vals, dtype=np.float64)


def load_positions(path):
    rows = _read_rows(path)
    start = 1 if len(rows[0]) >= 3 and not _is_float(rows[0][2]) else 0
    names, chroms, poss = [], [], []
    for r in rows[start:]:
        if len(r) < 3:
            raise DataError("位置文件每行需要 marker,chrom,pos 三列")
        names.append(r[0].strip())
        chroms.append(r[1].strip())
        poss.append(int(round(float(r[2]))))
    if len(set(names)) != len(names):
        raise DataError("位置文件存在重复标记名")
    return names, np.asarray(chroms, dtype=object), np.asarray(poss, dtype=np.int64)


def _pvalues_from_r(abs_r, n):
    """由 |r| 换算双尾 t 检验 p 值（与 apcs2 内部同一口径）。"""
    from scipy.stats import t as t_dist
    r = np.clip(np.asarray(abs_r, dtype=np.float64), 0.0, 1.0 - 1e-12)
    tstat = r * np.sqrt((n - 2) / (1.0 - r * r))
    return 2.0 * t_dist.sf(tstat, df=n - 2)


def _is_float(text):
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def _chrom_key(c):
    s = str(c).strip()
    return (0, int(s)) if s.isdigit() else (1, s.lower())


def prepare(data_dir):
    geno = os.path.join(data_dir, "matrix.csv")
    pheno = os.path.join(data_dir, "phenotype.csv")
    posit = os.path.join(data_dir, "positions.csv")

    X, samples, markers = load_genotype(geno)
    ph_ids, ph_vals = load_phenotype(pheno)
    pos_names, pos_chrom, pos_val = load_positions(posit)

    if set(samples) != set(ph_ids):
        raise DataError("基因型与表型的样本不一致（基因型 %d，表型 %d）" % (len(samples), len(ph_ids)))
    if set(markers) != set(pos_names):
        raise DataError("基因型与位置文件的标记不一致（基因型 %d，位置 %d）" % (len(markers), len(pos_names)))

    ph_map = dict(zip(ph_ids, ph_vals))
    y = np.asarray([ph_map[s] for s in samples], dtype=np.float64)
    pos_map = {n: (c, p) for n, c, p in zip(pos_names, pos_chrom, pos_val)}
    chrom = np.asarray([pos_map[m][0] for m in markers], dtype=object)
    pos = np.asarray([pos_map[m][1] for m in markers], dtype=np.int64)

    order = sorted(range(len(markers)),
                   key=lambda i: (_chrom_key(chrom[i]), int(pos[i])))
    X = X[:, order]
    markers = [markers[i] for i in order]
    chrom = chrom[order]
    pos = pos[order]
    return dict(X=X, y=y, markers=markers, chrom=chrom, pos=pos,
                geno=geno, pheno=pheno, posit=posit)


def main():
    ap = argparse.ArgumentParser(description="A-PCS 最小可复现示例")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "sample"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "example"))
    ap.add_argument("--W", type=float, default=DEFAULTS["W"])
    ap.add_argument("--q", type=float, default=DEFAULTS["q"])
    ap.add_argument("--b", type=float, default=DEFAULTS["b"])
    ap.add_argument("--t-col-min", type=float, default=DEFAULTS["t_col_min"])
    ap.add_argument("--t-col-max", type=float, default=DEFAULTS["t_col_max"])
    args = ap.parse_args()

    if not (0 < args.q < 1) or not (0 <= args.b <= 1):
        raise SystemExit("参数越界：0<q<1，0<=b<=1")
    if not (0 <= args.t_col_min <= args.t_col_max <= 1):
        raise SystemExit("参数越界：0<=t_col_min<=t_col_max<=1")

    os.makedirs(args.out, exist_ok=True)
    t_start = time.perf_counter()
    d = prepare(args.data)
    X, y, markers = d["X"], d["y"], d["markers"]
    chrom, pos = d["chrom"], d["pos"]
    print("读取：%d 样本 × %d 标记（染色体 %s）"
          % (X.shape[0], X.shape[1], ", ".join(map(str, sorted(set(map(str, chrom)))))))

    t0 = time.perf_counter()
    Xq, keep_qc = core.qc_markers(X, maf_min=DEFAULTS["maf_min"], miss_max=DEFAULTS["miss_max"])
    t_qc = time.perf_counter() - t0
    n_qc = int(len(keep_qc))
    print("质控：%d → %d 个标记，用时 %.3f 秒" % (len(markers), n_qc, t_qc))

    t0 = time.perf_counter()
    Z, _, _ = core.standardize(Xq)
    yz = (y - y.mean()) / (y.std() + 1e-12)
    t_std = time.perf_counter() - t0
    print("标准化：完成，用时 %.3f 秒" % t_std)

    t0 = time.perf_counter()
    pre = core.ld_precompute(Z, pos[keep_qc], chrom[keep_qc], W=args.W)
    L, W_used = pre[0], pre[5]
    t_ld = time.perf_counter() - t0
    print("LD 景观：%d 个窗口，L ∈ [%.3f, %.3f]，最终窗口 %.0f bp，用时 %.3f 秒"
          % (len(pre[2]), float(np.min(L)), float(np.max(L)), W_used, t_ld))

    t0 = time.perf_counter()
    kept, rmy = core.apcs2(Z, yz, pos[keep_qc], chrom[keep_qc], W=args.W,
                           q=args.q, b=args.b, t_col_min=args.t_col_min,
                           t_col_max=args.t_col_max, pre=pre)
    t_sel = time.perf_counter() - t0
    n_sel = int(np.sum(kept))
    print("筛选：保留 %d / %d 个标记，用时 %.3f 秒" % (n_sel, n_qc, t_sel))

    # ---- 输出：selected_markers.csv（按 |r| 降序，附 p 值）
    sel = np.where(kept)[0]
    gidx = keep_qc[sel]
    pvals = _pvalues_from_r(rmy[sel], X.shape[0])
    rows = [(markers[i], str(chrom[i]), int(pos[i]), float(rmy[j]), float(pv))
            for j, i, pv in zip(sel, gidx, pvals)]
    rows.sort(key=lambda r: -r[3])
    csv_path = os.path.join(args.out, "selected_markers.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["marker", "chrom", "pos", "abs_r", "p_value"])
        for name, c, p, r, pv in rows:
            w.writerow([name, c, p, "%.6f" % r, "%.6g" % pv])

    # ---- 输出：ld_landscape.png（L 曲线 + 两条阈值曲线）
    png_path = os.path.join(args.out, "ld_landscape.png")
    backend = plot_landscape.plot_ld_landscape(
        png_path, pos[keep_qc], L, chrom[keep_qc], W_used,
        q=args.q, b=args.b, t_col_min=args.t_col_min, t_col_max=args.t_col_max)
    print("LD 景观图：%s（%s 渲染）" % (os.path.basename(png_path), backend))

    # ---- 输出：summary.txt
    elapsed = time.perf_counter() - t_start
    comp = (len(markers) / float(n_sel)) if n_sel else float("inf")
    summary_path = os.path.join(args.out, "summary.txt")
    lines = [
        "A-PCS 示例运行摘要",
        "生成时间：%s" % time.strftime("%Y-%m-%d %H:%M:%S"),
        "",
        "【输入】",
        "基因型：%s（%d 样本 × %d 标记）" % (os.path.relpath(d["geno"], ROOT), X.shape[0], X.shape[1]),
        "表型：%s（%d 样本）" % (os.path.relpath(d["pheno"], ROOT), len(y)),
        "位置：%s（%d 标记，%d 条染色体）" % (os.path.relpath(d["posit"], ROOT), len(markers),
                                    len(set(map(str, chrom)))),
        "",
        "【参数】",
        "W = %s bp" % format(int(args.W), ","),
        "q = %g" % args.q,
        "b = %g" % args.b,
        "t_col_min = %g，t_col_max = %g" % (args.t_col_min, args.t_col_max),
        "质控：MAF >= %.2f，缺失率 <= %.0f%%" % (DEFAULTS["maf_min"], DEFAULTS["miss_max"] * 100),
        "",
        "【结果】",
        "原始标记数：%d" % len(markers),
        "质控后标记数：%d（剔除 %d）" % (n_qc, len(markers) - n_qc),
        "保留标记数：%d" % n_sel,
        "压缩比：%s" % ("%.1f×" % comp if np.isfinite(comp) else "—"),
        "总耗时：%.2f 秒" % elapsed,
        "保留标记 |r| 范围：%.3f – %.3f，p 值最小 %.3g"
        % (min(r[3] for r in rows), max(r[3] for r in rows), min(r[4] for r in rows)),
        "",
        "【LD 景观】",
        "窗口数：%d，最终窗口宽度：%s bp" % (len(pre[2]), format(int(W_used), ",")),
        "L 范围：%.3f – %.3f" % (float(np.min(L)), float(np.max(L))),
        "",
        "【输出文件】",
        "selected_markers.csv：marker, chrom, pos, abs_r, p_value（%d 行，按 abs_r 降序）" % n_sel,
        "ld_landscape.png：L(p) 与准入阈值 q·(1−bL)、去重阈值 t_col(p) 三条曲线（%s 渲染）" % backend,
    ]
    with open(summary_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")

    print("总耗时 %.2f 秒，压缩比 %s" % (elapsed, "%.1f×" % comp if np.isfinite(comp) else "—"))
    for p in (csv_path, summary_path, png_path):
        print("  %-28s %7.1f KB" % (os.path.relpath(p, ROOT), os.path.getsize(p) / 1024.0))


if __name__ == "__main__":
    main()
