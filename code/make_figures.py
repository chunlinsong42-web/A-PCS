# -*- coding: utf-8 -*-
"""Generate all report figures (Chinese labels, Microsoft YaHei)."""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

RES = "../results"
FIG = "../figures"
os.makedirs(FIG, exist_ok=True)

METHOD_LABELS = {
    "All": "全标记 GBLUP",
    "LDPrune": "LD剪枝",
    "PCS": "PCS (固定阈值)",
    "APCS": "A-PCS v1",
    "APCS2": "A-PCS (p值空间)",
    "APCS2w": "A-PCS (加权核)",
    "APCS_fixSig": "A-PCS v1 w/o 自适应准入",
    "APCS_fixCol": "A-PCS v1 w/o 自适应去重",
    "APCS2_fixSig": "A-PCS w/o 阈值调制",
    "APCS2_fixCol": "A-PCS w/o 自适应去重",
    "APCS2w": "A-PCS(加权核)",
    "Random": "随机同规模子集",
    "PCS_HGB": "PCS+HGB",
    "APCS_HGB": "A-PCS+HGB",
    "All_HGB": "全标记+HGB",
}
METHOD_ORDER = ["All", "LDPrune", "PCS", "APCS2w", "APCS2", "APCS_fixSig", "APCS_fixCol",
                "APCS2_fixSig", "APCS2_fixCol", "Random"]
COLORS = {"All": "#9aa5b1", "LDPrune": "#c4907f", "PCS": "#5b8ff9", "APCS": "#b08bd9",
          "APCS2": "#f6c3b1", "APCS2w": "#e8684a", "APCS_fixSig": "#b7d0f0",
          "APCS_fixCol": "#f6c3b1", "APCS2_fixSig": "#d9c7e8", "APCS2_fixCol": "#f0d9c0",
          "Random": "#cbc5b8"}


def agg(df, method_col="method", val="pcc"):
    g = df.groupby([ "trait", method_col])[val].agg(["mean", "std", "count"]).reset_index()
    return g


def fig_main_sim():
    df = pd.read_csv(f"{RES}/sim_full_v5.csv")
    traits = ["T1_block_oligo", "T2_polygenic", "T3_mixed_epi"]
    trait_names = {"T1_block_oligo": "性状T1：连锁块主效架构",
                   "T2_polygenic": "性状T2：热点多基因架构",
                   "T3_mixed_epi": "性状T3：混合+上位性"}
    methods = [m for m in METHOD_ORDER if m in df.method.unique()]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), sharey=False)
    for ax, t in zip(axes, traits):
        sub = df[df.trait == t]
        stats = sub.groupby("method")["pcc"].agg(["mean", "std", "count"]).reindex(methods)
        x = np.arange(len(methods))
        ax.bar(x, stats["mean"], yerr=stats["std"].fillna(0), capsize=3,
               color=[COLORS[m] for m in methods], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m].replace(" ", "\n", 1) for m in methods],
                           fontsize=7, rotation=28, ha="right")
        ax.set_title(trait_names[t], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        for i, m in enumerate(methods):
            ax.text(i, stats["mean"].iloc[i] + 0.012, f"{stats['mean'].iloc[i]:.3f}",
                    ha="center", fontsize=7)
    axes[0].set_ylabel("预测 PCC (5折×3种子 均值±SD)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_main_sim.png", bbox_inches="tight")
    print("fig_main_sim done")


def fig_mice():
    df = pd.read_csv(f"{RES}/mice_v5.csv")
    methods = [m for m in METHOD_ORDER if m in df.method.unique()]
    traits = list(pd.unique(df.trait))
    M = len(methods)
    fig, ax = plt.subplots(figsize=(12, 4))
    w = 0.8 / M
    x = np.arange(len(traits))
    stats = df.groupby(["trait", "method"])["pcc"].agg(["mean", "std"]).reset_index()
    for i, m in enumerate(methods):
        vals = [stats[(stats.trait == t) & (stats.method == m)]["mean"].values
                for t in traits]
        vals = [v[0] if len(v) else np.nan for v in vals]
        errs = [stats[(stats.trait == t) & (stats.method == m)]["std"].values
                for t in traits]
        errs = [e[0] if len(e) else 0 for e in errs]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, yerr=errs, capsize=2,
               label=METHOD_LABELS[m], color=COLORS[m], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(traits, fontsize=8, rotation=12)
    ax.set_ylabel("预测 PCC (5折×2种子 均值±SD)")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_mice.png", bbox_inches="tight")
    print("fig_mice done")


def fig_ld_sim():
    with open(f"{RES}/diag_sim.json") as f:
        diag = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax = axes[0]
    colors = {"hotspot": "#e8684a", "moderate": "#5b8ff9", "block": "#3aa675"}
    names = {"hotspot": "重组热点区", "moderate": "中度连锁区", "block": "强连锁区块"}
    for reg in ["hotspot", "moderate", "block"]:
        p = diag["ld_profile"][reg]
        r2 = [v if v is not None else np.nan for v in p["r2"]]
        ax.plot(p["bin_edges"][1:], r2, "o-", ms=3, color=colors[reg], label=names[reg])
    ax.set_xlabel("标记间物理距离 (kb)")
    ax.set_ylabel("平均 $r^2$ (LD)")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.set_title("(a) 三类区段的 LD 衰减曲线（仿真）", fontsize=10)
    ax.grid(alpha=0.3)
    ax = axes[1]
    pos = np.array(diag["landscape"]["pos"]) / 1e3
    chrom = np.array(diag["landscape"]["chrom"])
    dh = np.array(diag["landscape"]["dh"]) / 1e3
    m = chrom == 1
    ax.fill_between(pos[m], dh[m], step="mid", color="#5b8ff9", alpha=0.5, lw=0)
    ax.set_xlabel("1号染色体物理位置 (kb)")
    ax.set_ylabel("窗口 LD 半衰减距离 $D_{1/2}$ (kb)")
    ax.set_yscale("log")
    ax.set_title("(b) 沿染色体的局部 LD 衰减距离（仿真 1 号染色体）", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_ld_sim.png", bbox_inches="tight")
    print("fig_ld_sim done")


def fig_landscape():
    with open(f"{RES}/diag_sim.json") as f:
        diag = json.load(f)
    L = diag["landscape"]
    pos = np.array(L["pos"]) / 1e3
    chrom = np.array(L["chrom"])
    m = chrom == 1
    # true regime strip from simulator truth (for visual reference only)
    d = np.load(f"../data/sim_2026.npz", allow_pickle=True)
    reg_all = d["regime"]
    pos_all = d["pos"] / 1e3
    chrom_all = d["chrom"]
    reg_map = {"hotspot": 0, "moderate": 1, "block": 2}
    mr = chrom_all == 1
    reg_y = np.array([reg_map[r] for r in reg_all[mr]])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(pos[m], np.array(L["L"])[m], color="#5b8ff9", lw=0.8,
            label="LD 强度景观 $L(p)$")
    ax.plot(pos[m], np.array(L["t_sig"])[m] / max(np.array(L["t_sig"])[m].max(), 1e-9),
            color="#e8684a", lw=0.8, label="自适应准入阈值 $t_{sig}(p)$（归一化）")
    ax.plot(pos[m], np.array(L["t_col"])[m] / max(np.array(L["t_col"])[m].max(), 1e-9),
            color="#3aa675", lw=0.8, label="自适应共线性阈值 $t_{col}(p)$（归一化）")
    ax2 = ax.twinx()
    ax2.fill_between(pos_all[mr], 0, 1, where=reg_y == 2, color="#3aa675", alpha=0.12,
                     step="mid", lw=0)
    ax2.fill_between(pos_all[mr], 0, 1, where=reg_y == 0, color="#e8684a", alpha=0.10,
                     step="mid", lw=0)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.text(0.01, 0.93, "绿色底纹=强连锁块，红色底纹=重组热点（真实区段）", fontsize=8,
             transform=ax2.transAxes, color="#444")
    ax.set_xlabel("1 号染色体物理位置 (kb)")
    ax.set_ylabel("阈值 / LD 强度（归一化）")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("A-PCS 阈值随局部 LD 衰减自适应变化（仿真 1 号染色体）", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_landscape.png", bbox_inches="tight")
    print("fig_landscape done")


def fig_recall():
    with open(f"{RES}/diag_sim.json") as f:
        diag = json.load(f)
    traits = ["T1_block_oligo", "T2_polygenic", "T3_mixed_epi"]
    tnames = {"T1_block_oligo": "T1", "T2_polygenic": "T2", "T3_mixed_epi": "T3"}
    regimes = ["hotspot", "moderate", "block"]
    rnames = {"hotspot": "热点区", "moderate": "中度区", "block": "强连锁区"}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, t in zip(axes, traits):
        rec = diag[f"recall_{t}"]
        x = np.arange(len(regimes))
        w = 0.26
        methods = [("PCS", "PCS（固定阈值）", "#5b8ff9", -w),
                   ("APCS2", "A-PCS（本文）", "#e8684a", 0),
                   ("APCS", "A-PCS v1（迭代前）", "#c4a35b", w)]
        for key, lab, color, off in methods:
            if key not in rec:
                continue
            vals = [rec[key].get(r, {}).get("coverage", np.nan) for r in regimes]
            ax.bar(x + off, vals, w, label=lab, color=color, edgecolor="white")
            for i, v in enumerate(vals):
                if np.isfinite(v) and v > 0.005:
                    ax.text(i + off, v + 0.015, f"{v:.2f}", ha="center", fontsize=6,
                            rotation=90)
        ax.set_xticks(x); ax.set_xticklabels([rnames[r] for r in regimes], fontsize=8)
        ax.set_title(f"{tnames[t]}：因果位点信号覆盖度", fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("平均最大 LD 覆盖 $r^2$")
    axes[0].legend(fontsize=7, loc="upper center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_recall.png", bbox_inches="tight")
    print("fig_recall done")


def fig_loco():
    dfs = []
    for name in ["mice", "sim"]:
        try:
            d = pd.read_csv(f"{RES}/loco_all.csv")
            d = d[d["dataset"] == name]
            if len(d):
                dfs.append(d)
        except Exception:
            pass
    if not dfs:
        print("no loco results yet"); return
    df = pd.concat(dfs)
    order_src = [x for x in ["sim", "mice"] if x in set(df.dataset)]
    fig, axes = plt.subplots(1, len(order_src), figsize=(6.2 * len(order_src), 3.8))
    if len(order_src) == 1:
        axes = [axes]
    for ax, src in zip(axes, order_src):
        sub = df[df.dataset == src]
        methods = [m for m in METHOD_ORDER if m in sub.method.unique()]
        stats = sub.groupby("method")["pcc"].agg(["mean", "std"]).reindex(methods)
        x = np.arange(len(methods))
        ax.bar(x, stats["mean"], yerr=stats["std"].fillna(0), capsize=3,
               color=[COLORS[m] for m in methods], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=8, rotation=15, ha="right")
        ax.set_title(f"{'小鼠真实数据' if src=='mice' else '仿真数据'}：留群体外推", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        for i, mm in enumerate(methods):
            ax.text(i, stats['mean'].iloc[i] + 0.01, f"{stats['mean'].iloc[i]:.3f}",
                    ha="center", fontsize=8)
        if src == "mice":
            ax.set_ylabel("PCC（留一群体外推均值±SD）")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_loco.png", bbox_inches="tight")
    print("fig_loco done")


def fig_hgb():
    try:
        df = pd.read_csv(f"{RES}/hgb_all.csv")
    except Exception:
        print("no hgb results yet"); return
    order = ["All_HGB", "PCS_HGB", "APCS2_HGB", "APCS2w_HGB"]
    labs = ["全标记+HGB", "PCS+HGB", "A-PCS+HGB", "A-PCS(加权核)+HGB"]
    colors = ["#9aa5b1", "#5b8ff9", "#f6c3b1", "#e8684a"]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    x = np.arange(1)
    for i, (m, lab, c) in enumerate(zip(order, labs, colors)):
        s = df[df.method == m]["pcc"]
        ax.bar(x + (i - 1.5) * 0.2, [s.mean()], 0.2, yerr=[s.std()], capsize=3,
               label=lab, color=c, edgecolor="white")
    ax.set_xticks([])
    ax.set_xlabel("仿真 3 性状 + 小鼠 2 性状（合并）", fontsize=9)
    ax.set_ylabel("预测 PCC")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("替换非线性下游模型（HGB）后的方法排序一致性", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_hgb.png", bbox_inches="tight")
    print("fig_hgb done")


def fig_wheat():
    try:
        df = pd.read_csv(f"{RES}/wheat_v5.csv")
    except Exception:
        print("no wheat results"); return
    methods = [m for m in METHOD_ORDER if m in df.method.unique()]
    stats = df.groupby("method")["pcc"].agg(["mean", "std"]).loc[methods]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(methods))
    ax.bar(x, stats["mean"], yerr=stats["std"].fillna(0), capsize=3,
           color=[COLORS[m] for m in methods], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("预测 PCC (4环境×5折×3种子)")
    ax.set_title("小麦真实数据（n=599, 1279 标记）", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for i, m in enumerate(methods):
        ax.text(i, stats["mean"].iloc[i] + 0.008, f"{stats['mean'].iloc[i]:.3f}",
                ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_wheat.png", bbox_inches="tight")
    print("fig_wheat done")


def fig_mice_ld():
    try:
        d = np.load(f"{RES}/diag_mice_ld.npz")
    except Exception:
        print("no mice ld diag"); return
    centers, chrw, dh = d["centers"] / 1e6, d["chrw"], d["dh"] / 1e6
    fig, ax = plt.subplots(figsize=(11, 3.6))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for i, c in enumerate(np.unique(chrw)):
        m = chrw == c
        ax.plot(centers[m], dh[m], lw=0.8, color=colors[i % 20], label=str(c))
    ax.set_yscale("log")
    ax.set_xlabel("染色体物理位置 (Mb)")
    ax.set_ylabel("窗口 LD 半衰减距离 (Mb)")
    ax.set_title("小鼠真实数据：沿基因组的局部 LD 衰减距离异质性", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_mice_ld.png", bbox_inches="tight")
    print("fig_mice_ld done")


def fig_iter():
    """Algorithm iteration history on dev set (methods present per version)."""
    rows = []
    for v in ["v1", "v2", "v3"]:
        try:
            d = pd.read_csv(f"{RES}/sim_dev_{v}.csv")
        except Exception:
            continue
        for m in ["PCS", "APCS", "APCS2", "APCS2w"]:
            s = d[d.method == m]["pcc"]
            if len(s):
                rows.append(dict(version=v, method=m, pcc=s.mean(), sd=s.std()))
    if not rows:
        print("no dev results"); return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    versions = sorted(df.version.unique())
    bars = [("PCS", "#5b8ff9"), ("APCS", "#b08bd9"), ("APCS2", "#f6c3b1"), ("APCS2w", "#e8684a")]
    present = [(m, c) for m, c in bars if m in set(df.method)]
    n_b = len(present)
    w = 0.8 / n_b
    x = np.arange(len(versions))
    for i, (m, c) in enumerate(present):
        sub = df[df.method == m].set_index("version")
        vals = [sub.pcc.get(v, np.nan) for v in versions]
        offs = [j for j, v in enumerate(vals) if np.isfinite(v)]
        vv = [v for v in vals if np.isfinite(v)]
        ax.bar(np.array(offs) + (i - (n_b - 1) / 2) * w, vv, w,
               label=METHOD_LABELS[m], color=c, edgecolor="white")
        for j, v in zip(offs, vv):
            ax.text(j + (i - (n_b - 1) / 2) * w, v + 0.006, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}（开发集）" for v in versions])
    ax.set_ylabel("PCC（5折×2种子 均值）")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("A-PCS 算法迭代改进过程（开发集，逐版本新增方法）", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_iter.png", bbox_inches="tight")
    print("fig_iter done")


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = dict(sim=fig_main_sim, mice=fig_mice, ld=fig_ld_sim, land=fig_landscape,
               recall=fig_recall, loco=fig_loco, hgb=fig_hgb, wheat=fig_wheat,
                mld=fig_mice_ld, iter=fig_iter)
    if which == "all":
        for f in fns.values():
            try:
                f()
            except Exception as e:
                print("skip:", f.__name__, e)
    else:
        fns[which]()
