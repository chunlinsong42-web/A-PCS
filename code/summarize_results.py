# -*- coding: utf-8 -*-
"""汇总 results/ 下的逐折结果，生成 summary_metrics.csv 与 summary_tables.md。

输入：results/sim_full_v5.csv、mice_v5.csv、wheat_v5.csv
输出：
    results/summary_metrics.csv   method × trait 的 mean PCC / sd / mean n_sel
    results/summary_tables.md     各数据集完整表 + README 用精简对比表

用法：python code/summarize_results.py
"""
import csv
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")

SOURCES = [
    ("sim_full_v5.csv", "sim", "仿真（sim_full_v5）"),
    ("mice_v5.csv", "mice", "小鼠（mice_v5）"),
    ("wheat_v5.csv", "wheat", "小麦（wheat_v5）"),
]

METHOD_ORDER = ["All", "Random", "LDPrune", "PCS",
                "APCS2", "APCS2w", "APCS2_fixSig", "APCS2_fixCol"]

METHOD_ZH = {
    "All": "全标记",
    "Random": "随机子集",
    "LDPrune": "LD 剪枝",
    "PCS": "固定 PCS",
    "APCS2": "A-PCS",
    "APCS2w": "A-PCS（加权）",
    "APCS2_fixSig": "A-PCS（固定准入）",
    "APCS2_fixCol": "A-PCS（固定去重）",
}

TRAIT_ZH = {
    "T1_block_oligo": "T1 块状寡基因",
    "T2_polygenic": "T2 多基因",
    "T3_mixed_epi": "T3 混合+上位",
    "Biochem.Glucose": "血糖",
    "Biochem.HDL": "HDL",
    "Biochem.Tot.Cholesterol": "总胆固醇",
    "Obesity.BMI": "BMI",
    "Obesity.BodyLength": "体长",
    "Obesity.EndNormalBW": "末体重",
    "wheat_env1": "环境 1",
    "wheat_env2": "环境 2",
    "wheat_env4": "环境 4",
    "wheat_env5": "环境 5",
}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def aggregate(rows):
    """按 (trait, method) 汇总 mean PCC / sd / mean n_sel / 运行次数。"""
    buckets = {}
    for r in rows:
        key = (r["trait"], r["method"])
        buckets.setdefault(key, {"pcc": [], "n_sel": []})
        buckets[key]["pcc"].append(float(r["pcc"]))
        buckets[key]["n_sel"].append(float(r["n_sel"]))

    out = {}
    for (trait, method), v in buckets.items():
        pcc = v["pcc"]
        sd = statistics.stdev(pcc) if len(pcc) > 1 else 0.0
        out[(trait, method)] = {
            "n_runs": len(pcc),
            "mean_pcc": statistics.fmean(pcc),
            "sd_pcc": sd,
            "mean_n_sel": statistics.fmean(v["n_sel"]),
        }
    return out


def ordered_traits(agg):
    """按首次出现顺序返回 trait 列表。"""
    seen = []
    for (trait, _method) in agg:
        if trait not in seen:
            seen.append(trait)
    return seen


def fmt(v, nd=3):
    return ("%." + str(nd) + "f") % v


def method_row(agg, trait, method):
    return agg.get((trait, method))


def best_of(values):
    """返回最大值的下标（忽略 None）。"""
    best, idx = None, -1
    for i, v in enumerate(values):
        if v is None:
            continue
        if best is None or v > best:
            best, idx = v, i
    return idx


def write_metrics_csv(all_aggs):
    path = os.path.join(RESULTS, "summary_metrics.csv")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "trait", "method", "n_runs",
                    "mean_pcc", "sd_pcc", "mean_n_sel"])
        for dataset, agg in all_aggs:
            for trait in ordered_traits(agg):
                for method in METHOD_ORDER:
                    row = method_row(agg, trait, method)
                    if row is None:
                        continue
                    w.writerow([dataset, trait, method, row["n_runs"],
                                "%.6f" % row["mean_pcc"], "%.6f" % row["sd_pcc"],
                                "%.1f" % row["mean_n_sel"]])
    return path


def full_table(agg, title):
    traits = ordered_traits(agg)
    lines = ["### " + title, "",
             "| 方法 | " + " | ".join(TRAIT_ZH.get(t, t) for t in traits) + " | 平均 |",
             "| --- | " + " | ".join("---:" for _ in traits) + " | ---: |"]
    for method in METHOD_ORDER:
        cells = []
        vals = []
        for trait in traits:
            row = method_row(agg, trait, method)
            if row is None:
                cells.append("—")
                vals.append(None)
                continue
            cells.append("%s (%s)" % (fmt(row["mean_pcc"]), fmt(row["sd_pcc"])))
            vals.append(row["mean_pcc"])
        present = [v for v in vals if v is not None]
        avg = statistics.fmean(present) if present else None
        lines.append("| %s `%s` | %s | %s |"
                     % (METHOD_ZH[method], method, " | ".join(cells),
                        fmt(avg) if avg is not None else "—"))
    lines.append("")
    lines.append("均值±标准差：± 后为该 trait 在全部 seed×fold 上的 PCC 标准差；单位与 GBLUP 的 PCC 一致。")
    return "\n".join(lines)


def compact_table(all_aggs):
    """README 用：A-PCS vs 全标记 vs 固定 PCS，按 trait 标出最优。"""
    methods = ["APCS2", "All", "PCS"]
    header_zh = ["A-PCS `APCS2`", "全标记 `All`", "固定 PCS `PCS`"]
    lines = ["| 数据集 | 性状 | " + " | ".join(header_zh) + " |",
             "| --- | --- | " + " | ".join("---:" for _ in methods) + " |"]
    for dataset, title, agg in all_aggs:
        traits = ordered_traits(agg)
        for trait in traits:
            vals = [method_row(agg, trait, m)["mean_pcc"] for m in methods]
            bidx = best_of(vals)
            cells = []
            for i, v in enumerate(vals):
                s = fmt(v)
                cells.append("**%s**" % s if i == bidx else s)
            lines.append("| %s | %s | %s |"
                         % (title, TRAIT_ZH.get(trait, trait), " | ".join(cells)))
        # 数据集平均
        avgs = []
        for m in methods:
            present = [method_row(agg, t, m)["mean_pcc"] for t in traits]
            avgs.append(statistics.fmean(present))
        bidx = best_of(avgs)
        cells = ["**%s**" % fmt(v) if i == bidx else fmt(v) for i, v in enumerate(avgs)]
        lines.append("| **%s 平均** | — | %s |" % (title.split("（")[0], " | ".join(cells)))
    return "\n".join(lines)


def ds_mean(agg, method):
    traits = ordered_traits(agg)
    return statistics.fmean(method_row(agg, t, method)["mean_pcc"] for t in traits)


def ds_nsel(agg, method):
    traits = ordered_traits(agg)
    return statistics.fmean(method_row(agg, t, method)["mean_n_sel"] for t in traits)


def reading_notes(by_ds):
    sim, mice, wheat = by_ds["sim"], by_ds["mice"], by_ds["wheat"]
    t3 = "T3_mixed_epi"
    sim_apcs = ds_mean(sim, "APCS2")
    sim_all = ds_mean(sim, "All")
    sim_pcs = ds_mean(sim, "PCS")
    sim_apcsw = ds_mean(sim, "APCS2w")
    t3_apcs = method_row(sim, t3, "APCS2")["mean_pcc"]
    t3_all = method_row(sim, t3, "All")["mean_pcc"]
    t3_apcsw = method_row(sim, t3, "APCS2w")["mean_pcc"]
    mice_apcs, mice_all = ds_mean(mice, "APCS2"), ds_mean(mice, "All")
    wheat_apcs, wheat_all = ds_mean(wheat, "APCS2"), ds_mean(wheat, "All")
    r_sim = ds_nsel(sim, "All") / ds_nsel(sim, "APCS2")
    r_mice = ds_nsel(mice, "All") / ds_nsel(mice, "APCS2")
    r_wheat = ds_nsel(wheat, "All") / ds_nsel(wheat, "APCS2")
    return [
        "## 读法",
        "",
        "- 仿真（3 个性状平均）：A-PCS **%.3f**，全标记 **%.3f**，固定 PCS **%.3f**，加权版 APCS2w **%.3f**。"
        % (sim_apcs, sim_all, sim_pcs, sim_apcsw),
        "- 仿真的增益集中在 T3（混合+上位）：A-PCS **%.3f** vs 全标记 **%.3f**（%+.3f），"
        "加权版 **%.3f**（%+.3f）。"
        % (t3_apcs, t3_all, t3_apcs - t3_all, t3_apcsw, t3_apcsw - t3_all),
        "- 真实数据上 A-PCS 与全标记基本打平：小鼠 **%.3f** vs **%.3f**，小麦 **%.3f** vs **%.3f**。"
        % (mice_apcs, mice_all, wheat_apcs, wheat_all),
        "- 维度压缩（全标记 / A-PCS 的平均保留标记数）：仿真 **%.1f×**、小鼠 **%.1f×**、小麦 **%.1f×**。"
        % (r_sim, r_mice, r_wheat),
        "",
    ]


def main():
    all_aggs = []
    for filename, dataset, title in SOURCES:
        path = os.path.join(RESULTS, filename)
        rows = read_csv(path)
        agg = aggregate(rows)
        all_aggs.append((dataset, title, agg))

    metrics = write_metrics_csv([(d, a) for d, _, a in all_aggs])
    print("写出", os.path.relpath(metrics, ROOT))

    parts = [
        "# 结果汇总表",
        "",
        "由 `code/summarize_results.py` 从 `results/*.csv` 自动生成（逐折结果 → method × trait 汇总）。",
        "",
        "聚合口径：同一 trait、同一方法在 **全部 seed × fold** 上取均值，`sd` 为这些折的 PCC 标准差；"
        "`n_sel` 为保留下来的标记数均值。全部指标由无泄漏协议得到（特征选择不接触测试集）。",
        "",
        "方法说明：`APCS2` = A-PCS 主方法；`APCS2w` = 带 `|PCC|^γ` 加权的版本；"
        "`APCS2_fixSig` / `APCS2_fixCol` 分别固定准入/固定去重阈值的消融；"
        "`All` = 全标记 GBLUP（不做筛选）；`PCS` = 固定阈值 PCS；`LDPrune` = LD 剪枝；`Random` = 随机子集。",
        "",
    ]
    for _dataset, title, agg in all_aggs:
        parts.append(full_table(agg, title))
        parts.append("")

    parts += [
        "## README 用精简对比表",
        "",
        "三列分别是 A-PCS、全标记、固定 PCS 的平均 PCC；**粗体**为该行最优值。",
        "",
        compact_table(all_aggs),
        "",
        "读法见下一节；表中数值由本脚本从 `results/*.csv` 重算，不再逐字引用报告正文。",
        "",
    ]
    by_ds = {d: a for d, _t, a in all_aggs}
    parts += reading_notes(by_ds)
    out = os.path.join(RESULTS, "summary_tables.md")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(parts))
    print("写出", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
