# A-PCS：自适应皮尔逊共线性筛选（Adaptive Pearson Collinearity Screening）

本文件夹是大创项目《基于自适应特征选择与深度双流架构模型的作物抗病表型预测》中
**A-PCS 子模块**的完整技术报告、实验代码与结果。

## 文件结构

```
A-PCS/
├── A-PCS技术报告.pdf        # 主报告（含方法、迭代记录、全部实验结果）
├── code/                    # 全部实验代码（Python 3.13）
│   ├── apcs_core.py         # A-PCS 与基线方法实现 + GBLUP
│   ├── simulate.py          # 带 LD 异质性的作物基因组仿真器
│   ├── experiments.py       # 无泄漏交叉验证评估框架
│   ├── run_sim.py           # 仿真实验（--mode dev/full, --version v1/v2/v3）
│   ├── prepare_real.py      # 真实数据预处理（BGLR mice/wheat）
│   ├── run_mice.py          # 小鼠真实数据实验（1814 个体 × 10,346 SNP × 6 性状）
│   ├── run_wheat.py         # 小麦真实数据实验（599 × 1,279 × 4 环境）
│   ├── run_loco.py          # 留群体外推泛化实验
│   ├── run_transfer.py      # 跨数据集超参数迁移实验（仿真→小鼠零重调）
│   ├── run_hgb.py           # 更换非线性下游模型（HGB）检验
│   ├── diagnostics.py       # LD 异质性 / QTL 覆盖度诊断
│   ├── make_figures.py      # 报告图表生成
│   └── fill_report.py       # 报告数字注入（保证数字与结果 CSV 一致）
├── results/                 # 全部逐折结果 CSV 与诊断 JSON
├── figures/                 # 报告插图 PNG
└── README.md
```

## 数据来源

- 真实数据：R/BGLR 1.1.4（CRAN）自带的 mice（Valdar 等 heterogeneous stock 小鼠，
  含 38 个性状与物理图谱）与 wheat（国际小麦头行产量试验）数据。
- 仿真数据：`simulate.py` 一键重建（12 染色体 × 30 Mb、59,301 SNP、1600 个体、
  3 个 LD 异质区段、3 个亚群、3 种遗传架构，种子 2026/7）。

## 复现

```bash
cd code
python simulate.py            # 仿真器自检
python run_sim.py --mode full --version v3     # 仿真全量实验
python run_mice.py            # 小鼠实验
python run_wheat.py           # 小麦实验
python run_loco.py --source mice && python run_loco.py --source sim
python run_transfer.py        # 阈值迁移实验
python run_hgb.py --source sim && python run_hgb.py --source mice
python diagnostics.py both && python make_figures.py all
python fill_report.py         # 生成报告 HTML（随后由 Edge 渲染 PDF）
```

依赖：numpy, scipy, pandas, scikit-learn, matplotlib, lightgbm（可选）, pyreadr。
