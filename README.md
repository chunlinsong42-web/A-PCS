# A-PCS：自适应皮尔逊共线性筛选

**Adaptive Pearson Collinearity Screening** —— 一种面向全基因组预测、能随局部连锁不平衡（LD）结构自适应调整阈值的特征选择方法。

在全基因组预测中，标记数往往远高于个体数（p ≫ n），且标记间高度共线。经典做法是用皮尔逊相关做两段式筛选：先按"标记—表型"相关筛掉弱关联位点，再按"标记—标记"相关剔除冗余。但全基因组共用一组固定阈值忽略了一个事实：**染色体的 LD 衰减速度沿基因组剧烈变化**。近着丝粒和低重组区 LD 衰减慢，一个因果信号会复制出大量高相关代理位点；重组热点区 LD 衰减快，微效位点的边际相关天然很弱。固定阈值因此在强连锁区留下冗余、在热点区误删真实信号。

A-PCS 的做法是：先用滑动窗口估计每个区间的 LD 半衰减距离，得到沿基因组的 **LD 强度景观** `L(p) ∈ [0,1]`，再让两个阈值成为位置的连续函数——LD 越强，表型准入越严、共线性剔除越激进；LD 越弱则相反。v2 进一步把准入阈值从相关系数空间搬到显著性 p 值空间，使筛选规则对样本量、标记密度和物种 LD 尺度近似不变，从而可以跨数据集迁移。

## 方法要点

1. **LD 景观估计**：按染色体切滑动窗口（默认物理宽度 `W`），在每个窗口内计算 r² 随距离的衰减曲线，取 r² 降到初始值一半处的距离作为该窗口的 LD 半衰减距离 `d_half`；对 `d_half` 做分位数归一化与中值平滑，插值到每个标记，得到 `L(p)`。
2. **自适应准入**（v2）：标记 `i` 的准入水平为 `q·(1 − b·L(p_i))`，即强 LD 区用更小的名义显著水平，避免代理位点涌入；弱 LD 区放宽，保留微效位点。p 值由相关系数的 t 检验给出，与标记数无关。
3. **自适应去重**：保留阈值 `t_col(p) = t_max − (t_max − t_min)·L(p)`，去重半径取 `max(2·d_half(p), W)`；按 `|r|` 从大到小贪心扫描，若半径内已有保留位点且相关超过阈值则丢弃。
4. **加权下游接口**（可选）：用 `|PCC|^γ` 作为标记权重送入 GBLUP 等核方法，把筛选信息带到预测阶段。

## 主要结果

在带真实 LD 异质性的仿真数据（1600 个体 × 59,301 SNP，3 种遗传架构）与两套真实数据（小鼠 1814 × 10,346 × 6 性状；小麦 599 × 1,279 × 4 环境）上，用特征选择不接触测试集的无泄漏协议评估：

- p ≫ n 的仿真场景：相对全标记 GBLUP 最高提升 **+0.24 PCC**，维度压缩 **2.8–148×**（视性状架构而定）；
- 真实数据（10 个性状-环境组合）：与基线统计打平或接近（小鼠平均 PCC 0.412 vs 全标记 0.401），同时保持 **8–30×** 维度压缩；
- 因果信号覆盖度平均为固定阈值 PCS 的 **1.9 倍**；
- 留群体外推、更换非线性下游模型（HGB）、跨数据集阈值迁移三项检验中方法排序保持一致。

完整实验、消融与失败案例分析见 [A-PCS技术报告.pdf](A-PCS技术报告.pdf)。

## 安装

Python ≥ 3.10，依赖：

```bash
pip install numpy scipy pandas scikit-learn matplotlib
# 可选：lightgbm（HGB 实验）、pyreadr（读取 BGLR 数据）
```

## 快速开始

```python
import numpy as np
from apcs_core import qc_markers, standardize, ld_precompute, apcs2

X     = ...   # (n, m) 基因型矩阵，编码 0/1/2
y     = ...   # (n,)   表型
pos   = ...   # (m,)   标记物理位置（bp）
chrom = ...   # (m,)   染色体编号

# 1) 质控与标准化
Xq, keep = qc_markers(X, maf_min=0.05, miss_max=0.2)
pos, chrom = pos[keep], chrom[keep]
Z, _, _ = standardize(Xq)
yz = (y - y.mean()) / (y.std() + 1e-12)

# 2) 估计 LD 景观
W = 150_000                                   # 窗口宽度（bp），按物种 LD 尺度调整
pre = ld_precompute(Z, pos, chrom, W=W)

# 3) 自适应筛选
kept, r = apcs2(
    Z, yz, pos, chrom, W=W,
    q=0.1, b=0.7,                             # 准入水平与 LD 调制深度
    t_col_min=0.5, t_col_max=0.9,             # 去重阈值的两个端点
    pre=pre,
)

Xsel = Xq[:, kept]                            # 低维候选集
print(f"{Xq.shape[1]} -> {Xsel.shape[1]} markers")
```

`apcs2` 是推荐版本；`apcs`（v1，相关系数空间）与 `apcs2_fix_sig` / `apcs2_fix_col` 保留用于消融对比。`ld_precompute` 返回的 `L`、`d_half` 也可直接用于可视化 LD 景观。

参数经验值：`W` 取该物种 LD 半衰减距离的 2–3 倍；`q` 在 0.05–0.3 之间，`p/n` 越小越宽松；`b` 取 0.5–0.7；`(t_col_min, t_col_max)` 常用 `(0.5, 0.9)` 或 `(0.6, 0.95)`。

## 复现实验

```bash
cd code
python simulate.py                              # 仿真器自检
python run_sim.py --mode full --version v3      # 仿真全量实验
python run_mice.py                              # 小鼠数据
python run_wheat.py                             # 小麦数据
python run_loco.py --source mice                # 留群体外推
python run_transfer.py                          # 阈值迁移
python run_hgb.py --source sim                  # 更换下游模型
python diagnostics.py both && python make_figures.py all
python fill_report.py                           # 生成报告 HTML，再用浏览器打印为 PDF
```

所有逐折结果在 `results/`，图表在 `figures/`，报告数字由 `fill_report.py` 从结果 CSV 直接注入，保证正文与数据一致。

## 数据来源

- 真实数据：R/BGLR 1.1.4（CRAN）自带的 mice（Valdar 等 heterogeneous stock 小鼠）与 wheat（国际小麦头行产量试验）数据集。
- 仿真数据：`simulate.py` 一键重建（12 染色体 × 30 Mb、59,301 SNP、1600 个体、3 个 LD 异质区段、3 个亚群、3 种遗传架构，随机种子 2026/7）。

## 目录结构

```
A-PCS/
├── A-PCS技术报告.pdf   # 方法、实验与迭代记录
├── code/
│   ├── apcs_core.py    # A-PCS 核心实现 + 基线 + GBLUP
│   ├── simulate.py     # LD 异质性仿真器
│   ├── experiments.py  # 无泄漏交叉验证框架
│   ├── run_*.py        # 各数据集/检验的入口
│   ├── diagnostics.py  # LD 与 QTL 覆盖度诊断
│   ├── make_figures.py # 图表
│   └── fill_report.py  # 报告数字注入
├── results/            # 逐折结果 CSV 与诊断 JSON
├── figures/            # 报告插图
└── README.md
```

## 适用边界

- 对纯上位性（主效应为零）位点不敏感，这类信号只能通过边缘化效应间接保留；
- LD 景观估计依赖物理位置或可靠的标记顺序，无图谱时需用伪距离近似；
- 名义显著水平 `q` 未做有效检验数校正，块内代表唯一性承担了部分多重性控制；
- 下游换成强非线性模型时，最优 `q` 可能更宽松，本报告的参数网格以 GBLUP 为整定目标。

## 版本

- **v1.0.0** —— 首个公开版本：方法、代码、结果与技术报告。

## 联系

问题、建议或合作：提交 [Issue](https://github.com/chunlinsong42-web/A-PCS/issues)。
