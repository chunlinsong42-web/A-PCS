# data/sample：最小示例数据

用于开箱跑通 `code/example_run.py` 的合成数据集：200 个体 × 2000 标记，2 条染色体、各约 1 Mb。
只依赖 numpy/scipy 即可读取，总大小约 0.8 MB（< 3 MB）。

## 文件与字段

| 文件 | 行 | 列 | 说明 |
| --- | --- | --- | --- |
| `matrix.csv` | 200（+1 表头） | 2001 | 第 1 列 `sample_id`，其余 2000 列为标记名 `m0001`…`m2000`；取值 `0/1/2`（剂量），缺失留空 |
| `phenotype.csv` | 200（+1 表头） | 2 | `sample_id,phenotype`，表型已中心化（均值 0），单位任意 |
| `positions.csv` | 2000（+1 表头） | 3 | `marker,chrom,pos`；`chrom` 为 `1/2`，`pos` 为该染色体上的物理位置（bp） |

行内样本顺序与 `phenotype.csv` 一致（`S001`…`S200`）；标记按 `chrom,pos` 升序排列。

## 数据特征

- **规模**：200 样本 × 2000 标记；chrom 1 与 chrom 2 各 1000 个标记、各约 1.0 Mb（标记间距均值 ≈ 1 kb）。
- **LD 结构**：每条染色体按 350（强 LD）→ 350（弱 LD/重组热点）→ 300（强 LD）的区段生成，
  强区重组率 0.001/标记、弱区 0.08/标记，因此 `L(p)` 沿染色体有明显的高—低—高变化。
- **表型**：30 个因果位点的加性效应 + 环境噪声，遗传力中等；表型与基因型存在真实的弱—中等关联。
- **质控素材**：约 0.2% 基因型缺失、3 个单态位点、3 个低频位点（MAF 极低），
  让 `qc_markers` 的 MAF/缺失率过滤真正起作用（示例中 2000 → 1994 个标记）。

## 生成方式

```bash
python code/make_sample_data.py                 # 默认随机种子 20261103，写到 data/sample/
python code/make_sample_data.py --seed 7 --out /tmp/sample
```

生成逻辑见 `code/make_sample_data.py`：

1. 每条染色体上按 0.5–1.5 kb 的随机间距铺 1000 个标记，线性缩放到 1 Mb 以内；
2. 按区段设定每个标记的重组概率，用「单倍型行走」生成两条单倍型：遇到重组就重新抽样等位基因，
   否则继承前一位点——低重组区因此形成长 LD 块，高重组区 LD 迅速衰减；
3. 两条单倍型相加得到 `0/1/2` 剂量矩阵，再注入缺失、单态与低频位点；
4. 抽 30 个因果位点、按正态分布给效应量，叠加噪声得到表型。

随机种子固定，重复运行得到完全相同的数据（可直接复现实验）。

## 跑示例

```bash
python code/example_run.py        # 默认读 data/sample/，输出到 results/example/
```

产物：`results/example/selected_markers.csv`（marker, chrom, pos, abs_r, p_value，按 abs_r 降序）、
`summary.txt`（输入、参数、标记数、压缩比、耗时）、`ld_landscape.png`（L(p)、q·(1−bL)、t_col(p) 三条曲线）。
