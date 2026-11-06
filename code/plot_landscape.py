# -*- coding: utf-8 -*-
"""LD 景观与自适应阈值绘图。

一张图里画三条曲线（横轴=全基因组累加位置）：
  * L(p)            —— LD 强度景观（按染色体着色，左轴 0–1）
  * t_col(p)        —— 去重阈值（左轴 0–1）
  * q·(1 − b·L(p))  —— 表型准入水平（右轴，量级很小）

优先用 matplotlib（Agg 后端）；没有 matplotlib 时回退到 Pillow 手绘，
保证只装 numpy/scipy/Pillow 也能出图。
"""
import os

import numpy as np

_TAB10 = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
]
COLOR_TCOL = (200, 60, 60)
COLOR_Q = (40, 140, 90)


def _ordered_unique(chrom):
    out = []
    for c in np.asarray(chrom).tolist():
        s = str(c)
        if s not in out:
            out.append(s)
    return out


def cumulative_positions(pos, chrom):
    """把各染色体物理位置拼成一条全基因组累加坐标轴。"""
    pos = np.asarray(pos, dtype=np.float64)
    chrom = np.asarray(chrom, dtype=object)
    uniq = _ordered_unique(chrom)
    cum = np.zeros(len(pos), dtype=np.float64)
    offset = 0.0
    span = float(pos.max() - pos.min()) if len(pos) else 0.0
    gap = 0.02 * max(1.0, span)
    for c in uniq:
        sel = np.array([str(x) == c for x in chrom])
        if not np.any(sel):
            continue
        p = pos[sel]
        cum[sel] = p - p.min() + offset
        offset += float(p.max() - p.min()) + gap
    return cum, uniq


def thresholds(L, q, b, t_col_min, t_col_max):
    """A-PCS 的两条位置依赖阈值。"""
    L = np.asarray(L, dtype=np.float64)
    q_eff = q * (1.0 - b * L)
    t_col = t_col_max - (t_col_max - t_col_min) * L
    return q_eff, t_col


def _fmt_bp(v):
    if abs(v) >= 1e6:
        return "%.1f Mb" % (v / 1e6)
    if abs(v) >= 1e3:
        return "%.0f kb" % (v / 1e3)
    return "%.0f" % v


def _nice_qmax(q_max, step=0.03):
    """右轴取整到 step 的整数倍，避免出现 0.13 这种刻度。"""
    return max(step, float(np.ceil(float(q_max) / step) * step))


def plot_ld_landscape(path, pos, L, chrom, W, q=0.1, b=0.7,
                      t_col_min=0.5, t_col_max=0.9,
                      title="A-PCS LD landscape and adaptive thresholds"):
    """画三条曲线并保存 PNG，返回使用的后端名。"""
    pos = np.asarray(pos, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    chrom = np.asarray(chrom, dtype=object)
    cum, uniq = cumulative_positions(pos, chrom)
    q_eff, t_col = thresholds(L, q, b, t_col_min, t_col_max)
    subtitle = ("W = %s bp | %d markers | %d chromosomes | "
                "q=%g, b=%g, t_col=%.2f-%.2f"
                % (format(int(W), ","), len(pos), len(uniq), q, b, t_col_min, t_col_max))
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        return _with_mpl(plt, path, cum, L, chrom, uniq, q_eff, t_col, title, subtitle)
    except ImportError:
        pass
    return _with_pillow(path, cum, L, chrom, uniq, q_eff, t_col, title, subtitle)


# ------------------------------------------------------------------ matplotlib
def _with_mpl(plt, path, cum, L, chrom, uniq, q_eff, t_col, title, subtitle):
    from matplotlib.ticker import FuncFormatter

    colors = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(11.8, 5.8), dpi=150)
    ax2 = ax.twinx()

    for i, c in enumerate(uniq):
        sel = np.array([str(x) == c for x in chrom])
        if not np.any(sel):
            continue
        ax.plot(cum[sel], L[sel], color=colors[i % len(colors)], linewidth=0.9,
                label="L(p) chromosome %s" % c)
    ax.plot(cum, t_col, color=[v / 255.0 for v in COLOR_TCOL], linewidth=1.6,
            linestyle="--", label="t_col(p)")
    ax2.plot(cum, q_eff, color=[v / 255.0 for v in COLOR_Q], linewidth=1.6,
             linestyle="-.", label="q(1-bL)")

    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(float(np.min(cum)), float(np.max(cum)) + 1e-9)
    ax2.set_ylim(0.0, _nice_qmax(float(np.max(q_eff)) * 1.1))
    ax.set_xlabel("Genomic position (bp, cumulative)")
    ax.set_ylabel("LD strength L / t_col(p)")
    ax2.set_ylabel("admission level q(1-bL)")
    ax.set_title(title + "\n" + subtitle, fontsize=11.5)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=min(len(h1) + len(h2), 5), fontsize=8.5, frameon=False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: _fmt_bp(v)))
    fig.tight_layout()
    fig.savefig(path, format="png")
    plt.close(fig)
    return "matplotlib"


# ---------------------------------------------------------------------- Pillow
class _Fonts(object):
    def __init__(self):
        from PIL import ImageFont
        self._ImageFont = ImageFont
        self._cache = {}

    def get(self, size, bold=False):
        key = (size, bold)
        if key in self._cache:
            return self._cache[key]
        cands = []
        if bold:
            cands.append(r"C:\Windows\Fonts\arialbd.ttf")
        cands += [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\DejaVuSans.ttf"]
        font = None
        for c in cands:
            if os.path.exists(c):
                try:
                    font = self._ImageFont.truetype(c, size)
                    break
                except Exception:
                    continue
        if font is None:
            font = self._ImageFont.load_default()
        self._cache[key] = font
        return font


def _thin(pts, max_points=40000):
    if len(pts) <= max_points:
        return pts
    step = int(np.ceil(len(pts) / float(max_points)))
    idx = list(range(0, len(pts), step))
    if idx[-1] != len(pts) - 1:
        idx.append(len(pts) - 1)
    return [pts[i] for i in idx]


def _dashed(draw, pts, color, width=3, dash=16, gap=10):
    """把折线画成虚线段（Pillow 没有原生虚线）。"""
    carry = 0.0
    on = True
    remaining = dash
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        seg = float(np.hypot(x1 - x0, y1 - y0))
        if seg <= 1e-9:
            continue
        pos = 0.0
        while pos < seg:
            step = min(remaining, seg - pos)
            if on:
                f0, f1 = pos / seg, (pos + step) / seg
                draw.line([(x0 + (x1 - x0) * f0, y0 + (y1 - y0) * f0),
                           (x0 + (x1 - x0) * f1, y0 + (y1 - y0) * f1)],
                          fill=color, width=width)
            pos += step
            remaining -= step
            if remaining <= 1e-9:
                on = not on
                remaining = dash if on else gap
        carry += seg  # 仅用于保持可读性，段间状态已由 remaining 维护


def _with_pillow(path, cum, L, chrom, uniq, q_eff, t_col, title, subtitle):
    from PIL import Image, ImageDraw

    Wc, Hc = 1600, 860
    img = Image.new("RGB", (Wc, Hc), "white")
    draw = ImageDraw.Draw(img)
    fonts = _Fonts()
    f_title = fonts.get(27, True)
    f_sub = fonts.get(16)
    f_tick = fonts.get(16)
    f_label = fonts.get(18)
    f_legend = fonts.get(16)

    left, top, right, bottom = 130, 118, 120, 116
    x0, y0, x1, y1 = left, top, Wc - right, Hc - bottom

    draw.text((left, 26), title, fill=(25, 32, 42), font=f_title)
    draw.text((left, 70), subtitle, fill=(96, 106, 118), font=f_sub)

    xmin, xmax = float(np.min(cum)), float(np.max(cum))
    if xmax <= xmin:
        xmax = xmin + 1.0
    qmax = _nice_qmax(float(np.max(q_eff)) * 1.1)

    def X(v):
        return x0 + (v - xmin) / (xmax - xmin) * (x1 - x0)

    def YL(v):
        return y0 + (1.0 - float(v)) * (y1 - y0)

    def YR(v):
        return y0 + (1.0 - float(v) / qmax) * (y1 - y0)

    # 网格（左轴）
    for k in range(5):
        val = k / 4.0
        yv = YL(val)
        draw.line([(x0, yv), (x1, yv)], fill=(228, 232, 238), width=1)
        draw.text((left - 50, yv - 9), "%.2f" % val, fill=(110, 118, 128), font=f_tick)
    for k in range(7):
        val = xmin + (xmax - xmin) * k / 6.0
        xv = X(val)
        draw.line([(xv, y0), (xv, y1)], fill=(242, 244, 248), width=1)
        label = _fmt_bp(val)
        tw = draw.textlength(label, font=f_tick)
        draw.text((xv - tw / 2, y1 + 12), label, fill=(110, 118, 128), font=f_tick)
    # 右轴刻度（准入水平）
    for k in range(5):
        val = qmax * k / 4.0
        yv = YR(val)
        label = ("%.2f" % val) if qmax >= 0.01 else ("%.3f" % val)
        draw.line([(x1, yv), (x1 + 6, yv)], fill=(40, 140, 90), width=1)
        draw.text((x1 + 12, yv - 9), label, fill=(40, 140, 90), font=f_tick)

    # 曲线：先画阈值，再画 L
    qpts = _thin([(X(a), YR(b)) for a, b in zip(cum, q_eff)])
    _dashed(draw, qpts, COLOR_Q, width=3, dash=14, gap=9)
    tpts = _thin([(X(a), YL(b)) for a, b in zip(cum, t_col)])
    _dashed(draw, tpts, COLOR_TCOL, width=3, dash=18, gap=10)
    for i, c in enumerate(uniq):
        sel = np.array([str(x) == c for x in chrom])
        if not np.any(sel):
            continue
        pts = _thin([(X(a), YL(b)) for a, b in zip(cum[sel], L[sel])])
        if len(pts) > 1:
            draw.line(pts, fill=_TAB10[i % len(_TAB10)], width=3, joint="curve")

    # 坐标轴
    draw.line([(x0, y0), (x0, y1)], fill=(60, 68, 78), width=1)
    draw.line([(x0, y1), (x1, y1)], fill=(60, 68, 78), width=1)
    draw.line([(x1, y0), (x1, y1)], fill=(40, 140, 90), width=1)
    xlabel = "Genomic position (bp, cumulative)"
    tw = draw.textlength(xlabel, font=f_label)
    draw.text(((x0 + x1) / 2 - tw / 2, Hc - 44), xlabel, fill=(60, 68, 78), font=f_label)
    _vertical_text(img, draw, (26, y0 + (y1 - y0) / 2 - 90), "LD strength L / t_col(p)",
                   (60, 68, 78), f_label)
    _vertical_text(img, draw, (Wc - 34, y0 + (y1 - y0) / 2 - 80), "admission q(1-bL)",
                   (40, 140, 90), f_label)

    # 图例
    lx, ly = left, Hc - 74
    entries = [("chromosome %s" % c, _TAB10[i % len(_TAB10)]) for i, c in enumerate(uniq)]
    entries += [("t_col(p)", COLOR_TCOL), ("q(1-bL) [right axis]", COLOR_Q)]
    for text, color in entries:
        draw.line([(lx, ly + 8), (lx + 22, ly + 8)], fill=color, width=3)
        draw.text((lx + 28, ly), text, fill=(70, 78, 88), font=f_legend)
        lx += 28 + draw.textlength(text, font=f_legend) + 24
        if lx > x1 - 120:
            lx = left
            ly += 22

    img.save(path, format="PNG")
    return "pillow"


def _vertical_text(img, draw, xy, text, fill, font):
    from PIL import Image, ImageDraw
    box = draw.textbbox((0, 0), text, font=font)
    tw, th = int(box[2] - box[0]), int(box[3] - box[1])
    tmp = Image.new("RGBA", (tw + 8, th + 8), (255, 255, 255, 0))
    ImageDraw.Draw(tmp).text((4, 4), text, fill=fill, font=font)
    rot = tmp.rotate(90, expand=True)
    img.paste(rot, (int(xy[0]), int(xy[1])), rot)
