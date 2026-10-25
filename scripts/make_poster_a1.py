"""按 MD6329 展示模板（A1 横向）制作中文 Poster。

版式对齐参考模板：深蓝校名横幅 → 双语标题条 → 信息条 → 三栏主体 →
底部 RESULTS + CONCLUSION → 页脚。尺寸 33.46 × 23.62 inch（A1 横向）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams["font.family"] = ["Arial", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

NAVY = "#0B3D63"
BLUE = "#1478B8"
BLUE_L = "#E8F2FA"
BLUE_M = "#CFE4F4"
CYAN = "#2AA7C7"
INK = "#1B2733"
MUTED = "#56646F"
LINE = "#B9D3E6"
ORANGE = "#C0703A"
GREEN = "#3F7A46"

W, H = 33.46, 23.62


def band(fig, x, y, w, h, title, en=None, color=BLUE):
    fig.patches.append(Rectangle((x, y), w, h, facecolor=color, edgecolor="none", zorder=3))
    fig.patches.append(Rectangle((x, y), w, 0.0035, facecolor="white", edgecolor="none", zorder=4, alpha=0.35))
    fig.text(x + 0.008, y + h / 2, title, fontsize=20, fontweight="bold", color="white",
             va="center", ha="left", zorder=5)
    if en:
        fig.text(x + w - 0.008, y + h / 2, en, fontsize=11.5, color="#D6E9F6",
                 va="center", ha="right", zorder=5)


def panel(fig, x, y, w, h, fill="white", edge=LINE, lw=1.2):
    fig.patches.append(FancyBboxPatch((x, y), w, h,
                                      boxstyle="round,pad=0.0015,rounding_size=0.004",
                                      facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2))


def text_block(fig, x, y, lines, size=13, leading=0.0132, color=INK, weight="normal"):
    for i, line in enumerate(lines):
        if not line:
            continue
        fig.text(x, y - i * leading, line, fontsize=size, color=color,
                 va="top", ha="left", zorder=6, fontweight=weight, linespacing=1.28)


def metric(fig, x, y, value, label, color=ORANGE, size=25, label_size=10.5):
    fig.text(x, y, value, fontsize=size, fontweight="bold", color=color, va="top", ha="left", zorder=6)
    fig.text(x, y - 0.019, label, fontsize=label_size, color=MUTED, va="top", ha="left", zorder=6)


def picture(fig, path, cx, top, max_w, max_h):
    path = Path(path)
    if not path.exists():
        fig.text(cx, top - max_h / 2, f"[缺图 {path.name}]", fontsize=11, color="#B00020",
                 ha="center", va="center", zorder=6)
        return
    image = mpimg.imread(path)
    ih, iw = image.shape[:2]
    aspect = iw / ih
    w_in, h_in = max_w * W, max_h * H
    if w_in / aspect <= h_in:
        h_in = w_in / aspect
    else:
        w_in = h_in * aspect
    w, h = w_in / W, h_in / H
    axes = fig.add_axes([cx - w / 2, top - h, w, h])
    axes.imshow(image)
    axes.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-dir", default="manuscript/figures")
    parser.add_argument("--output", default="poster/MD6329_poster_A1")
    args = parser.parse_args()
    figures = Path(args.figure_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(W, H), dpi=150)
    fig.patch.set_facecolor("white")

    # ================= 顶部横幅 =================
    fig.patches.append(Rectangle((0, 0.884), 1.0, 0.116, facecolor=NAVY, edgecolor="none", zorder=1))
    fig.patches.append(Rectangle((0, 0.884), 1.0, 0.0025, facecolor=CYAN, edgecolor="none", zorder=2))
    fig.text(0.022, 0.962, "NANYANG TECHNOLOGICAL UNIVERSITY", fontsize=22, fontweight="bold",
             color="white", va="center", zorder=4)
    fig.text(0.022, 0.928, "MD6329 Deep Learning  ·  Final Course Project", fontsize=14,
             color="#BBD8EE", va="center", zorder=4)
    fig.text(0.022, 0.901, "南洋理工大学 · 深度学习课程项目", fontsize=13,
             color="#9CC4E0", va="center", zorder=4)
    fig.text(0.978, 0.958, "2026", fontsize=44, fontweight="bold", color="white",
             va="center", ha="right", zorder=4)
    fig.text(0.978, 0.915, "课程作品展示  ·  Course Project Showcase", fontsize=15,
             color="#BBD8EE", va="center", ha="right", zorder=4)

    # ================= 标题条 =================
    fig.patches.append(FancyBboxPatch((0.014, 0.812), 0.972, 0.062,
                                      boxstyle="round,pad=0.0018,rounding_size=0.005",
                                      facecolor=BLUE_L, edgecolor=BLUE, linewidth=2.0, zorder=2))
    fig.text(0.026, 0.855, "Project Title:", fontsize=14, fontweight="bold", color=BLUE,
             va="center", zorder=4)
    fig.text(0.088, 0.855, "Calibration Is Not Enough — A Three-Level Audit of Selective Prediction",
             fontsize=19, fontweight="bold", color=INK, va="center", zorder=4)
    fig.text(0.088, 0.828, "for Multi-Label ECG Classification",
             fontsize=19, fontweight="bold", color=INK, va="center", zorder=4)

    # ================= 信息条 =================
    fig.patches.append(Rectangle((0.014, 0.763), 0.972, 0.042, facecolor="#F4F9FD",
                                 edgecolor=LINE, linewidth=1.2, zorder=2))
    info = [
        (0.026, "项目名称：", "面向多标签心电图的选择性预测三级可靠性审计"),
        (0.300, "学生姓名：", "姜政宇"),
        (0.430, "课程：", "MD6329 Deep Learning"),
        (0.620, "数据集：", "PTB-XL v1.0.3（21,799 条 / 18,869 人）"),
        (0.855, "代码：", "可复现脚本 + lectrace 演示"),
    ]
    for x, label, value in info:
        fig.text(x, 0.784, label, fontsize=12, color=MUTED, va="center", zorder=4)
        fig.text(x + 0.048, 0.784, value, fontsize=12.5, fontweight="bold", color=INK,
                 va="center", zorder=4)

    # ================= 左栏：INTRODUCTION =================
    lx, lw = 0.014, 0.238
    band(fig, lx, 0.706, lw, 0.030, "INTRODUCTION", "研究背景与问题")
    panel(fig, lx, 0.300, lw, 0.398)

    fig.text(lx + 0.010, 0.688, "Research Background and Significance", fontsize=13,
             fontweight="bold", color=BLUE, va="top", zorder=6)
    text_block(fig, lx + 0.010, 0.670, [
        "AI 心电图最有价值的落点是分诊：对每条记录",
        "决定「直接签发」还是「转人工」。此时重要的是",
        "被签发记录中的错误率，而不是平均 AUROC。",
        "",
        "近年论文普遍同时报告校准指标（ECE），并默认",
        "「概率校准得好」意味着「置信度可用于决定何时",
        "拒答」。本项目检验这一推断是否成立。",
    ], size=12.2, leading=0.0130)

    fig.text(lx + 0.010, 0.578, "核心发现", fontsize=13, fontweight="bold", color=BLUE,
             va="top", zorder=6)
    metric(fig, lx + 0.012, 0.556, "0.0238 → 0.0068", "ECE（校准误差，改善 3.5 倍）", ORANGE, size=17, label_size=10)
    metric(fig, lx + 0.012, 0.516, "0.1900 → 0.1914", "AURC（风险–覆盖曲线下面积）", ORANGE, size=17, label_size=10)
    metric(fig, lx + 0.012, 0.476, "30.98 → 31.23", "80% 覆盖率下的记录级错误率（%）", ORANGE, size=17, label_size=10)
    text_block(fig, lx + 0.012, 0.446, [
        "概率质量显著改善，分诊行为几乎不动。",
    ], size=12.2, color=INK, weight="bold")

    fig.text(lx + 0.010, 0.418, "Pros & Cons of Existing Research", fontsize=13,
             fontweight="bold", color=BLUE, va="top", zorder=6)
    rows = [
        ("传统阈值法", "固定 0.5 阈值只定义分类器，\n不定义分诊策略"),
        ("校准方法", "改善概率质量，但单调变换\n不改变样本排序"),
        ("选择性预测", "把「打分函数」与「拒答粒度」\n当作既定前提"),
    ]
    y_cursor = 0.396
    for name, desc in rows:
        fig.patches.append(Rectangle((lx + 0.010, y_cursor - 0.040), lw - 0.020, 0.040,
                                     facecolor="#FBFCFE", edgecolor=LINE, linewidth=1.0, zorder=3))
        fig.text(lx + 0.017, y_cursor - 0.012, name, fontsize=11.2, fontweight="bold",
                 color=INK, va="center", zorder=5)
        fig.text(lx + 0.062, y_cursor - 0.013, desc, fontsize=9.6, color=MUTED,
                 va="center", zorder=5, linespacing=1.25)
        y_cursor -= 0.045

    # ================= 中栏：METHODOLOGY =================
    mx, mw = 0.258, 0.372
    band(fig, mx, 0.706, mw, 0.030, "METHODOLOGY", "三级审计协议与误差预算")
    panel(fig, mx, 0.300, mw, 0.398)

    fig.text(mx + 0.012, 0.688, "三级审计：按「侵入性」排序的后处理干预", fontsize=13,
             fontweight="bold", color=BLUE, va="top", zorder=6)

    levels = [
        ("L1", "校准层", "只改 logit → 概率映射", "温度缩放 T = 1.187", BLUE_M, BLUE),
        ("L2", "分数层", "改接受判据的统计量", "熵 / max-prob / 全对概率", "#D9E9F7", BLUE),
        ("L3", "粒度层", "改接受或拒答的决策单元", "记录级 vs 标签级", "#CCE3F2", "#0F5E8C"),
    ]
    y_cursor = 0.655
    for code, name, what, inst, fill, edge in levels:
        fig.patches.append(FancyBboxPatch((mx + 0.012, y_cursor - 0.052), 0.150, 0.052,
                                          boxstyle="round,pad=0.0012,rounding_size=0.004",
                                          facecolor=fill, edgecolor=edge, linewidth=1.4, zorder=3))
        fig.text(mx + 0.020, y_cursor - 0.014, code, fontsize=15, fontweight="bold",
                 color=edge, va="center", zorder=5)
        fig.text(mx + 0.040, y_cursor - 0.014, name, fontsize=12.5, fontweight="bold",
                 color=INK, va="center", zorder=5)
        fig.text(mx + 0.020, y_cursor - 0.036, what, fontsize=10.4, color=MUTED, va="center", zorder=5)
        fig.text(mx + 0.174, y_cursor - 0.020, inst, fontsize=10.6, color=INK, va="center", zorder=5)
        y_cursor -= 0.058

    fig.text(mx + 0.012, 0.478, "数据与评估协议", fontsize=13, fontweight="bold",
             color=BLUE, va="top", zorder=6)
    text_block(fig, mx + 0.012, 0.460, [
        "PTB-XL v1.0.3 · 12 导联 · 100 Hz · 10 秒。官方患者级划分：fold 1–8 训练",
        "（17,418 条）、fold 9 承担全部拟合决策（2,183 条）、fold 10 只评估一次",
        "（2,198 条 / 1,904 人）。模型为 3 段 1D ResNet，51,429 参数。",
    ], size=11.4, leading=0.0128)

    fig.text(mx + 0.012, 0.392, "误差预算：可优化的上限在哪里", fontsize=13,
             fontweight="bold", color=BLUE, va="top", zorder=6)
    fig.text(mx + 0.030, 0.372, r"$R_{current} - R_{oracle} = \varepsilon_{floor} + \varepsilon_{rank}$",
             fontsize=15, color=INK, va="top", zorder=6)
    budget_rows = [
        ("oracle", "用真实对错排序（完美排序，不可实现）", "#8FA9BC"),
        ("reference", "四架构集成的 max-prob 排序", CYAN),
        ("current", "部署模型的 max-prob 排序", ORANGE),
    ]
    y_cursor = 0.348
    for name, desc, color in budget_rows:
        fig.patches.append(Rectangle((mx + 0.016, y_cursor - 0.008), 0.010, 0.010,
                                     facecolor=color, edgecolor="none", zorder=4))
        fig.text(mx + 0.032, y_cursor - 0.003, name, fontsize=10.8, fontweight="bold",
                 color=color, va="center", zorder=5)
        fig.text(mx + 0.088, y_cursor - 0.003, desc, fontsize=10.4, color=MUTED,
                 va="center", zorder=5)
        y_cursor -= 0.019

    fig.text(mx + 0.016, 0.290, "样本外推断：患者级 bootstrap（1000 次），同一患者的记录整群进出。",
             fontsize=9.8, color=MUTED, va="top", zorder=6, style="italic")

    # ================= 右栏：RQ1 + RQ2 =================
    rx, rw = 0.642, 0.344

    band(fig, rx, 0.616, rw, 0.026, "RQ1 — 校准 ≠ 可靠", "probabilities vs. decisions")
    panel(fig, rx, 0.470, rw, 0.140)
    picture(fig, figures / "fig1.png", rx + rw / 2, 0.600, rw - 0.030, 0.120)
    text_block(fig, rx + 0.012, 0.492, [
        "(a) 可靠性曲线回到对角线；(b) 风险–覆盖两条曲线几乎重合。",
        "三个 seed 下校准后的错误率均略高于未校准基线：",
    ], size=11.2, leading=0.0128)
    fig.text(rx + 0.012, 0.478, "+0.18  ·  +0.37  ·  +0.46 个百分点（均值 +0.34 ± 0.14，三次同号）",
             fontsize=12.4, fontweight="bold", color=ORANGE, va="top", zorder=6)

    band(fig, rx, 0.436, rw, 0.026, "RQ2 — 打分函数的总预算", "the budget is 6.9 points")
    panel(fig, rx, 0.300, rw, 0.130)
    picture(fig, figures / "fig3.png", rx + rw / 2, 0.428, rw - 0.120, 0.118)
    fig.text(rx + 0.012, 0.418, "6.9", fontsize=46, fontweight="bold", color=ORANGE, va="top", zorder=6)
    fig.text(rx + 0.070, 0.410, "pp", fontsize=20, fontweight="bold", color=ORANGE, va="top", zorder=6)
    fig.text(rx + 0.012, 0.386, "任何打分函数", fontsize=10.6, color=MUTED, va="top", zorder=6)
    fig.text(rx + 0.012, 0.374, "的全部空间", fontsize=10.6, color=MUTED, va="top", zorder=6)
    text_block(fig, rx + 0.090, 0.372, [
        "80% 覆盖率下，完美排序仍会错 23.1%，",
        "模型实际错 30.0%。熵 → max-prob 拿回",
        "1.1 pp，学习型打分器再加 0.4 pp（噪声内）。",
    ], size=10.6, leading=0.0125)

    # ================= 底部：RESULTS =================
    bx, bw = 0.014, 0.620
    band(fig, bx, 0.246, bw, 0.030, "EXPERIMENTAL RESULTS", "粒度、亚组与多 seed")
    panel(fig, bx, 0.040, bw, 0.200)

    # RQ3
    fig.text(bx + 0.012, 0.228, "RQ3 · 粒度是数量级更大的杠杆", fontsize=13,
             fontweight="bold", color=GREEN, va="top", zorder=6)
    picture(fig, figures / "fig5.png", bx + 0.078, 0.212, 0.120, 0.150)
    text_block(fig, bx + 0.012, 0.196, [
        "记录级：79.6% 标签覆盖 · 9.90% 标签错误",
        "标签级：76.4% 标签覆盖 · 5.61% 标签错误",
        "",
        "标签错误减半，但整条无人工签发率从",
        "79.6% 降到 39.8%，平均每条仍有 1.18 个",
        "标签需要复核。是否值得取决于复核流程，",
        "而不是模型本身。",
    ], size=11.0, leading=0.0128)

    # 亚组
    fig.text(bx + 0.212, 0.228, "误差由谁承担", fontsize=13, fontweight="bold",
             color=ORANGE, va="top", zorder=6)
    picture(fig, figures / "fig6.png", bx + 0.330, 0.212, 0.190, 0.120)
    text_block(fig, bx + 0.212, 0.156, [
        "年龄 18–44 岁 16.3% vs 65–89 岁 40.6%，",
        "24.3 个百分点的差距是换分数效应的 20 倍。",
        "设备 TVD 0.326：测试集本身已是偏移集。",
    ], size=10.6, leading=0.0125)

    # 多 seed 表
    fig.text(bx + 0.438, 0.228, "三 seed 复现", fontsize=13, fontweight="bold",
             color=BLUE, va="top", zorder=6)
    header = ["seed", "coverage", "err", "Δ", "AUROC"]
    col_x = [bx + 0.438, bx + 0.480, bx + 0.524, bx + 0.552, bx + 0.578]
    for x, h in zip(col_x, header):
        fig.text(x, 0.206, h, fontsize=9.6, fontweight="bold", color=MUTED, va="center", zorder=6)
    fig.patches.append(Rectangle((bx + 0.435, 0.1985), 0.172, 0.0012,
                                 facecolor=LINE, edgecolor="none", zorder=4))
    seed_rows = [
        ("42", "79.6%", "31.14", "+0.18", "0.9076"),
        ("43", "78.3%", "29.81", "+0.37", "0.9082"),
        ("44", "80.7%", "31.81", "+0.46", "0.9083"),
    ]
    y_cursor = 0.188
    for row in seed_rows:
        for x, cell in zip(col_x, row):
            fig.text(x, y_cursor, cell, fontsize=9.8, color=INK, va="center", zorder=6)
        y_cursor -= 0.016
    fig.text(bx + 0.438, 0.136, "均值  79.5 ± 1.2    30.9 ± 1.0", fontsize=9.6,
             color=MUTED, va="center", zorder=6)
    fig.text(bx + 0.438, 0.122, "Δ 均值 +0.34 ± 0.14，三次全为正", fontsize=9.6,
             color=ORANGE, fontweight="bold", va="center", zorder=6)
    fig.text(bx + 0.438, 0.100, "判别力极稳（AUROC ±0.0004），", fontsize=9.6,
             color=MUTED, va="center", zorder=6)
    fig.text(bx + 0.438, 0.086, "波动全部落在拒答决策上。", fontsize=9.6,
             color=MUTED, va="center", zorder=6)

    # ================= 底部：CONCLUSION =================
    cx, cw = 0.642, 0.344
    band(fig, cx, 0.246, cw, 0.030, "CONCLUSION", "结论与启示")
    panel(fig, cx, 0.040, cw, 0.200, fill="#F4FAFF")

    points = [
        ("校准不等于可靠", "单调变换只改概率数值，不改样本排序；三个 seed 同向。"),
        ("分数层已接近耗尽", "6.9 pp 的总预算里，后处理分数与学习型打分器合计只拿回约 1.5 pp。"),
        ("粒度是真正的杠杆", "记录级换标签级，标签错误减半，代价是整条签发率腰斩。"),
        ("换报告方式", "报告风险–覆盖前沿、声明误差目标与拒答粒度，而不是单点加 ECE。"),
    ]
    y_cursor = 0.220
    for idx, (head, detail) in enumerate(points, start=1):
        fig.patches.append(FancyBboxPatch((cx + 0.012, y_cursor - 0.040), 0.020, 0.020,
                                          boxstyle="circle,pad=0.001", facecolor=BLUE,
                                          edgecolor="none", zorder=4))
        fig.text(cx + 0.022, y_cursor - 0.030, str(idx), fontsize=11, fontweight="bold",
                 color="white", va="center", ha="center", zorder=5)
        fig.text(cx + 0.040, y_cursor - 0.022, head, fontsize=12.2, fontweight="bold",
                 color=INK, va="center", zorder=5)
        fig.text(cx + 0.040, y_cursor - 0.040, detail, fontsize=10.4, color=MUTED,
                 va="center", zorder=5)
        y_cursor -= 0.048

    fig.patches.append(FancyBboxPatch((cx + 0.012, 0.052), cw - 0.024, 0.038,
                                      boxstyle="round,pad=0.0012,rounding_size=0.004",
                                      facecolor=BLUE, edgecolor="none", zorder=4))
    fig.text(cx + cw / 2, 0.071, "报告前沿、误差目标与拒答粒度 —— 而不是单点覆盖率加 ECE",
             fontsize=12.4, fontweight="bold", color="white", ha="center", va="center", zorder=5)

    # ================= 页脚 =================
    fig.patches.append(Rectangle((0, 0), 1.0, 0.030, facecolor="#F2F7FB", edgecolor="none", zorder=1))
    fig.patches.append(Rectangle((0, 0.0295), 1.0, 0.0012, facecolor=LINE, edgecolor="none", zorder=2))
    fig.text(0.014, 0.015, "Nanyang Technological University  ·  MD6329 Deep Learning  ·  Final Course Project",
             fontsize=10.5, color=MUTED, va="center", zorder=4)
    fig.text(0.500, 0.015, "代码：train.py → fit_calibration.py → evaluate.py  |  分析：decompose_gap.py, selective_head_insample.py",
             fontsize=10.5, color=MUTED, va="center", zorder=4)
    fig.text(0.986, 0.015, "2026.10", fontsize=10.5, color=MUTED, va="center", ha="right", zorder=4)

    for suffix, dpi in (("pdf", 300), ("png", 150)):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
