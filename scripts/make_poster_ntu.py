"""按毕业设计展示模板版式生成 A0 横向海报（33.46 × 23.62 inch）。

版式：顶部校徽横幅 → 英文标题条 → 中英信息条 → 三栏内容区 → 底部条。
学校、专业、配色替换为 NTU / AI in Medicine。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

NTU_BLUE = "#00205B"
NTU_BLUE_M = "#0F5C99"
NTU_GOLD = "#B49B57"
NTU_CYAN = "#1B7FA8"
CARD_FILL = "#F4FAFD"
CARD_EDGE = "#9FC2D8"
INK = "#16232E"
MUTED = "#4A5B68"
WHITE = "#FFFFFF"
ORANGE = "#A9662F"
GREEN = "#48733F"
PURPLE = "#5A4D8A"

W, H = 33.46, 23.62

LEFT_X, MID_X, RIGHT_X = 0.016, 0.334, 0.664
LEFT_W, MID_W, RIGHT_W = 0.306, 0.318, 0.320


def add(fig, patch):
    patch.set_transform(fig.transFigure)
    fig.add_artist(patch)


def rect(fig, x, y, w, h, color, z=1, alpha=1.0, lw=0):
    add(fig, Rectangle((x, y), w, h, facecolor=color, edgecolor="none",
                       linewidth=lw, alpha=alpha, zorder=z))


def section_bar(fig, x, y, w, text, color=NTU_CYAN, h=0.027, size=26):
    rect(fig, x, y, w, h, color, z=5)
    rect(fig, x + 0.005, y + 0.007, 0.008, h - 0.014, WHITE, z=6, alpha=0.9)
    fig.text(x + 0.021, y + h / 2, text, fontsize=size, fontweight="bold",
             color=WHITE, va="center", ha="left", zorder=7)


def panel(fig, x, y, w, h, fill=CARD_FILL, edge=CARD_EDGE, lw=1.8):
    add(fig, FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0025,rounding_size=0.006",
                            linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2))


def body(fig, x, y, lines, size=17, leading=0.0138, color=INK):
    for i, line in enumerate(lines):
        if not line:
            continue
        fig.text(x, y - i * leading, line, fontsize=size, color=color,
                 va="top", ha="left", zorder=8)


def stat(fig, x, y, value, label, color=NTU_CYAN, size=24, label_size=12.5):
    fig.text(x, y, value, fontsize=size, fontweight="bold", color=color, va="top", ha="left", zorder=8)
    fig.text(x, y - 0.0165, label, fontsize=label_size, color=MUTED, va="top", ha="left", zorder=8)


def picture(fig, path, x_center, y_top, max_w, max_h):
    path = Path(path)
    if not path.exists():
        fig.text(x_center, y_top - max_h / 2, f"[missing {path.name}]", fontsize=14,
                 color="#B00020", ha="center", va="center", zorder=9)
        return y_top - max_h
    image = mpimg.imread(path)
    ih, iw = image.shape[:2]
    aspect = iw / ih
    width_in, height_in = max_w * W, max_h * H
    if width_in / aspect <= height_in:
        height_in = width_in / aspect
    else:
        width_in = height_in * aspect
    w, h = width_in / W, height_in / H
    axes = fig.add_axes([x_center - w / 2, y_top - h, w, h], zorder=9)
    axes.imshow(image)
    axes.axis("off")
    return y_top - h


def compact_table(fig, x, y, w, rows, weights, size=12.5, head_size=12.5, row_h=0.0185):
    total = sum(weights)
    widths = [c / total * w for c in weights]
    for r, row in enumerate(rows):
        if r == 0:
            rect(fig, x, y - row_h, w, row_h, NTU_BLUE_M, z=6)
        elif r % 2 == 0:
            rect(fig, x, y - row_h, w, row_h, "#E6F1F8", z=5)
        cx = x
        for c, cell in enumerate(row):
            fig.text(cx + 0.0035, y - row_h / 2, str(cell),
                     fontsize=head_size if r == 0 else size,
                     fontweight="bold" if r == 0 else "normal",
                     color=WHITE if r == 0 else INK, va="center", ha="left", zorder=8)
            cx += widths[c]
        y -= row_h
    rect(fig, x, y, w, 0.0011, CARD_EDGE, z=6)
    return y


def draw_logo(fig, cx, cy, r, logo_path: Path | None):
    if logo_path and logo_path.exists():
        image = mpimg.imread(logo_path)
        ih, iw = image.shape[:2]
        h = 2 * r
        w = h * iw / ih * (H / W)
        axes = fig.add_axes([cx - w / 2, cy - h / 2, w, h], zorder=10)
        axes.imshow(image)
        axes.axis("off")
        return
    add(fig, Circle((cx, cy), r, facecolor="none", edgecolor=NTU_GOLD, linewidth=3.2, zorder=10))
    add(fig, Circle((cx, cy), r * 0.855, facecolor="none", edgecolor=NTU_GOLD, linewidth=1.1, zorder=10))
    fig.text(cx, cy + r * 0.20, "NTU", fontsize=31, fontweight="bold", color=WHITE,
             ha="center", va="center", zorder=11)
    fig.text(cx, cy - r * 0.30, "SINGAPORE", fontsize=10, color=NTU_GOLD,
             ha="center", va="center", zorder=11)


def ecg_decoration(fig, x0, x1, y_center, amplitude, color, alpha):
    t = np.linspace(0, 8 * np.pi, 3200)
    wave = np.sin(t) * 0.18
    for k in range(4):
        centre = (k + 0.55) * 2 * np.pi
        wave += 0.60 * np.exp(-((t - centre) ** 2) / 0.0035)
        wave -= 0.32 * np.exp(-((t - centre - 0.24) ** 2) / 0.005)
    xs = np.linspace(x0, x1, len(t))
    ys = y_center + amplitude * wave
    line = Line2D(xs, ys, color=color, linewidth=2.2, alpha=alpha, zorder=1)
    line.set_transform(fig.transFigure)
    fig.add_artist(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-dir", default="manuscript/figures")
    parser.add_argument("--output", default="poster/ntu_poster_AIinMedicine")
    parser.add_argument("--logo", default="poster/assets/ntu_logo.png")
    parser.add_argument("--name", default="姜政宇  Zhengyu")
    parser.add_argument("--supervisor", default="〔指导教师〕")
    args = parser.parse_args()

    figures = Path(args.figure_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(W, H), dpi=110)
    fig.patch.set_facecolor(WHITE)

    # ---------------- banner ----------------
    rect(fig, 0, 0.845, 1.0, 0.155, NTU_BLUE, z=0)
    rect(fig, 0, 0.845, 1.0, 0.0055, NTU_GOLD, z=1)
    ecg_decoration(fig, 0.46, 0.995, 0.912, 0.028, "#7FBCE0", 0.35)

    draw_logo(fig, 0.052, 0.9225, 0.0285, Path(args.logo))
    fig.text(0.090, 0.9455, "Nanyang Technological University", fontsize=32, fontweight="bold",
             color=WHITE, va="center", zorder=10)
    fig.text(0.090, 0.9075, "AI in Medicine   ·   MD6329 Deep Learning", fontsize=20,
             color=NTU_GOLD, va="center", zorder=10)
    fig.text(0.984, 0.9560, "2026", fontsize=50, fontweight="bold", color=WHITE,
             va="center", ha="right", zorder=10)
    fig.text(0.984, 0.9075, "Capstone Project Showcase", fontsize=18, color=NTU_GOLD,
             va="center", ha="right", zorder=10)

    # ---------------- title bar ----------------
    rect(fig, 0, 0.786, 1.0, 0.059, NTU_CYAN, z=2)
    fig.text(0.016, 0.8155, "Calibration Is Not Enough: A Three-Level Audit of Selective Prediction "
                            "for Multi-Label ECG Classification",
             fontsize=29, fontweight="bold", color=WHITE, va="center", zorder=4)

    # ---------------- info bar ----------------
    rect(fig, 0, 0.742, 1.0, 0.044, "#E8F3FA", z=2)
    rect(fig, 0, 0.7412, 1.0, 0.0012, CARD_EDGE, z=3)
    fig.text(0.016, 0.7635, "项目名称：校准并不足够 —— 多标签心电图选择性预测的三级审计",
             fontsize=20.5, fontweight="bold", color=INK, va="center", zorder=4)
    fig.text(0.512, 0.7635, f"学生姓名：{args.name}", fontsize=18, color=INK, va="center", zorder=4)
    fig.text(0.700, 0.7635, f"指导教师：{args.supervisor}", fontsize=18, color=INK, va="center", zorder=4)
    fig.text(0.870, 0.7635, "专业：AI in Medicine", fontsize=18, color=INK, va="center", zorder=4)

    # =================== LEFT ===================
    top = 0.722
    section_bar(fig, LEFT_X, top, LEFT_W, "INTRODUCTION  研究背景", NTU_CYAN)
    panel(fig, LEFT_X, 0.382, LEFT_W, top - 0.382 - 0.029)

    y = top - 0.050
    fig.text(LEFT_X + 0.014, y, "□  研究背景与意义", fontsize=20, fontweight="bold", color=NTU_BLUE,
             va="top", zorder=8)
    body(fig, LEFT_X + 0.016, y - 0.030, [
        "AI 心电图最现实的落点是分诊：对每条记录决定直接签发，",
        "还是转交医生复核。此时重要的不是平均 AUROC，而是",
        "「被签发的记录里错多少」。",
        "文献普遍同时报告 AUROC 与校准误差 ECE，并默认",
        "「概率校准得好，置信度就可用来决定是否拒答」。",
        "本项目检验这一推断。",
    ], size=17, leading=0.0136)

    fig.text(LEFT_X + 0.014, 0.552, "□  一句话结论", fontsize=20, fontweight="bold", color=NTU_BLUE,
             va="top", zorder=8)
    stat(fig, LEFT_X + 0.016, 0.518, "0.0238 → 0.0068", "test ECE（温度缩放后）", ORANGE, size=23)
    stat(fig, LEFT_X + 0.016, 0.478, "0.1900 → 0.1914", "AURC（同模型、同测试折）", ORANGE, size=23)
    stat(fig, LEFT_X + 0.016, 0.438, "30.98 → 31.23", "80% 覆盖率下 accepted error (%)", ORANGE, size=23)

    fig.text(LEFT_X + 0.016, 0.394,
             "校准改善 3.5 倍，前沿不动；三种子同向（+0.18 / +0.37 / +0.46）。",
             fontsize=14, color=INK, va="top", zorder=8)

    section_bar(fig, LEFT_X, 0.348, LEFT_W, "WHY CALIBRATION FAILS  机制", NTU_CYAN)
    panel(fig, LEFT_X, 0.062, LEFT_W, 0.348 - 0.062 - 0.029)
    picture(fig, figures / "fig1.png", LEFT_X + LEFT_W / 2, 0.302, LEFT_W - 0.055, 0.072)
    body(fig, LEFT_X + 0.016, 0.218, [
        "(a) 可靠性图回到对角线；(b) 风险–覆盖曲线几乎重合。",
        "",
        "温度缩放是 logit 的单调映射：同一条记录内五个标签",
        "的排序不变，样本级重排也极小（AURC 仅漂移 0.0014）。",
        "",
        "它改善的是概率，不是「哪条记录可以签发」。",
        "",
        "三级审计按侵入性排序：",
        "L1 校准  →  L2 分数  →  L3 粒度",
    ], size=16.5, leading=0.0132)

    # =================== MIDDLE ===================
    section_bar(fig, MID_X, top, MID_W, "METHODOLOGY  方法", NTU_CYAN)
    panel(fig, MID_X, 0.062, MID_W, top - 0.062 - 0.029)

    y = top - 0.050
    fig.text(MID_X + 0.014, y, "□  模型与训练", fontsize=20, fontweight="bold", color=NTU_BLUE,
             va="top", zorder=8)
    rows = [
        ["组件", "配置", "输出"],
        ["Input", "12 导联 × 1000 点（100 Hz，10 s）", "12×1000"],
        ["Stem", "Conv1d(k=15, s=2) + BN + ReLU", "32×500"],
        ["Block 1–3", "2×Conv1d(k=7) + BN + ReLU + Dropout", "32×500/250/125"],
        ["Pool+Head", "GAP → Linear(32→5)", "5 logits"],
    ]
    compact_table(fig, MID_X + 0.014, y - 0.012, MID_W - 0.028, rows, [1.05, 2.55, 1.25],
                  size=12.5, head_size=12.5, row_h=0.0188)
    body(fig, MID_X + 0.014, y - 0.122, [
        "51,429 参数 · AdamW（lr 1e-3，wd 1e-4）· batch 64 ·",
        "20 epoch · 按验证损失取 best（epoch 14）",
    ], size=15, leading=0.0128)

    y = y - 0.160
    fig.text(MID_X + 0.014, y, "□  三级审计协议", fontsize=20, fontweight="bold", color=NTU_BLUE,
             va="top", zorder=8)
    body(fig, MID_X + 0.016, y - 0.030, [
        "L1  校准层：只改 logit→概率映射，看概率质量能否",
        "     转化为决策质量",
        "L2  分数层：改接受判据（熵 / max-prob / 全对概率 /",
        "     学习型 correctness head）",
        "L3  粒度层：改决策单元（整条记录 vs 单个标签）",
    ], size=16, leading=0.0132)

    y = y - 0.140
    fig.text(MID_X + 0.014, y, "□  误差预算（本文新增）", fontsize=20, fontweight="bold", color=NTU_BLUE,
             va="top", zorder=8)
    fig.text(MID_X + 0.020, y - 0.034, "R_current = R_oracle + ε_floor + ε_rank",
             fontsize=16.5, color=NTU_BLUE, va="top", zorder=8, fontfamily="DejaVu Sans")
    body(fig, MID_X + 0.016, y - 0.064, [
        "oracle：按真实正确性排序（不可实现的上界）",
        "reference：四架构集成排序；current：部署模型",
        "ε_rank 与患者级 bootstrap 噪声比较，判定可辨识性",
    ], size=15.5, leading=0.0128)

    y = y - 0.146
    fig.text(MID_X + 0.014, y, "□  数据与协议纪律", fontsize=20, fontweight="bold", color=NTU_BLUE,
             va="top", zorder=8)
    body(fig, MID_X + 0.016, y - 0.030, [
        "PTB-XL v1.0.3 · 21,799 条 / 18,869 人 · 官方患者级划分",
        "fold 1–8 训练 17,418 条；fold 9 拟合温度与全部阈值；",
        "fold 10 只评估一次",
        "保留 411 条全零标签；age=300 按「>89」分组，不删",
    ], size=15.5, leading=0.0128)

    # =================== RIGHT ===================
    section_bar(fig, RIGHT_X, top, RIGHT_W, "EXPERIMENTAL RESULTS  实验", NTU_CYAN)
    panel(fig, RIGHT_X, 0.352, RIGHT_W, top - 0.352 - 0.029)

    y = top - 0.048
    fig.text(RIGHT_X + 0.013, y, "□  三个随机种子", fontsize=19, fontweight="bold", color=NTU_BLUE,
             va="top", zorder=8)
    rows = [
        ["Seed", "覆盖率", "基线错误", "校准后", "Δ(pp)", "AUROC"],
        ["42", "79.62%", "30.96%", "31.14%", "+0.18", "0.9076"],
        ["43", "78.30%", "29.44%", "29.81%", "+0.37", "0.9082"],
        ["44", "80.66%", "31.35%", "31.81%", "+0.46", "0.9083"],
        ["均值", "79.53", "30.58", "30.92", "+0.34", "0.9080"],
    ]
    compact_table(fig, RIGHT_X + 0.013, y - 0.028, RIGHT_W - 0.026, rows,
                  [0.78, 1.18, 1.28, 1.18, 0.95, 1.18], size=12, head_size=11.5, row_h=0.0182)

    y = y - 0.124
    fig.text(RIGHT_X + 0.013, y, "□  分数层：预算只有 6.9 个百分点", fontsize=19, fontweight="bold",
             color=NTU_BLUE, va="top", zorder=8)
    picture(fig, Path("poster/assets/fig_combo.png"), RIGHT_X + RIGHT_W / 2, y - 0.022, RIGHT_W - 0.040, 0.116)
    body(fig, RIGHT_X + 0.016, y - 0.150, [
        "80% 覆盖率下，完美排序仍错 23.1%，实测 30.0%：",
        "任何打分函数的全部空间只有 6.9pp。换分数拿到",
        "1.1pp，学习型 head 再加 0.4pp，落在噪声内。",
    ], size=15.5, leading=0.0128)

    section_bar(fig, RIGHT_X, 0.318, RIGHT_W, "CONCLUSION  结论", NTU_CYAN)
    panel(fig, RIGHT_X, 0.062, RIGHT_W, 0.318 - 0.062 - 0.029)
    body(fig, RIGHT_X + 0.014, 0.250, [
        "1. 校准不等于可选择可靠性：ECE 降 3.5 倍，前沿不动。",
        "2. 分数层已近乎用尽：6.9pp 的空间里只取回 1.5pp。",
        "3. 粒度才是杠杆：标签级在 76.4% 覆盖下把错误从",
        "     9.90% 降到 5.61%，代价是整条全接受率从 79.6%",
        "     降到 39.8%，平均 1.18 个标签转人工。",
        "4. 年龄亚组差 16.3% vs 40.6%，是分数效应的二十倍。",
        "",
        "局限：单中心、单模型家族；标签级阈值在 fold 9 上整段调参；",
        "未做外部验证，故风险控制数字是经验规则而非统计保证。",
    ], size=15.5, leading=0.0132)

    # ---------------- footer ----------------
    rect(fig, 0, 0.0, 1.0, 0.050, NTU_BLUE, z=2)
    rect(fig, 0, 0.0495, 1.0, 0.0015, NTU_GOLD, z=3)
    fig.text(0.016, 0.025, "数据：PTB-XL v1.0.3（PhysioNet）   ·   代码：train.py → fit_calibration.py → evaluate.py，"
                           "分析 decompose_gap.py / selective_head_insample.py",
             fontsize=13.5, color=WHITE, va="center", zorder=8)
    fig.text(0.984, 0.025, "Nanyang Technological University  ·  AI in Medicine  ·  2026",
             fontsize=13.5, color=NTU_GOLD, va="center", ha="right", zorder=8)

    for suffix, dpi in (("pdf", 200), ("png", 110)):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=dpi, facecolor=WHITE)
    plt.close(fig)
    print(f"wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
