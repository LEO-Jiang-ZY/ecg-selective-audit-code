"""A0 横向英文海报：四栏、六图两表，文字压到断句式。

版式沿用课程展示模板的四段结构（校名横幅 → 标题条 → 信息条 → 分栏正文），
配色取自 code/src/ptbxl/palette.py 的参考色卡。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptbxl.palette import INK, MUTED, STYLE  # noqa: E402

W, H = 33.46, 23.62

BAND = STYLE["blue"]["edge"]        # #4D77A7
BAND_SOFT = STYLE["aqua"]["fill"]   # #DAF0F4
BAND_PALE = STYLE["steel"]["fill"]  # #DCE6EE
ACCENT = STYLE["coral"]["edge"]     # #C97F86
CARD_EDGE = "#C9DCE8"
WHITE = "#FFFFFF"

COL_X = [0.014, 0.260, 0.506, 0.752]
COL_W = 0.230
TOP = 0.786
BOTTOM = 0.028
GAP = 0.013
BAR_H = 0.021


def add(fig, patch):
    patch.set_transform(fig.transFigure)
    fig.add_artist(patch)


def rect(fig, x, y, w, h, color, z=1, alpha=1.0):
    add(fig, Rectangle((x, y), w, h, facecolor=color, edgecolor="none", alpha=alpha, zorder=z))


def card(fig, x, y, h, title, size=21, fill=WHITE):
    """画章节条 + 卡片，返回正文可用起点。"""
    rect(fig, x, y, COL_W, BAR_H, BAND, z=5)
    rect(fig, x + 0.0035, y + 0.005, 0.006, BAR_H - 0.010, WHITE, z=6, alpha=0.9)
    fig.text(x + 0.016, y + BAR_H / 2, title, fontsize=size, fontweight="bold",
             color=WHITE, va="center", ha="left", zorder=7)
    add(fig, FancyBboxPatch((x, y - h), COL_W, h, boxstyle="round,pad=0.002,rounding_size=0.005",
                            linewidth=1.4, edgecolor=CARD_EDGE, facecolor=fill, zorder=2))
    return y - BAR_H - 0.012


def text_block(fig, x, y, lines, size=17, leading=0.0148, color=INK):
    for i, line in enumerate(lines):
        if not line:
            continue
        fig.text(x, y - i * leading, line, fontsize=size, color=color,
                 va="top", ha="left", zorder=8)


def stat(fig, x, y, value, label, color=INK, size=26):
    fig.text(x, y, value, fontsize=size, fontweight="bold", color=color, va="top", ha="left", zorder=8)
    gap = size * 0.00098            # 27pt → 0.0265；22pt → 0.0216
    fig.text(x, y - gap, label, fontsize=13.5, color=MUTED, va="top", ha="left", zorder=8)


def picture(fig, path, x_center, y_top, max_w, max_h):
    path = Path(path)
    if not path.exists():
        fig.text(x_center, y_top - max_h / 2, f"[missing {path.name}]", fontsize=13,
                 color=ACCENT, ha="center", va="center", zorder=9)
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


def table(fig, x, y, w, rows, weights, size=15.5, row_h=0.0152, highlight_last=False):
    total = sum(weights)
    widths = [c / total * w for c in weights]
    for r, row in enumerate(rows):
        if r == 0:
            rect(fig, x, y - row_h, w, row_h, BAND, z=6)
        elif highlight_last and r == len(rows) - 1:
            rect(fig, x, y - row_h, w, row_h, BAND_SOFT, z=5)
        elif r % 2 == 0:
            rect(fig, x, y - row_h, w, row_h, "#F3F7FA", z=5)
        cx = x
        for c, cell in enumerate(row):
            fig.text(cx + 0.0028, y - row_h / 2, str(cell), fontsize=size,
                     fontweight="bold" if (r == 0 or (highlight_last and r == len(rows) - 1)) else "normal",
                     color=WHITE if r == 0 else INK, va="center", ha="left", zorder=8)
            cx += widths[c]
        y -= row_h
    rect(fig, x, y, w, 0.0009, CARD_EDGE, z=6)
    return y


def logo(fig, cx, cy, r):
    add(fig, Circle((cx, cy), r, facecolor="none", edgecolor=STYLE["aqua"]["edge"], linewidth=2.8, zorder=10))
    add(fig, Circle((cx, cy), r * 0.86, facecolor="none", edgecolor=STYLE["aqua"]["edge"], linewidth=0.9, zorder=10))
    fig.text(cx, cy + r * 0.20, "NTU", fontsize=28, fontweight="bold", color=WHITE,
             ha="center", va="center", zorder=11)
    fig.text(cx, cy - r * 0.30, "SINGAPORE", fontsize=8.5, color=STYLE["aqua"]["fill"],
             ha="center", va="center", zorder=11)


def ecg_line(fig, x0, x1, y_center, amplitude, color, alpha):
    t = np.linspace(0, 8 * np.pi, 3200)
    wave = np.sin(t) * 0.18
    for k in range(4):
        centre = (k + 0.55) * 2 * np.pi
        wave += 0.60 * np.exp(-((t - centre) ** 2) / 0.0035)
        wave -= 0.32 * np.exp(-((t - centre - 0.24) ** 2) / 0.005)
    xs = np.linspace(x0, x1, len(t))
    line = Line2D(xs, y_center + amplitude * wave, color=color, linewidth=1.8, alpha=alpha, zorder=1)
    line.set_transform(fig.transFigure)
    fig.add_artist(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-dir", default="manuscript/figures")
    parser.add_argument("--output", default="poster/poster_EN")
    parser.add_argument("--name", default="Zhengyu")
    parser.add_argument("--supervisor", default="[Supervisor]")
    parser.add_argument("--dpi", type=int, default=100)
    args = parser.parse_args()

    figures = Path(args.figure_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(W, H), dpi=args.dpi)
    fig.patch.set_facecolor(WHITE)

    # ---------------- banner ----------------
    rect(fig, 0, 0.845, 1.0, 0.155, BAND, z=0)
    rect(fig, 0, 0.845, 1.0, 0.0042, STYLE["aqua"]["edge"], z=1)
    ecg_line(fig, 0.52, 0.996, 0.900, 0.024, STYLE["aqua"]["fill"], 0.5)

    logo(fig, 0.046, 0.9225, 0.026)
    fig.text(0.082, 0.9470, "Nanyang Technological University", fontsize=31, fontweight="bold",
             color=WHITE, va="center", zorder=10)
    fig.text(0.082, 0.9100, "College of Engineering   ·   AI in Medicine   ·   MD6329 Deep Learning",
             fontsize=18, color=STYLE["aqua"]["fill"], va="center", zorder=10)
    fig.text(0.984, 0.9570, "2026", fontsize=44, fontweight="bold", color=WHITE,
             va="center", ha="right", zorder=10)
    fig.text(0.984, 0.9100, "Capstone Project Showcase", fontsize=16,
             color=STYLE["aqua"]["fill"], va="center", ha="right", zorder=10)

    # ---------------- title ----------------
    rect(fig, 0, 0.786, 1.0, 0.059, BAND_SOFT, z=2)
    fig.text(0.014, 0.8155,
             "Calibration Is Not Enough: A Three-Level Audit of Selective Prediction for Multi-Label ECG Classification",
             fontsize=26, fontweight="bold", color=INK, va="center", zorder=4)

    # ---------------- info strip ----------------
    rect(fig, 0, 0.742, 1.0, 0.044, BAND_PALE, z=2)
    rect(fig, 0, 0.7413, 1.0, 0.0010, CARD_EDGE, z=3)
    fig.text(0.014, 0.7640, args.name, fontsize=20, fontweight="bold", color=INK, va="center", zorder=4)
    fig.text(0.062, 0.7640, "·  MD6329 Deep Learning", fontsize=16, color=MUTED, va="center", zorder=4)
    fig.text(0.245, 0.7640, f"Supervisor: {args.supervisor}", fontsize=16, color=INK, va="center", zorder=4)
    fig.text(0.430, 0.7640, "PTB-XL v1.0.3  ·  21,799 records  ·  12 leads  ·  100 Hz",
             fontsize=16, color=INK, va="center", zorder=4)
    fig.text(0.984, 0.7640, "Code + interactive deck:  github.com/LEO-Jiang-ZY/ecg-selective-audit",
             fontsize=15, color=MUTED, va="center", ha="right", zorder=4)

    # =================== COLUMN 1 ===================
    x = COL_X[0]
    y = TOP
    c = card(fig, x, y, 0.105, "PROBLEM")
    text_block(fig, x + 0.012, c, [
        "Deployment is triage: issue the record,",
        "or defer it to a cardiologist.",
        "What matters is the error among",
        "issued records — not average AUROC.",
    ], size=17, leading=0.0146)
    y -= 0.105 + GAP

    c = card(fig, x, y, 0.315, "PROTOCOL")
    bottom = picture(fig, figures / "fig_pipeline.png", x + COL_W / 2, c, COL_W - 0.024, 0.196)
    text_block(fig, x + 0.012, bottom - 0.016, [
        "Trained once on folds 1–8.",
        "T, τ and the correctness head",
        "fitted on fold 9, applied once.",
    ], size=15.5, leading=0.0136, color=MUTED)
    y -= 0.315 + GAP

    c = card(fig, x, y, 0.172, "DESIGN")
    table(fig, x + 0.010, c, COL_W - 0.020, [
        ["Split", "Records", "Role"],
        ["Folds 1–8", "17,418", "train"],
        ["Fold 9", "2,183", "fit T, τ"],
        ["Fold 10", "2,198", "evaluate once"],
        ["Patients", "1,904", "bootstrap"],
    ], weights=[1.15, 1.0, 1.45], size=15.5, row_h=0.0182)
    y -= 0.165 + GAP

    h = y - BOTTOM
    c = card(fig, x, y, h, "THE GAP", fill="#FBFDFE")
    text_block(fig, x + 0.012, c, [
        "Calibration fixes the number",
        "attached to a prediction.",
        "Triage needs the ordering of",
        "records, which a monotone map",
        "barely changes.",
        "",
        "Measured: 0.0014 AURC, 0.25 pp error.",
    ], size=17, leading=0.0152)

    # =================== COLUMN 2 ===================
    x = COL_X[1]
    y = TOP
    c = card(fig, x, y, 0.165, "RQ1  ·  THREE NUMBERS")
    stat(fig, x + 0.012, c + 0.000, "3.5×", "ECE  0.0238 → 0.0068", color=BAND, size=26)
    stat(fig, x + 0.012, c - 0.046, "0.1900 → 0.1914", "AURC on fold 10", color=ACCENT, size=21)
    stat(fig, x + 0.012, c - 0.090, "30.98 → 31.23 %", "accepted error @ 80 %", color=ACCENT, size=21)
    y -= 0.172 + GAP

    c = card(fig, x, y, 0.270, "CALIBRATION vs FRONTIER")
    bottom = picture(fig, figures / "fig1.png", x + COL_W / 2, c, COL_W - 0.014, 0.180)
    text_block(fig, x + 0.012, bottom - 0.014, [
        "(a) reliability moves onto the diagonal.",
        "(b) the risk–coverage curve does not move.",
    ], size=15, leading=0.0132, color=MUTED)
    y -= 0.270 + GAP

    c = card(fig, x, y, 0.170, "THREE SEEDS")
    table(fig, x + 0.008, c, COL_W - 0.016, [
        ["Seed", "Cov %", "Base %", "Cal %", "Δ pp"],
        ["42", "79.62", "30.96", "31.14", "+0.18"],
        ["43", "78.30", "29.44", "29.81", "+0.37"],
        ["44", "80.66", "31.35", "31.81", "+0.46"],
        ["mean", "79.53", "30.58", "30.92", "+0.34"],
    ], weights=[0.9, 1.0, 1.0, 1.0, 0.9], size=15, row_h=0.0172, highlight_last=True)
    y -= 0.170 + GAP

    h = y - BOTTOM
    c = card(fig, x, y, h, "READING", fill="#FBFDFE")
    text_block(fig, x + 0.012, c, [
        "Same sign in every seed: the",
        "calibrated policy never lowers",
        "the accepted error here.",
        "AUROC moves 0.0004; coverage moves 2.4 pts.",
    ], size=16, leading=0.0146)

    # =================== COLUMN 3 ===================
    x = COL_X[2]
    y = TOP
    c = card(fig, x, y, 0.230, "SCORE")
    bottom = picture(fig, figures / "fig2.png", x + COL_W / 2, c, COL_W - 0.040, 0.150)
    text_block(fig, x + 0.012, bottom - 0.014, [
        "max-prob and all-correct beat",
        "entropy by 1.25 points @ 80 %;",
        "at a 15 % target, none helps.",
    ], size=15, leading=0.0132, color=MUTED)
    y -= 0.230 + GAP

    c = card(fig, x, y, 0.225, "ERROR BUDGET")
    bottom = picture(fig, figures / "fig3.png", x + COL_W / 2, c, COL_W - 0.040, 0.146)
    text_block(fig, x + 0.012, bottom - 0.014, [
        "Grey: what no ordering can fix.",
        "Peach: the whole score budget.",
    ], size=15, leading=0.0132, color=MUTED)
    y -= 0.225 + GAP

    c = card(fig, x, y, 0.105, "THE CEILING")
    fig.text(x + 0.012, c - 0.008, "6.5 pp", fontsize=34, fontweight="bold", color=ACCENT,
             va="top", ha="left", zorder=8)
    text_block(fig, x + 0.108, c - 0.004, [
        "is all any score can",
        "buy at 80 % coverage.",
    ], size=16, leading=0.0140)
    y -= 0.105 + GAP

    h = y - BOTTOM
    c = card(fig, x, y, h, "WHAT IS LEFT", fill="#FBFDFE")
    text_block(fig, x + 0.012, c, [
        "A perfect ordering still leaves",
        "23.6 % accepted error at 80 %.",
        "",
        "A learned correctness head recovers",
        "0.47 pp — inside the 1.0 pp seed",
        "spread, so: suggestive, not established.",
        "",
        "The lever is the decision unit, not the score.",
    ], size=16, leading=0.0146)

    # =================== COLUMN 4 ===================
    x = COL_X[3]
    y = TOP
    c = card(fig, x, y, 0.220, "GRANULARITY")
    bottom = picture(fig, figures / "fig5.png", x + COL_W / 2, c, COL_W - 0.046, 0.148)
    text_block(fig, x + 0.012, bottom - 0.014, [
        "Label-level halves label error,",
        "9.90 % → 5.61 %, but only 39.8 %",
        "of records can be issued in full.",
    ], size=15, leading=0.0132, color=MUTED)
    y -= 0.220 + GAP

    c = card(fig, x, y, 0.235, "SUBGROUPS")
    bottom = picture(fig, figures / "fig6.png", x + COL_W / 2, c, COL_W - 0.014, 0.166)
    text_block(fig, x + 0.012, bottom - 0.014, [
        "Age 16.3 % (18–44) vs 40.6 % (65–89):",
        "a 24-point gap, 20× any score gain.",
    ], size=15, leading=0.0132, color=MUTED)
    y -= 0.235 + GAP

    c = card(fig, x, y, 0.146, "TAKE-HOME")
    text_block(fig, x + 0.012, c, [
        "1   Calibration ≠ selective reliability.",
        "2   The score layer's budget is small.",
        "3   Granularity changes triage; score does not.",
        "4   Report frontier, target, unit.",
    ], size=15.5, leading=0.0190)
    y -= 0.146 + GAP

    h = y - BOTTOM
    c = card(fig, x, y, h, "LIMITS", fill="#FBFDFE")
    text_block(fig, x + 0.012, c, [
        "Single centre, one architecture family;",
        "thresholds tuned on all of fold 9; no",
        "external validation. Numbers are",
        "empirical, not conformal guarantees.",
    ], size=15, leading=0.0136, color=MUTED)

    # ---------------- footer ----------------
    rect(fig, 0, 0, 1.0, 0.018, BAND, z=2)
    fig.text(0.014, 0.009, "PTB-XL v1.0.3 (Wagner et al., Sci. Data 2020)  ·  ECGResNet1D, 51,429 parameters  ·  "
                           "patient-level bootstrap, 1,000 resamples",
             fontsize=12.5, color=WHITE, va="center", zorder=6)
    fig.text(0.984, 0.009, "Nanyang Technological University  ·  2026", fontsize=12.5,
             color=STYLE["aqua"]["fill"], va="center", ha="right", zorder=6)

    fig.savefig(out.with_suffix(".pdf"), facecolor=WHITE)
    fig.savefig(out.with_suffix(".png"), facecolor=WHITE, dpi=150)
    plt.close(fig)
    print("wrote", out.with_suffix(".pdf"), "and", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
