"""生成会议 Poster（24 × 36 inch，纵向），输出 PDF 与 PNG。

三栏网格：每栏三张等高卡片，卡片顶部放标题，正文自标题下方开始，
图片按原始宽高比居中嵌入。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

INK = "#1F2933"
MUTED = "#5F6B7A"
CARD = "#F7F9FC"
EDGE = "#D5DFEA"
BLUE = "#185FA5"
BLUE_L = "#E3EFFA"
ORANGE = "#B9773F"
ORANGE_L = "#FBE9D9"
GREEN = "#4F7D4B"
GREEN_L = "#E2EEDF"
PURPLE = "#635594"
PURPLE_L = "#E9E4F4"

W, H = 24.0, 36.0

LEFT, MID, RIGHT = 0.028, 0.350, 0.672
COL_W = 0.300
CARD_H = 0.196
GAP = 0.015
TOPS = [0.906, 0.906 - CARD_H - GAP, 0.906 - 2 * (CARD_H + GAP)]
BODY_OFFSET = 0.062
TITLE_OFFSET = 0.031


def card(fig, x, top, h, title, accent=BLUE, fill="white"):
    y = top - h
    fig.patches.append(FancyBboxPatch(
        (x, y), COL_W, h,
        boxstyle="round,pad=0.0035,rounding_size=0.010",
        linewidth=2.2, edgecolor="#C2D2E2", facecolor=fill, zorder=1,
    ))
    fig.patches.append(Rectangle((x + 0.010, top - 0.047), 0.009, 0.026,
                                 facecolor=accent, edgecolor="none", zorder=3))
    fig.text(x + 0.028, top - TITLE_OFFSET, title, fontsize=27, fontweight="bold",
             color=INK, va="center", ha="left", zorder=4)
    return top - BODY_OFFSET


def body(fig, x, y, lines, size=20, leading=0.0158, color=INK):
    for i, line in enumerate(lines):
        if not line:
            continue
        fig.text(x, y - i * leading, line, fontsize=size, color=color,
                 va="top", ha="left", zorder=4, linespacing=1.25)


def stat(fig, x, y, value, label, color=BLUE, size=33, label_size=15):
    fig.text(x, y, value, fontsize=size, fontweight="bold", color=color,
             va="top", ha="left", zorder=4)
    fig.text(x, y - 0.0225, label, fontsize=label_size, color=MUTED,
             va="top", ha="left", zorder=4)
    return y - 0.0225 - 0.0135


def picture(fig, path, x_center, y_top, max_w, max_h):
    path = Path(path)
    if not path.exists():
        fig.text(x_center, y_top - max_h / 2, f"[missing {path.name}]", fontsize=16,
                 color="#B00020", ha="center", va="center")
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
    axes = fig.add_axes([x_center - w / 2, y_top - h, w, h])
    axes.imshow(image)
    axes.axis("off")
    return y_top - h


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-dir", default="manuscript/figures")
    parser.add_argument("--output", default="poster/calibration_is_not_enough")
    args = parser.parse_args()
    figures = Path(args.figure_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(W, H), dpi=150)
    fig.patch.set_facecolor("white")

    # ---- title band ----
    fig.patches.append(Rectangle((0, 0.9375), 1.0, 0.0625, facecolor=BLUE_L, edgecolor="none", zorder=0))
    fig.patches.append(Rectangle((0, 0.9362), 1.0, 0.0016, facecolor=BLUE, edgecolor="none", zorder=1))
    fig.text(0.028, 0.9785, "Calibration Is Not Enough", fontsize=58, fontweight="bold", color=INK, va="center")
    fig.text(0.028, 0.9585, "A three-level audit of selective prediction for multi-label ECG classification",
             fontsize=26, color=MUTED, va="center")
    fig.text(0.028, 0.9455, "Zhengyu  ·  MD6329 Deep Learning  ·  Nanyang Technological University",
             fontsize=16, color=MUTED, va="center")

    fig.patches.append(FancyBboxPatch((0.735, 0.9445), 0.235, 0.0285,
                                      boxstyle="round,pad=0.003,rounding_size=0.005",
                                      facecolor="white", edgecolor=BLUE, linewidth=1.5, zorder=2))
    fig.text(0.8525, 0.9645, "ECE improves 3.5×", fontsize=19, fontweight="bold", color=BLUE,
             ha="center", va="center", zorder=4)
    fig.text(0.8525, 0.9530, "the frontier does not move", fontsize=15, color=MUTED,
             ha="center", va="center", zorder=4)

    # ================= LEFT =================
    y = card(fig, LEFT, TOPS[0], CARD_H, "The clinical question", BLUE)
    body(fig, LEFT + 0.016, y, [
        "AI-ECG is a triage layer: sign the record,",
        "or defer it to a cardiologist. What matters",
        "is the error rate among the records the",
        "model chose to issue — not average AUROC.",
        "",
        "Papers report calibration (ECE) and read it",
        "as evidence that confidence can be trusted",
        "to decide when to abstain. We test that.",
    ], size=20)

    y = card(fig, LEFT, TOPS[1], CARD_H, "RQ1 — probabilities ≠ decisions", ORANGE)
    cursor = stat(fig, LEFT + 0.018, y, "0.0238 → 0.0068", "ECE, fold 10", ORANGE)
    cursor = stat(fig, LEFT + 0.018, cursor - 0.010, "0.1900 → 0.1914", "AURC, entropy score", ORANGE)
    cursor = stat(fig, LEFT + 0.018, cursor - 0.010, "30.98 → 31.23", "accepted error @ 80% coverage (%)", ORANGE)
    body(fig, LEFT + 0.018, cursor - 0.012, [
        "Temperature scaling repairs the probabilities",
        "and leaves the accepted set where it was.",
    ], size=19)

    y = card(fig, LEFT, TOPS[2], CARD_H, "Same sign in three seeds", GREEN)
    picture(fig, figures / "fig1.png", LEFT + COL_W / 2, y + 0.004, COL_W - 0.040, 0.060)
    body(fig, LEFT + 0.016, y - 0.062, [
        "(a) reliability returns to the diagonal;",
        "(b) risk–coverage curves stay superimposed.",
        "",
        "Δ vs. uncalibrated, matched coverage:",
        "+0.18 · +0.37 · +0.46 points  (mean +0.34 ± 0.14)",
        "three of three positive, macro AUROC ±0.0004.",
    ], size=18.5, leading=0.0152)

    # ================= MIDDLE =================
    y = card(fig, MID, TOPS[0], CARD_H, "A three-level audit", BLUE)
    body(fig, MID + 0.016, y, [
        "Post-hoc interventions, ordered by how",
        "invasive they are.",
        "",
        "L1  calibration",
        "      change the logit → probability map",
        "L2  score",
        "      change the statistic that decides",
        "L3  granularity",
        "      change the unit that is accepted",
    ], size=19.5, leading=0.0152)

    y = card(fig, MID, TOPS[1], CARD_H, "RQ2 — the budget is 6.9 points", ORANGE)
    cursor = stat(fig, MID + 0.018, y, "6.9 pp", "entire room for any abstention score", ORANGE, size=40)
    picture(fig, figures / "fig3.png", MID + COL_W / 2, cursor - 0.014, COL_W - 0.055, 0.086)
    body(fig, MID + 0.016, cursor - 0.104, [
        "At 80% coverage a perfect ordering still errs",
        "on 23.1%; the model errs on 30.0%.",
    ], size=19)

    y = card(fig, MID, TOPS[2], CARD_H, "Scoring is nearly exhausted", PURPLE)
    body(fig, MID + 0.016, y, [
        "entropy → max-prob      recovers 1.1 of 6.9",
        "learned correctness head   adds 0.4, within",
        "      patient-bootstrap noise",
        "",
        "Capacity is not the constraint: 16 M",
        "parameters is 0.0075 worse in AURC than",
        "the 51 k model on the same recipe.",
        "",
        "The binding limit sits in discrimination",
        "and label ambiguity, not in the score.",
    ], size=19, leading=0.0152)

    # ================= RIGHT =================
    y = card(fig, RIGHT, TOPS[0], CARD_H, "RQ3 — granularity beats scoring", GREEN)
    picture(fig, figures / "fig5.png", RIGHT + COL_W / 2, y + 0.004, COL_W - 0.040, 0.062)
    body(fig, RIGHT + 0.016, y - 0.064, [
        "record-level   79.6% label coverage   9.90% error",
        "label-level     76.4% label coverage   5.61% error",
        "",
        "Half the label error — but only 39.8% of",
        "records can then be issued untouched, and",
        "1.18 labels per record still need a reader.",
    ], size=18, leading=0.0150)

    y = card(fig, RIGHT, TOPS[1], CARD_H, "Who carries the error", ORANGE)
    picture(fig, figures / "fig6.png", RIGHT + COL_W / 2, y + 0.004, COL_W - 0.045, 0.058)
    body(fig, RIGHT + 0.016, y - 0.060, [
        "Age 16.3% (18–44) vs 40.6% (65–89):",
        "a 24.3-point gap, twenty times the effect",
        "of changing the abstention score.",
        "",
        "Device TVD 0.326 — the test fold is itself",
        "a shifted set, not an i.i.d. sample.",
    ], size=19, leading=0.0152)

    y = card(fig, RIGHT, TOPS[2], CARD_H, "Take-home", BLUE, fill=BLUE_L)
    body(fig, RIGHT + 0.016, y, [
        "Report the risk–coverage frontier under a",
        "declared error target, together with the",
        "abstention granularity.",
        "",
        "ECE is a property of probabilities, not a",
        "certificate for triage. Calibration and",
        "selective reliability are separate targets.",
    ], size=19.5, leading=0.0158)

    # ================= FOOTER =================
    fig.patches.append(Rectangle((0, 0.0), 1.0, 0.272, facecolor="#FBFCFE", edgecolor="none", zorder=0))
    fig.patches.append(Rectangle((0, 0.2705), 1.0, 0.0014, facecolor=EDGE, edgecolor="none", zorder=1))

    fig.patches.append(FancyBboxPatch((LEFT, 0.026), 0.944, 0.222,
                                      boxstyle="round,pad=0.0035,rounding_size=0.010",
                                      linewidth=1.3, edgecolor=EDGE, facecolor="white", zorder=1))
    fig.patches.append(Rectangle((LEFT + 0.010, 0.216), 0.009, 0.024, facecolor=MUTED,
                                 edgecolor="none", zorder=3))
    fig.text(LEFT + 0.028, 0.228, "Data, protocol, and references", fontsize=25,
             fontweight="bold", color=INK, va="center", zorder=4)

    body(fig, LEFT + 0.016, 0.188, [
        "PTB-XL v1.0.3 — 21,799 recordings / 18,869 patients, 100 Hz, 12 leads, 10 s. Official patient-level folds: 1–8 training (17,418), fold 9 for every fitting",
        "decision (2,183), fold 10 evaluated once (2,198 records / 1,904 patients). Model: three-block 1D ResNet, 51,429 parameters. Thresholds and temperatures",
        "fitted on fold 9 only. Bootstrap intervals resample patients, not records. Every number regenerates from the project scripts.",
    ], size=16, leading=0.0165, color=MUTED)

    body(fig, LEFT + 0.016, 0.128, [
        "[1] Wagner et al. PTB-XL, a large publicly available ECG dataset. Sci Data 2020.    [2] Guo et al. On Calibration of Modern Neural Networks. ICML 2017.",
        "[3] Geifman & El-Yaniv. Selective Classification for Deep Neural Networks. NeurIPS 2017.    [4] Angelopoulos et al. Conformal Risk Control. ICLR 2023.",
        "[5] Rabanser & Papernot. What Does It Take to Build a Performant Selective Classifier? NeurIPS 2025.    [6] Strodthoff et al. Deep Learning for ECG",
        "Analysis. IEEE JBHI 2021.    [7] EL ALLAM & Hamlich. Per-Class Conformal Risk Control for Multi-Label ECG Classification. Physiol Meas 2026.",
    ], size=15, leading=0.0155, color=MUTED)

    body(fig, LEFT + 0.016, 0.045, [
        "Code: train.py → fit_calibration.py → evaluate.py; analysis by decompose_gap.py and selective_head_insample.py; figures by paper_figures.py, figure_head.py.",
    ], size=15, leading=0.0150, color=MUTED)

    for suffix, dpi in (("pdf", 300), ("png", 150)):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
