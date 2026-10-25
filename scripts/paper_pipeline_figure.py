"""论文用总流程图：数据 → 模型 → 校准 → 三级审计 → 决策输出。

与既有结果图统一采用浅色填充 + 深色描边的配色，保证 A4 打印不糊。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ptbxl.palette import (  # noqa: E402
    AUDIT_EDGE,
    AUDIT_FILL,
    INK,
    MUTED,
    OUT_EDGE,
    OUT_FILL,
    PIPE_EDGE as BOX_EDGE,
    PIPE_FILL as BOX_FILL,
)

FONT = 8.6
SMALL = 7.0


def box(ax, x, y, w, h, title, sub=None, fill=BOX_FILL, edge=BOX_EDGE, title_size=FONT):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.6,rounding_size=1.4",
            linewidth=1.1, edgecolor=edge, facecolor=fill, zorder=2,
        )
    )
    if sub:
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center",
                fontsize=title_size, color=INK, zorder=3, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=SMALL, color=MUTED, zorder=3, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, color=INK, zorder=3, fontweight="bold",
                linespacing=1.35)


def arrow(ax, start, end, color=MUTED, style="-|>", width=1.0, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle=style, mutation_scale=11,
            linewidth=width, color=color, zorder=1,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2, shrinkB=2,
        )
    )


def build(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.35), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---- 第一行：数据与模型 ----
    h = 20
    y1 = 70
    box(ax, 1.5, y1, 20, h, "PTB-XL v1.0.3", "12 × 1,000 @ 100 Hz\n21,799 records")
    box(ax, 26.5, y1, 20, h, "ECGResNet1D", "51,429 parameters\ntrained on folds 1–8")
    box(ax, 51.5, y1, 18, h, "5 logits", "NORM · MI · STTC\nCD · HYP")
    box(ax, 74.5, y1, 24, h, "Calibrated scores", "p = σ(z / T)\nT fitted on fold 9", fill=AUDIT_FILL, edge=AUDIT_EDGE)

    for x0, x1 in [(21.5, 26.5), (46.5, 51.5), (69.5, 74.5)]:
        arrow(ax, (x0, y1 + h / 2), (x1, y1 + h / 2))

    # ---- 第二行：三级审计（作为单一容器，避免与上游方框一一对应的误读） ----
    h2 = 24
    y2 = 33
    ax.add_patch(
        FancyBboxPatch(
            (1.5, y2), 97, h2,
            boxstyle="round,pad=0.6,rounding_size=1.4",
            linewidth=1.1, edgecolor=AUDIT_EDGE, facecolor=AUDIT_FILL, zorder=2,
        )
    )
    ax.text(3.4, y2 + h2 - 3.4,
            "Three-level audit — every parameter is fitted on fold 9 and applied once to fold 10",
            fontsize=SMALL, color=MUTED, ha="left", va="center", style="italic", zorder=3)

    columns = [
        (18.5, "L1 · calibration", "reliability diagram\nECE, NLL, Brier"),
        (50.5, "L2 · abstention score", "entropy | max-prob\nall-correct probability"),
        (82.5, "L3 · granularity", "per record\nper label"),
    ]
    for cx, title, sub in columns:
        ax.text(cx, y2 + 11.4, title, ha="center", va="center",
                fontsize=FONT, color=INK, fontweight="bold", zorder=3)
        ax.text(cx, y2 + 5.6, sub, ha="center", va="center",
                fontsize=SMALL, color=MUTED, zorder=3, linespacing=1.4)
    for vx in (34.5, 66.5):
        ax.plot([vx, vx], [y2 + 2.2, y2 + h2 - 6.4], color=AUDIT_EDGE,
                linewidth=0.6, alpha=0.45, zorder=2)

    arrow(ax, (86.5, y1), (86.5, y2 + h2 + 1.2), color=AUDIT_EDGE)

    # ---- 第三行：决策与度量 ----
    h3 = 17
    y3 = 5
    box(ax, 1.5, y3, 44, h3, "Accept / defer at threshold τ",
        "coverage c  ·  accepted any-label error R(c)", fill=OUT_FILL, edge=OUT_EDGE)
    box(ax, 52.5, y3, 46, h3, "Reported quantities",
        "risk–coverage frontier, AURC, coverage at fixed error\noracle ≤ ensemble reference ≤ observed", fill=OUT_FILL, edge=OUT_EDGE)

    arrow(ax, (23.5, y2), (23.5, y3 + h3 + 1.2), color=OUT_EDGE)
    arrow(ax, (45.5, y3 + h3 / 2), (52.5, y3 + h3 / 2), color=OUT_EDGE)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", path, path.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    build(Path("result/report/paper_figures/fig0_pipeline.png"))
    build(Path("manuscript/figures/fig_pipeline.png"))
