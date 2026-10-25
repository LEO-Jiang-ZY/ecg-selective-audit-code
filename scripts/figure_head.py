"""绘制 correctness head 与三种 post-hoc 分数的前沿对比。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ptbxl.selective import (
    CorrectnessHead,
    binary_entropy,
    record_features,
    risk_coverage_from_score,
)
from selective_head_insample import build_frame

from ptbxl.palette import GRID, INK, MUTED, STYLE  # noqa: E402
COLORS = {
    "entropy": ("#FBE4D0", "#C4834A"),
    "max_prob": ("#DCE8F6", "#3C6E9F"),
    "all_correct": ("#E0EEDD", "#5F8C5A"),
    "correctness_head": ("#E8E3F3", "#6E5F9E"),
}
LABELS = {
    "entropy": "entropy",
    "max_prob": "max-prob",
    "all_correct": "all-correct",
    "correctness_head": "correctness head",
}


def main() -> None:
    cfg = yaml.safe_load(Path("result/runs/seed_42/config.yaml").read_text(encoding="utf-8"))
    frame = build_frame(cfg)
    from ptbxl.split import patient_fold_split

    splits = patient_fold_split(frame, cfg["data"]["folds"])
    stored = np.load("result/analysis/selective/main_model_features.npz")

    def payload_of(name):
        return {
            "features": stored[f"{name}_features"],
            "logits": stored[f"{name}_logits"],
            "labels": stored[f"{name}_labels"],
            "frame": splits[name].reset_index(drop=True),
        }

    train, test = payload_of("train"), payload_of("test")
    X_train = record_features(train["features"].astype(float), train["logits"].astype(float), train["frame"])
    p_train = expit(train["logits"].astype(float))
    y_train = train["labels"].astype(float)
    target = np.all((p_train > 0.5) == (y_train > 0.5), axis=1).astype(int)
    head = CorrectnessHead(kind="gbdt", seed=42).fit_full(X_train, target)

    X = record_features(test["features"].astype(float), test["logits"].astype(float), test["frame"])
    y = test["labels"].astype(float)
    p = expit(test["logits"].astype(float))
    confident = np.maximum(p, 1 - p)
    scores = {
        "entropy": -binary_entropy(p).mean(axis=1),
        "max_prob": confident.mean(axis=1),
        "all_correct": np.prod(confident, axis=1),
        "correctness_head": head.score(X),
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=300, gridspec_kw={"width_ratios": [1.55, 1.0]})
    ax = axes[0]
    summary = {}
    for key, score in scores.items():
        curve = risk_coverage_from_score(y, p, score)
        from ptbxl.metrics import aurc

        a = float(aurc(curve["risk"], curve["coverage"]))
        summary[key] = a
        fill, edge = COLORS[key]
        ax.plot(curve["coverage"], 100 * curve["risk"], color=edge, linewidth=1.5,
                label=f"{LABELS[key]}  AURC={a:.4f}", zorder=4)
    ax.axvline(0.80, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
    ax.text(0.795, 3, "80%", fontsize=6.5, color=MUTED, rotation=90, va="bottom", ha="right")
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0, 45)
    ax.set_xlabel("coverage", fontsize=8, color=INK)
    ax.set_ylabel("accepted any-label error (%)", fontsize=8, color=INK)
    ax.set_title("(a) Risk–coverage on fold 10", fontsize=8.5, color=INK, pad=6)
    ax.legend(fontsize=6.2, frameon=False, loc="upper left")

    ax2 = axes[1]
    keys = list(scores)
    deltas = [100 * (summary[k] - summary["max_prob"]) for k in keys]
    positions = np.arange(len(keys))[::-1]
    for pos, key, delta in zip(positions, keys, deltas):
        fill, edge = COLORS[key]
        ax2.barh(pos, delta, height=0.55, color=fill, edgecolor=edge, linewidth=0.9, zorder=4)
        offset = 0.02 if delta >= 0 else -0.02
        ax2.text(delta + offset, pos, f"{delta:+.2f}", va="center",
                 ha="left" if delta >= 0 else "right", fontsize=6.8, color=INK)
    ax2.axvline(0, color=MUTED, linewidth=0.8, zorder=3)
    ax2.set_yticks(positions)
    ax2.set_yticklabels([LABELS[k] for k in keys], fontsize=7)
    ax2.set_xlim(-0.6, 0.6)
    ax2.set_xlabel("ΔAURC × 100 vs max-prob (negative = better)", fontsize=7.2, color=INK)
    ax2.set_title("(b) Difference against max-prob", fontsize=8.5, color=INK, pad=6)

    for a in axes:
        for spine in ["top", "right"]:
            a.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            a.spines[spine].set_color(MUTED)
            a.spines[spine].set_linewidth(0.6)
        a.tick_params(colors=INK, labelsize=7.5, width=0.6, length=3)
        a.grid(True, color=GRID, linewidth=0.6, axis="x" if a is ax2 else "both")
        a.set_axisbelow(True)

    fig.tight_layout(pad=0.6)
    out = Path("result/report/paper_figures/fig5_correctness_head.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", out)
    print(json.dumps({k: round(v, 4) for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
