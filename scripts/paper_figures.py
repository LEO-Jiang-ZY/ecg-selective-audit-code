"""生成论文用的核心图件：校准-前沿解耦、分数前沿、亚组误差、拒答粒度。

输出目录默认为 result/report/paper_figures/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.special import expit
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptbxl.inference import calibrated_logits, load_run
from ptbxl.metrics import (
    aurc,
    coverage_at_error,
    error_vector,
    expected_calibration_error,
    predictive_entropy,
)
from ptbxl.utils import jsonable

from ptbxl.palette import GRID, INK, MUTED, STYLE  # noqa: E402


def _style_axes(ax: plt.Axes) -> None:
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(MUTED)
        ax.spines[spine].set_linewidth(0.6)
    ax.tick_params(colors=INK, labelsize=7.5, width=0.6, length=3)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    centers, accuracy, confidence, counts = [], [], [], []
    for b in range(n_bins):
        mask = index == b
        if mask.sum() == 0:
            continue
        centers.append(0.5 * (edges[b] + edges[b + 1]))
        accuracy.append(float(y[mask].mean()))
        confidence.append(float(p[mask].mean()))
        counts.append(int(mask.sum()))
    return np.array(centers), np.array(accuracy), np.array(confidence), np.array(counts)


def clopper_pearson(errors: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    low = 0.0 if errors == 0 else float(beta.ppf(alpha / 2, errors, n - errors + 1))
    high = 1.0 if errors == n else float(beta.ppf(1 - alpha / 2, errors + 1, n - errors))
    return low, high


def risk_coverage_from_score(y: np.ndarray, p: np.ndarray, score: np.ndarray, n_points: int = 200) -> dict:
    """score 越小越可信；返回等距 coverage 上的 selective risk。"""
    order = np.argsort(score)
    errors = error_vector(y, p, "any_label", 0.5, from_logits=False)[order]
    cumulative = np.cumsum(errors)
    coverages = np.linspace(1 / len(y), 1.0, n_points)
    risks = np.array(
        [
            cumulative[max(1, int(np.ceil(c * len(y)))) - 1] / max(1, int(np.ceil(c * len(y))))
            for c in coverages
        ]
    )
    return {"coverage": coverages, "risk": risks}


def fig_calibration_vs_frontier(y_test, logits_raw, logits_cal, out: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=300)

    # (a) reliability
    ax = axes[0]
    ax.plot([0, 1], [0, 1], linestyle=(0, (4, 3)), color=MUTED, linewidth=0.9, label="ideal", zorder=2)
    payload = {}
    for key, logits, color in [
        ("uncalibrated", logits_raw, "orange"),
        ("calibrated", logits_cal, "blue"),
    ]:
        centers, acc, conf, counts = reliability_curve(y_test, expit(logits))
        ax.plot(conf, acc, marker="o", markersize=3.2, linewidth=1.2,
                color=STYLE[color]["edge"], markerfacecolor="white", markeredgewidth=0.9,
                label=key, zorder=4)
        payload[f"ece_{key}"] = expected_calibration_error(y_test, expit(logits), from_logits=False)
        payload[f"bins_{key}"] = len(centers)
    ax.set_xlabel("predicted probability", fontsize=8, color=INK)
    ax.set_ylabel("observed frequency", fontsize=8, color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("(a) Reliability on fold 10", fontsize=8.5, color=INK, pad=6)
    ax.text(0.03, 0.93,
            f"ECE  {payload['ece_uncalibrated']:.4f} → {payload['ece_calibrated']:.4f}",
            fontsize=7, color=INK, va="top")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    _style_axes(ax)

    # (b) risk-coverage
    ax = axes[1]
    for key, logits, color in [
        ("ERM", logits_raw, "orange"),
        ("calibrated", logits_cal, "blue"),
    ]:
        probs = expit(logits)
        curve = risk_coverage_from_score(y_test, probs, predictive_entropy(probs, from_logits=False))
        a = aurc(curve["risk"], curve["coverage"])
        payload[f"aurc_{key}"] = a
        payload[f"risk_at_80_{key}"] = float(np.interp(0.80, curve["coverage"], curve["risk"]))
        ax.plot(curve["coverage"], 100 * curve["risk"], linewidth=1.5,
                color=STYLE[color]["edge"], label=f"{key}  AURC={a:.4f}", zorder=4)
    ax.axvline(0.80, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)
    ax.text(0.795, 3, "target 80%", fontsize=6.5, color=MUTED, rotation=90, va="bottom", ha="right")
    ax.set_xlabel("coverage (fraction auto-issued)", fontsize=8, color=INK)
    ax.set_ylabel("accepted any-label error (%)", fontsize=8, color=INK)
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0, 45)
    ax.set_title("(b) Selective risk on fold 10", fontsize=8.5, color=INK, pad=6)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    _style_axes(ax)

    fig.tight_layout(pad=0.6)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return payload


def fig_score_frontier(y_test, logits_cal, out: Path) -> dict:
    p = expit(logits_cal)
    scores = {
        "entropy": (predictive_entropy(p, from_logits=False), "orange"),
        "max-prob": (-np.maximum(p, 1 - p).mean(axis=1), "blue"),
        "all-correct": (-np.prod(np.maximum(p, 1 - p), axis=1), "green"),
    }
    fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=300)
    payload = {}
    for name, (score, color) in scores.items():
        curve = risk_coverage_from_score(y_test, p, score)
        a = aurc(curve["risk"], curve["coverage"])
        payload[name] = {
            "aurc": a,
            "coverage_at_15": coverage_at_error(curve["risk"], curve["coverage"], 0.15),
            "coverage_at_20": coverage_at_error(curve["risk"], curve["coverage"], 0.20),
            "risk_at_80": float(np.interp(0.80, curve["coverage"], curve["risk"])),
        }
        ax.plot(curve["coverage"], 100 * curve["risk"], linewidth=1.6,
                color=STYLE[color]["edge"], label=f"{name}  AURC={a:.4f}", zorder=4)
    for level in (15, 20):
        ax.axhline(level, color=MUTED, linewidth=0.7, linestyle=(0, (2, 2)), zorder=1)
        ax.text(0.205, level + 0.7, f"{level}% error", fontsize=6.5, color=MUTED)
    ax.set_xlabel("coverage", fontsize=8, color=INK)
    ax.set_ylabel("accepted any-label error (%)", fontsize=8, color=INK)
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0, 45)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    _style_axes(ax)
    fig.tight_layout(pad=0.5)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return payload


def fig_subgroup(evaluation: dict, out: Path) -> None:
    panels = [
        ("age_group", "Age group", ["18-44", "45-64", "65-89"]),
        ("quality_any", "Signal quality", ["clean", "noise"]),
        ("sex_label", "Sex", ["female", "male"]),
        ("device_group", "Device", ["CS-12 E", "CS-12", "AT-6 C 5.5", "AT-6 6", "AT-60 3"]),
    ]
    overall = evaluation["accepted"]["any_label"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4), dpi=300, sharex=True)
    max_len = max(len(order) for _, _, order in panels)
    for ax, (column, title, order) in zip(axes.ravel(), panels):
        rows = {row["group"]: row for row in evaluation["subgroups"][column]}
        labels, errors, lows, highs, sizes = [], [], [], [], []
        for group in order:
            row = rows.get(group)
            if row is None or row["accepted_error"] is None:
                continue
            n = int(row["n"])
            errors_count = int(round(row["accepted_error"] * n))
            low, high = clopper_pearson(errors_count, n)
            labels.append(f"{group} (n={n})")
            errors.append(100 * row["accepted_error"])
            lows.append(100 * max(low, 0.0))
            highs.append(100 * high)
            sizes.append(n)
        offset = (max_len - 1) / 2
        ypos = offset - np.arange(len(labels))
        for y, e, lo, hi in zip(ypos, errors, lows, highs):
            ax.plot([lo, hi], [y, y], color=STYLE["blue"]["edge"], linewidth=1.1, solid_capstyle="butt", zorder=3)
            ax.plot([e], [y], marker="o", markersize=4.2, color=STYLE["blue"]["edge"],
                    markerfacecolor=STYLE["blue"]["fill"], markeredgewidth=1.0, zorder=4)
        ax.axvline(100 * overall, color=STYLE["orange"]["edge"], linewidth=0.9,
                   linestyle=(0, (4, 3)), zorder=2)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=6.6)
        ax.set_title(title, fontsize=8, color=INK, pad=3, loc="left")
        ax.set_xlim(0, 55)
        ax.set_ylim(-offset - 0.7, offset + 0.7)
        _style_axes(ax)
    for ax in axes[1]:
        ax.set_xlabel("accepted any-label error (%)", fontsize=7.5, color=INK)
    handles = [
        Line2D([0], [0], marker="o", color=STYLE["blue"]["edge"], linestyle="-", markersize=4.2,
               markerfacecolor=STYLE["blue"]["fill"], label="subgroup (95% CI)"),
        Line2D([0], [0], color=STYLE["orange"]["edge"], linestyle=(0, (4, 3)), linewidth=0.9,
               label=f"overall {100 * overall:.1f}%"),
    ]
    fig.legend(handles=handles, fontsize=6.5, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(pad=0.6, rect=(0, 0.03, 1, 1))
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def fig_granularity(evaluation: dict, per_label_csv: Path, out: Path) -> None:
    import pandas as pd

    table = pd.read_csv(per_label_csv)
    fig, ax = plt.subplots(figsize=(4.4, 3.2), dpi=300)
    for _, row in table.iterrows():
        ax.scatter(100 * row["label_coverage"], 100 * row["label_error"],
                   s=40 + 320 * row["full_accept_rate"], facecolor=STYLE["green"]["fill"],
                   edgecolor=STYLE["green"]["edge"], linewidth=1.0, zorder=4)
        ax.annotate(f"{row['scheme']}\nfull-accept {100 * row['full_accept_rate']:.0f}%",
                    (100 * row["label_coverage"], 100 * row["label_error"]),
                    textcoords="offset points", xytext=(8, -12), fontsize=6.2, color=INK)
    record_x = 100 * evaluation["accepted"]["coverage"]
    record_y = 100 * evaluation["accepted"]["macro_label"]
    ax.scatter([record_x], [record_y], marker="D", s=46, facecolor=STYLE["orange"]["fill"],
               edgecolor=STYLE["orange"]["edge"], linewidth=1.1, zorder=5)
    ax.annotate("record-level\n(entropy, 80%)", (record_x, record_y), textcoords="offset points",
                xytext=(-6, 10), fontsize=6.2, color=INK, ha="right")
    ax.set_xlabel("label-level coverage (%)", fontsize=8, color=INK)
    ax.set_ylabel("label-level error (%)", fontsize=8, color=INK)
    ax.set_xlim(50, 88)
    ax.set_ylim(2, 13)
    _style_axes(ax)
    fig.tight_layout(pad=0.5)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="result/runs/seed_42/config.yaml")
    parser.add_argument("--calibration", default="result/runs/seed_42/calibration.json")
    parser.add_argument("--evaluation", default="result/runs/seed_42/evaluation.json")
    parser.add_argument("--per-label", default="result/analysis/per_label_abstention.summary.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="result/report/paper_figures")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test = load_run(args.config, "test", device=args.device)
    logits_raw = test["logits"].astype(float)
    logits_cal, _ = calibrated_logits(logits_raw, args.calibration)
    y_test = test["labels"].astype(float)

    payload = {"n_test": int(len(y_test))}
    payload["fig1"] = fig_calibration_vs_frontier(y_test, logits_raw, logits_cal, out_dir / "fig1_calibration_vs_frontier.png")
    payload["fig2"] = fig_score_frontier(y_test, logits_cal, out_dir / "fig2_score_frontier.png")

    evaluation = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    fig_subgroup(evaluation, out_dir / "fig3_subgroup_error.png")
    fig_granularity(evaluation, Path(args.per_label), out_dir / "fig4_abstention_granularity.png")

    (out_dir / "figure_metrics.json").write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for key, value in payload.items():
        if isinstance(value, dict):
            print(key, {k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in value.items()})
    print("written to", out_dir)


if __name__ == "__main__":
    main()
