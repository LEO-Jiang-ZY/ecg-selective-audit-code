"""生成训练曲线与 risk-coverage 图（浅色主题，英文标签避免字体缺字）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def plot_training(history, output: Path):
    epochs = [row["epoch"] for row in history]
    train = [row["train_loss"] for row in history]
    val = [row["validation_loss"] for row in history]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160)
    ax.plot(epochs, train, color="#185FA5", linewidth=1.4, label="train")
    ax.plot(epochs, val, color="#BA7517", linewidth=1.4, linestyle="--", label="validation")
    best = int(np.argmin(val))
    ax.scatter([epochs[best]], [val[best]], color="#A32D2D", s=18, zorder=3)
    ax.annotate(f"best epoch {epochs[best]}", (epochs[best], val[best]), textcoords="offset points", xytext=(6, 8), fontsize=8, color="#A32D2D")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("Training and validation loss")
    ax.grid(True, color="#E5E5E5", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def plot_risk_coverage(evaluation, output: Path):
    units = [evaluation.get("primary_error_unit", "any_label")] + [
        unit for unit in evaluation.get("secondary_error_units", []) if unit != evaluation.get("primary_error_unit")
    ]
    fig, axes = plt.subplots(1, len(units), figsize=(5.2 * len(units), 3.6), dpi=160, squeeze=False)
    for ax, unit in zip(axes[0], units):
        base = evaluation.get("baseline", {}).get("risk_coverage", {}).get(unit)
        cal = evaluation.get("risk_coverage", {}).get(unit)
        if base:
            ax.plot(base["coverage"], base["risk"], color="#5F5E5A", linewidth=1.4, label="ERM baseline")
        if cal:
            ax.plot(cal["coverage"], cal["risk"], color="#185FA5", linewidth=1.4, linestyle="--", label="calibrated + abstention")
        ax.axvline(0.8, color="#BA7517", linewidth=0.8, linestyle=":")
        ax.text(0.805, ax.get_ylim()[1] * 0.92, "80%", fontsize=8, color="#854F0B")
        ax.set_xlabel("coverage")
        ax.set_ylabel("accepted-set error")
        ax.set_title(f"Risk-coverage ({unit})")
        ax.set_xlim(0.2, 1.0)
        ax.grid(True, color="#E5E5E5", linewidth=0.6)
        ax.legend(frameon=False, fontsize=8)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", default="result/results/evaluation.json")
    parser.add_argument("--history", default="result/results/history.json")
    parser.add_argument("--output-dir", default="result/report/figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_training(load(Path(args.history)), output_dir / "training_curve.png")
    plot_risk_coverage(load(Path(args.evaluation)), output_dir / "risk_coverage.png")
    print(f"已写入 {output_dir / 'training_curve.png'}")
    print(f"已写入 {output_dir / 'risk_coverage.png'}")


if __name__ == "__main__":
    main()
