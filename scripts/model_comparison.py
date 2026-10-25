"""对比 ResNet1D / ResUNet / ResNet50 / Transformer 的 accuracy、AURC 与 coverage。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptbxl.inference import calibrated_logits, load_run
from ptbxl.metrics import error_vector

ALPHAS = (0.10, 0.15, 0.20, 0.25)
COLORS = {
    "resnet1d": "#5F5E5A",
    "resunet1d": "#185FA5",
    "resnet50_1d": "#BA7517",
    "transformer1d": "#0F6E56",
}
LABELS = {
    "resnet1d": "ResNet1D (current)",
    "resunet1d": "ResUNet1D",
    "resnet50_1d": "ResNet50-1D",
    "transformer1d": "Transformer1D",
}


def risk_curve(scores: np.ndarray, errors: np.ndarray, n_points: int = 200):
    order = np.argsort(-scores)
    sorted_errors = errors[order]
    n = len(errors)
    coverages = np.linspace(1 / n, 1.0, min(n, n_points))
    counts = np.maximum(1, np.ceil(coverages * n).astype(int))
    cumulative = np.cumsum(sorted_errors)
    risks = cumulative[counts - 1] / counts
    return coverages, risks


def coverage_at_error(coverages: np.ndarray, risks: np.ndarray, alpha: float) -> float:
    valid = np.where(risks <= alpha)[0]
    return float(coverages[valid[-1]]) if valid.size else float("nan")


def evaluate_run(name: str, config_path: str, device: str = "cpu") -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    run = load_run(config_path, "test", device=device)
    calibration = json.loads(Path(cfg["calibration"]["output"]).read_text(encoding="utf-8"))
    logits, _ = calibrated_logits(run["logits"].astype(float), cfg["calibration"]["output"])
    labels = run["labels"].astype(float)
    probabilities = expit(logits)
    scores = np.maximum(probabilities, 1 - probabilities).mean(axis=1)
    errors = error_vector(labels, logits, "any_label")
    coverages, risks = risk_curve(scores, errors)
    result = {
        "model": name,
        "label": LABELS[name],
        "params": int(sum(p.numel() for p in run["model"].parameters())),
        "temperature": float(calibration["temperature"]),
        "accuracy_exact": float(1 - errors.mean()),
        "label_accuracy": float(((probabilities >= 0.5).astype(int) == labels).mean()),
        "aurc": float(np.trapz(risks, coverages)),
        "risk@80": float(np.interp(0.8, coverages, risks)),
        "coverages": coverages,
        "risks": risks,
        "n_test": int(len(labels)),
    }
    for alpha in ALPHAS:
        result[f"coverage@{alpha:.2f}"] = coverage_at_error(coverages, risks, alpha)
    return result


def plot_individual(result: dict, output: Path):
    coverages, risks = result["coverages"], result["risks"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
    ax.plot(coverages, risks, color=COLORS[result["model"]], linewidth=1.6, label="risk-coverage")
    for alpha in ALPHAS:
        value = result[f"coverage@{alpha:.2f}"]
        if not np.isfinite(value):
            continue
        index = int(np.argmin(np.abs(coverages - value)))
        accuracy = 1 - risks[index]
        ax.scatter([coverages[index]], [risks[index]], color=COLORS[result["model"]], s=26, zorder=3)
        ax.annotate(
            f"cov {value*100:.1f}%\nacc {accuracy*100:.1f}%",
            (coverages[index], risks[index]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=7,
            color="#2C2C2A",
        )
        ax.axhline(alpha, color="#E5E5E5", linewidth=0.6, linestyle=":")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, max(0.45, float(risks.max()) * 1.05))
    ax.set_xlabel("coverage")
    ax.set_ylabel("accepted any-label error")
    ax.set_title(
        f"{result['label']}  |  Accuracy={result['accuracy_exact']*100:.1f}%  "
        f"AURC={result['aurc']:.4f}  Params={result['params']:,}",
        fontsize=9,
    )
    ax.grid(True, color="#E5E5E5", linewidth=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def plot_combined(results: list[dict], output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=160)
    for result in results:
        axes[0].plot(result["coverages"], result["risks"], color=COLORS[result["model"]], linewidth=1.5, label=result["label"])
        value = result["coverage@0.20"]
        if np.isfinite(value):
            index = int(np.argmin(np.abs(result["coverages"] - value)))
            axes[0].scatter([result["coverages"][index]], [result["risks"][index]], color=COLORS[result["model"]], s=22, zorder=3)
    axes[0].axvline(0.8, color="#BA7517", linewidth=0.8, linestyle=":")
    axes[0].set_xlabel("coverage")
    axes[0].set_ylabel("accepted any-label error")
    axes[0].set_title("Risk-coverage across architectures")
    axes[0].grid(True, color="#E5E5E5", linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=8)

    names = [result["label"] for result in results]
    x = np.arange(len(ALPHAS))
    width = 0.8 / len(results)
    for offset, result in enumerate(results):
        values = [result[f"coverage@{alpha:.2f}"] for alpha in ALPHAS]
        axes[1].bar(x + offset * width - 0.4 + width / 2, values, width=width, color=COLORS[result["model"]], label=result["label"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{alpha*100:.0f}%" for alpha in ALPHAS])
    axes[1].set_xlabel("accepted-error target")
    axes[1].set_ylabel("coverage")
    axes[1].set_title("Coverage at fixed error")
    axes[1].grid(True, axis="y", color="#E5E5E5", linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=7)
    for spine in ["top", "right"]:
        axes[0].spines[spine].set_visible(False)
        axes[1].spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="result/analysis")
    parser.add_argument("--figure-dir", default="result/report/figures")
    parser.add_argument("--report", default="result/report/model_comparison.md")
    args = parser.parse_args()

    runs = [
        ("resnet1d", "result/runs/seed_42/config.yaml"),
        ("resunet1d", "result/runs/model_compare/resunet1d/config.yaml"),
        ("resnet50_1d", "result/runs/model_compare/resnet50_1d/config.yaml"),
        ("transformer1d", "result/runs/model_compare/transformer1d/config.yaml"),
    ]
    results = [evaluate_run(name, path, args.device) for name, path in runs]

    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        plot_individual(result, figure_dir / f"model_{result['model']}_frontier.png")
    plot_combined(results, figure_dir / "model_comparison.png")

    rows = []
    for result in results:
        row = {key: value for key, value in result.items() if key not in {"coverages", "risks"}}
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "model_comparison.csv", index=False)
    (output_dir / "model_comparison.json").write_text(
        json.dumps(
            [
                {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in result.items()}
                for result in results
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    headers = ["model", "params", "accuracy_exact", "label_accuracy", "aurc", "coverage@0.10", "coverage@0.15", "coverage@0.20", "coverage@0.25"]
    lines = ["# 模型对比：ResNet1D / ResUNet / ResNet50 / Transformer", ""]
    lines.append("> seed 42；同一 fold 1–8/9/10、同一 20 epoch、同一全局温度校准；前沿分数为 max-prob。")
    lines.append("")
    lines.append("| 模型 | 参数量 | Accuracy (exact) | Label Acc | AURC | Cov@10% | Cov@15% | Cov@20% | Cov@25% |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        lines.append(
            f"| {result['label']} | {result['params']:,} | {result['accuracy_exact']*100:.1f}% | "
            f"{result['label_accuracy']*100:.1f}% | {result['aurc']:.4f} | "
            f"{result['coverage@0.10']*100:.1f}% | {result['coverage@0.15']*100:.1f}% | "
            f"{result['coverage@0.20']*100:.1f}% | {result['coverage@0.25']*100:.1f}% |"
        )
    lines.append("")
    lines.append("> Accuracy 为 coverage=100%、阈值 0.5 时的记录级 exact match；Label Acc 为逐标签准确率。Cov@α 为测试集经验前沿上 `accepted error <= α` 的最大 coverage。")
    lines.append("")
    lines.append("![model comparison](figures/model_comparison.png)")
    lines.append("")
    lines.append("## 分模型前沿")
    lines.append("")
    for result in results:
        lines.append(f"### {result['label']}")
        lines.append("")
        lines.append(f"![{result['model']}](figures/model_{result['model']}_frontier.png)")
        lines.append("")
    lines.append("## 读法")
    lines.append("")
    best_aurc = min(results, key=lambda item: item["aurc"])
    best_acc = max(results, key=lambda item: item["accuracy_exact"])
    lines.append(f"- 最低 AURC：**{best_aurc['label']}**（{best_aurc['aurc']:.4f}）。")
    lines.append(f"- 最高 Accuracy：**{best_acc['label']}**（{best_acc['accuracy_exact']*100:.1f}%）。")
    for alpha in ALPHAS:
        best = max(results, key=lambda item: item[f"coverage@{alpha:.2f}"])
        lines.append(f"- Cov@{alpha*100:.0f}% 最高：**{best['label']}**（{best[f'coverage@{alpha:.2f}']*100:.1f}%）。")
    lines.append("")
    lines.append("**限制**：单 seed、单 test；Cov@α 是测试集经验选点，不能部署；ResNet50/Transformer 未做逐模型调参，容量差异与训练预算差异会混淆结论。")
    lines.append("")
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"已写入 {output_dir / 'model_comparison.csv'}、{figure_dir / 'model_comparison.png'}、{report}")


if __name__ == "__main__":
    main()
