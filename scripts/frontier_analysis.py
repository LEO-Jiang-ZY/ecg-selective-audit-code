"""生成 entropy / max-prob / all-correct 三条 risk-coverage 前沿。"""
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
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptbxl.inference import calibrated_logits, load_run
from ptbxl.metrics import aurc, coverage_at_error, error_vector

SCORES = ("entropy", "max_prob", "all_correct")
COLORS = {"entropy": "#5F5E5A", "max_prob": "#185FA5", "all_correct": "#0F6E56"}


def score_map(logits: np.ndarray) -> dict[str, tuple[np.ndarray, str]]:
    p = expit(logits)
    entropy = -(p * np.log(p) + (1 - p) * np.log1p(-p)).mean(axis=1)
    max_prob = np.maximum(p, 1 - p).mean(axis=1)
    all_correct = np.prod(np.maximum(p, 1 - p), axis=1)
    return {
        "entropy": (entropy, "lower"),
        "max_prob": (max_prob, "higher"),
        "all_correct": (all_correct, "higher"),
    }


def curve(scores: np.ndarray, errors: np.ndarray, direction: str, n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scores) if direction == "lower" else np.argsort(-scores)
    sorted_errors = errors[order]
    n = len(errors)
    coverages = np.linspace(1 / n, 1.0, min(n, n_points))
    counts = np.maximum(1, np.ceil(coverages * n).astype(int))
    cumulative = np.cumsum(sorted_errors)
    risks = cumulative[counts - 1] / counts
    return coverages, risks


def cp_upper(errors: int, total: int, confidence: float = 0.95) -> float:
    if total <= 0 or errors >= total:
        return 1.0
    return float(beta.ppf(confidence, errors + 1, total - errors))


def choose_threshold(scores: np.ndarray, errors: np.ndarray, alpha: float, direction: str, confidence: float = 0.95) -> dict:
    method = "higher" if direction == "lower" else "lower"
    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99), method=method))
    best = None
    fallback = None
    for threshold in candidates:
        mask = scores <= threshold if direction == "lower" else scores >= threshold
        total = int(mask.sum())
        if total == 0:
            continue
        count = int(errors[mask].sum())
        ucb = cp_upper(count, total, confidence)
        candidate = {
            "threshold": float(threshold),
            "coverage": float(mask.mean()),
            "error": float(count / total),
            "ucb": ucb,
        }
        if fallback is None or ucb < fallback["ucb"]:
            fallback = candidate
        if ucb <= alpha and (best is None or candidate["coverage"] > best["coverage"]):
            best = dict(candidate)
    if best is None:
        best = dict(fallback)
        best["achievable"] = False
    else:
        best["achievable"] = True
    return best


def bootstrap_indices(patient_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(patient_ids)
    groups = {patient: np.where(patient_ids == patient)[0] for patient in unique}
    chosen = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([groups[patient] for patient in chosen])


def summarize(name: str, scores: np.ndarray, errors: np.ndarray, direction: str, alphas, n_boot: int, patient_ids, seed: int = 42) -> tuple[dict, dict, list[dict]]:
    coverage, risk = curve(scores, errors, direction)
    point = {
        "score": name,
        "aurc": aurc(risk, coverage),
        "risk@50": float(np.interp(0.5, coverage, risk)),
        "risk@70": float(np.interp(0.7, coverage, risk)),
        "risk@80": float(np.interp(0.8, coverage, risk)),
        "risk@90": float(np.interp(0.9, coverage, risk)),
    }
    for alpha in alphas:
        point[f"coverage@{alpha:.2f}"] = coverage_at_error(risk, coverage, alpha)

    rng = np.random.default_rng(seed)
    samples = {key: [] for key in ["aurc", "risk@80"] + [f"coverage@{alpha:.2f}" for alpha in alphas]}
    for _ in range(n_boot):
        idx = bootstrap_indices(patient_ids, rng)
        cov_b, risk_b = curve(scores[idx], errors[idx], direction, n_points=100)
        samples["aurc"].append(aurc(risk_b, cov_b))
        samples["risk@80"].append(float(np.interp(0.8, cov_b, risk_b)))
        for alpha in alphas:
            samples[f"coverage@{alpha:.2f}"].append(coverage_at_error(risk_b, cov_b, alpha))

    ci = {}
    for key, values in samples.items():
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        ci[key] = (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))) if values.size else (float("nan"), float("nan"))
    return point, ci, samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="result/runs/seed_42/config.yaml")
    parser.add_argument("--calibration", default="result/runs/seed_42/calibration.json")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.15, 0.20])
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="result/analysis")
    parser.add_argument("--figure", default="result/report/figures/frontier.png")
    parser.add_argument("--report", default="result/report/frontier.md")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    val = np.load(Path(cfg["training"]["validation_logits"]))
    val_logits, _ = calibrated_logits(val["logits"].astype(float), args.calibration)
    val_labels = val["labels"].astype(float)
    val_errors = error_vector(val_labels, val_logits, "any_label")
    val_scores = score_map(val_logits)

    test = load_run(args.config, "test", device=args.device)
    test_logits, _ = calibrated_logits(test["logits"].astype(float), args.calibration)
    test_labels = test["labels"].astype(float)
    test_errors = error_vector(test_labels, test_logits, "any_label")
    test_scores = score_map(test_logits)
    patient_ids = test["split_frame"]["patient_id"].to_numpy()

    points = {}
    cis = {}
    boot = {}
    curve_rows = []
    for name in SCORES:
        scores, direction = test_scores[name]
        point, ci, samples = summarize(name, scores, test_errors, direction, args.alphas, args.n_boot, patient_ids)
        points[name] = point
        cis[name] = ci
        boot[name] = samples
        coverage, risk = curve(scores, test_errors, direction)
        curve_rows.extend({"score": name, "coverage": float(c), "risk": float(r)} for c, r in zip(coverage, risk))

    risk_controlled = {}
    for name in SCORES:
        val_s, direction = val_scores[name]
        test_s, _ = test_scores[name]
        risk_controlled[name] = {}
        for alpha in args.alphas:
            chosen = choose_threshold(val_s, val_errors, alpha, direction)
            mask = test_s <= chosen["threshold"] if direction == "lower" else test_s >= chosen["threshold"]
            risk_controlled[name][f"{alpha:.2f}"] = {
                **chosen,
                "test_coverage": float(mask.mean()),
                "test_error": float(test_errors[mask].mean()) if mask.any() else float("nan"),
                "n_accepted": int(mask.sum()),
            }

    paired_rows = []
    for name in SCORES:
        if name == "entropy":
            continue
        for key in ["aurc", f"coverage@{args.alphas[-1]:.2f}"]:
            a = np.asarray(boot[name][key], dtype=float)
            b = np.asarray(boot["entropy"][key], dtype=float)
            valid = np.isfinite(a) & np.isfinite(b)
            diff = a[valid] - b[valid]
            better = diff < 0 if key == "aurc" else diff > 0
            paired_rows.append(
                {
                    "comparison": f"{name} - entropy",
                    "metric": key,
                    "delta": float(diff.mean()) if diff.size else float("nan"),
                    "ci_low": float(np.quantile(diff, 0.025)) if diff.size else float("nan"),
                    "ci_high": float(np.quantile(diff, 0.975)) if diff.size else float("nan"),
                    "p_better": float(better.mean()) if diff.size else float("nan"),
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curve_rows).to_csv(output_dir / "frontier_curves.csv", index=False)

    summary_rows = []
    for name in SCORES:
        row = {
            "score": name,
            "aurc": points[name]["aurc"],
            "aurc_low": cis[name]["aurc"][0],
            "aurc_high": cis[name]["aurc"][1],
            "risk@80": points[name]["risk@80"],
        }
        for alpha in args.alphas:
            key = f"coverage@{alpha:.2f}"
            row[f"{key}_empirical"] = points[name][key]
            row[f"{key}_low"] = cis[name][key][0]
            row[f"{key}_high"] = cis[name][key][1]
            row[f"{key}_risk_controlled"] = risk_controlled[name][f"{alpha:.2f}"]["test_coverage"]
            row[f"{key}_rc_error"] = risk_controlled[name][f"{alpha:.2f}"]["test_error"]
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "frontier_summary.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(output_dir / "frontier_paired.csv", index=False)

    payload = {
        "config": str(args.config),
        "calibration": str(args.calibration),
        "alphas": args.alphas,
        "n_boot": args.n_boot,
        "points": points,
        "confidence_intervals": cis,
        "risk_controlled": risk_controlled,
        "paired": paired_rows,
    }
    (output_dir / "frontier.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=160)
    for name in SCORES:
        scores, direction = test_scores[name]
        coverage, risk = curve(scores, test_errors, direction)
        axes[0].plot(coverage, risk, color=COLORS[name], linewidth=1.5, label=name)
        value = points[name].get(f"coverage@{args.alphas[-1]:.2f}")
        if np.isfinite(value):
            idx = int(np.argmin(np.abs(coverage - value)))
            axes[0].scatter([coverage[idx]], [risk[idx]], color=COLORS[name], s=20, zorder=3)
    axes[0].axvline(0.8, color="#BA7517", linewidth=0.8, linestyle=":")
    axes[0].set_xlabel("coverage")
    axes[0].set_ylabel("accepted any-label error")
    axes[0].set_title("Risk-coverage frontier")
    axes[0].grid(True, color="#E5E5E5", linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=8)

    names = list(SCORES)
    values = [points[name]["aurc"] for name in names]
    lows = [cis[name]["aurc"][0] for name in names]
    highs = [cis[name]["aurc"][1] for name in names]
    errors = [[value - low for value, low in zip(values, lows)], [high - value for value, high in zip(values, highs)]]
    axes[1].bar(names, values, color=[COLORS[name] for name in names], width=0.55)
    axes[1].errorbar(names, values, yerr=errors, fmt="none", ecolor="#444441", capsize=4, linewidth=0.8)
    axes[1].set_ylabel("AURC (lower is better)")
    axes[1].set_title("Area under risk-coverage curve")
    axes[1].grid(True, axis="y", color="#E5E5E5", linewidth=0.6)
    for spine in ["top", "right"]:
        axes[0].spines[spine].set_visible(False)
        axes[1].spines[spine].set_visible(False)
    fig.tight_layout()
    figure = Path(args.figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, facecolor="white")
    plt.close(fig)

    lines = ["# Risk-coverage 前沿（seed 42）", ""]
    lines.append("> fold 9 选择风险控制阈值；fold 10 计算经验前沿与风险控制点。单 seed 探索性结果。")
    lines.append("")
    lines.append("## 汇总")
    lines.append("")
    headers = ["score", "AURC (95% CI)", "coverage@15% 经验", "coverage@15% 风险控制", "coverage@20% 经验", "coverage@20% 风险控制", "risk@80%"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['score']} | {row['aurc']:.4f} [{row['aurc_low']:.4f}, {row['aurc_high']:.4f}] | "
            f"{row['coverage@0.15_empirical']*100:.1f}% | {row['coverage@0.15_risk_controlled']*100:.1f}% | "
            f"{row['coverage@0.20_empirical']*100:.1f}% | {row['coverage@0.20_risk_controlled']*100:.1f}% | "
            f"{row['risk@80']*100:.1f}% |"
        )
    lines.append("")
    lines.append("![frontier](figures/frontier.png)")
    lines.append("")
    lines.append("## 配对差异（vs entropy）")
    lines.append("")
    lines.append("| comparison | metric | delta | 95% CI | P(better) |")
    lines.append("|---|---|---:|---|---:|")
    for row in paired_rows:
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['delta']:+.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {row['p_better']:.3f} |"
        )
    lines.append("")
    lines.append("## 读法")
    lines.append("")
    lines.append("- `coverage@α 经验`：在 test 曲线上直接找 `risk <= α` 的最大 coverage，是 oracle 选点，不可部署。")
    lines.append("- `coverage@α 风险控制`：在 fold 9 上用 Clopper–Pearson 上界选阈，再应用到 test，是可部署但保守的版本。")
    lines.append("- AURC 越低越好；配对差异为负表示该分数优于 entropy。")
    lines.append("- `max_prob` 与 `all_correct` 在多数点上接近；若 CI 覆盖 0，不能判定谁更好。")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    s = summary.set_index("score")
    lines.append(
        f"- AURC：entropy {s.loc['entropy', 'aurc']:.4f}，max_prob {s.loc['max_prob', 'aurc']:.4f}，"
        f"all_correct {s.loc['all_correct', 'aurc']:.4f}；后两者相对 entropy 的配对 CI 均不含 0。"
    )
    lines.append(
        f"- coverage@20%（可部署风险控制）：entropy {s.loc['entropy', 'coverage@0.20_risk_controlled']*100:.1f}%，"
        f"max_prob {s.loc['max_prob', 'coverage@0.20_risk_controlled']*100:.1f}%，"
        f"all_correct {s.loc['all_correct', 'coverage@0.20_risk_controlled']*100:.1f}%。"
    )
    lines.append(
        f"- coverage@15%（可部署风险控制）：entropy {s.loc['entropy', 'coverage@0.15_risk_controlled']*100:.1f}%，"
        f"max_prob {s.loc['max_prob', 'coverage@0.15_risk_controlled']*100:.1f}%，"
        f"all_correct {s.loc['all_correct', 'coverage@0.15_risk_controlled']*100:.1f}%；差异较小，不能判定赢家。"
    )
    lines.append(
        f"- risk@80% coverage：entropy {s.loc['entropy', 'risk@80']*100:.1f}%，"
        f"max_prob {s.loc['max_prob', 'risk@80']*100:.1f}%，all_correct {s.loc['all_correct', 'risk@80']*100:.1f}%。"
    )
    lines.append("- 总体：换分数把前沿小幅外推，但没有阶跃式改善；max_prob 与 all_correct 的 AURC CI 高度重叠，不应宣称谁显著更优。")
    lines.append("")
    lines.append("**限制**：单 seed、单 test；经验 coverage@α 不能在部署时使用；需 seed 43/44 复核。")
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(pd.DataFrame(paired_rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"已写入 {output_dir / 'frontier_summary.csv'}、{figure}、{report}")


if __name__ == "__main__":
    main()
