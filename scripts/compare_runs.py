"""并排比较两个 run 的验证与测试结果。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def extract(run_dir: Path) -> dict:
    evaluation = load(run_dir / "evaluation.json")
    calibration = load(run_dir / "calibration.json")
    meta = load(run_dir / "run_meta.json")
    history = load(run_dir / "history.json")
    primary = evaluation["primary_error_unit"]
    baseline = evaluation["baseline"]["matched_coverage"]
    accepted = evaluation["accepted"]
    boot = evaluation["bootstrap"]["selective_risk"]
    return {
        "best_epoch": meta.get("best_epoch"),
        "best_val_loss": meta.get("best_validation_loss"),
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["validation_loss"],
        "temperature": calibration["temperature"],
        "entropy_threshold": calibration["entropy_threshold"],
        "val_coverage": calibration["val_coverage"],
        "test_coverage": accepted["coverage"],
        "baseline_error": baseline["accepted_error"][primary],
        "baseline_coverage": baseline["coverage"],
        "accepted_error": accepted[primary],
        "accepted_macro": accepted.get("macro_label"),
        "delta": accepted[primary] - baseline["accepted_error"][primary],
        "macro_auroc": evaluation["auroc"]["macro"],
        "macro_auprc": evaluation["auprc"]["macro"],
        "exact_match": evaluation["overall"]["exact_match"],
        "any_label_error": evaluation["overall"]["any_label_error"],
        "macro_label_error": evaluation["overall"]["macro_label_error"],
        "ece": evaluation["overall"]["ece"],
        "ci_low": boot.get("lower"),
        "ci_high": boot.get("upper"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", default="result/runs/seed_42")
    parser.add_argument("--run-b", default="result/runs/epochs_50")
    parser.add_argument("--label-a", default="20 epoch")
    parser.add_argument("--label-b", default="50 epoch")
    parser.add_argument("--output", default="result/report/epochs_comparison.md")
    args = parser.parse_args()

    a = extract(Path(args.run_a))
    b = extract(Path(args.run_b))

    rows = [
        ("best epoch", a["best_epoch"], b["best_epoch"], "—"),
        ("best val loss", a["best_val_loss"], b["best_val_loss"], b["best_val_loss"] - a["best_val_loss"]),
        ("final train loss", a["final_train_loss"], b["final_train_loss"], b["final_train_loss"] - a["final_train_loss"]),
        ("final val loss", a["final_val_loss"], b["final_val_loss"], b["final_val_loss"] - a["final_val_loss"]),
        ("temperature", a["temperature"], b["temperature"], b["temperature"] - a["temperature"]),
        ("entropy threshold", a["entropy_threshold"], b["entropy_threshold"], b["entropy_threshold"] - a["entropy_threshold"]),
        ("val coverage", a["val_coverage"], b["val_coverage"], b["val_coverage"] - a["val_coverage"]),
        ("test coverage", a["test_coverage"], b["test_coverage"], b["test_coverage"] - a["test_coverage"]),
        ("ERM baseline error", a["baseline_error"], b["baseline_error"], b["baseline_error"] - a["baseline_error"]),
        ("accepted any-label error", a["accepted_error"], b["accepted_error"], b["accepted_error"] - a["accepted_error"]),
        ("delta vs baseline", a["delta"], b["delta"], b["delta"] - a["delta"]),
        ("accepted macro-label error", a["accepted_macro"], b["accepted_macro"], b["accepted_macro"] - a["accepted_macro"]),
        ("macro AUROC", a["macro_auroc"], b["macro_auroc"], b["macro_auroc"] - a["macro_auroc"]),
        ("macro AUPRC", a["macro_auprc"], b["macro_auprc"], b["macro_auprc"] - a["macro_auprc"]),
        ("exact match", a["exact_match"], b["exact_match"], b["exact_match"] - a["exact_match"]),
        ("any-label error", a["any_label_error"], b["any_label_error"], b["any_label_error"] - a["any_label_error"]),
        ("macro-label error", a["macro_label_error"], b["macro_label_error"], b["macro_label_error"] - a["macro_label_error"]),
        ("ECE", a["ece"], b["ece"], b["ece"] - a["ece"]),
    ]

    lines = ["# 20 vs 50 epoch 对照", "", "> 单变量对照：同一 seed 42、同一数据与划分、同一超参数，只把 epochs 从 20 改为 50。"]
    lines.append("")
    lines.append(f"| 指标 | {args.label_a} | {args.label_b} | 差值（B - A） |")
    lines.append("|---|---:|---:|---:|")
    for name, value_a, value_b, diff in rows:
        if isinstance(value_a, int) and isinstance(value_b, int):
            lines.append(f"| {name} | {value_a} | {value_b} | — |")
        else:
            lines.append(f"| {name} | {value_a:.4f} | {value_b:.4f} | {diff:+.4f} |")
    lines.append("")
    lines.append(
        f"- {args.label_a} 的 accepted error 95% CI：[{a['ci_low']:.4f}, {a['ci_high']:.4f}]"
    )
    lines.append(
        f"- {args.label_b} 的 accepted error 95% CI：[{b['ci_low']:.4f}, {b['ci_high']:.4f}]"
    )
    lines.append("")

    val_gain = a["best_val_loss"] - b["best_val_loss"]
    test_gain = a["accepted_error"] - b["accepted_error"]
    if val_gain > 0.001 and test_gain > 0.002:
        conclusion = "50 epoch 在验证损失和测试主终点上都更好；值得用多 seed 复核，但不能仅凭单次 test 判定。"
    elif val_gain > 0.001:
        conclusion = "50 epoch 改善了验证损失，但测试主终点没有同步改善；提示验证集收益未迁移到测试集。"
    elif test_gain > 0.002:
        conclusion = "测试主终点略好，但验证损失没有改善；更可能是测试集波动，不能作为 50 epoch 更优的证据。"
    else:
        conclusion = "50 epoch 没有超过 20 epoch；当前 best-checkpoint 选择已经吸收了更长训练的可选收益。"
    lines.append(f"**判断**：{conclusion}")
    lines.append("")
    lines.append("**限制**：两个 run 使用同一个 fold 10，50 epoch 的 test 结果是探索性二次评估；不能据此在 test 上挑选 epoch。")
    lines.append("")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    frame = pd.DataFrame(
        [{"metric": name, args.label_a: value_a, args.label_b: value_b, "diff": diff} for name, value_a, value_b, diff in rows]
    )
    frame.to_csv(output.with_suffix(".csv"), index=False)
    print("\n".join(lines))
    print(f"已写入 {output} 与 {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
