"""把 evaluation.json / calibration.json 汇总为 Markdown 运行报告。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def pct(value, digits: int = 1) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    return f"{number * 100:.{digits}f}%"


def signed(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:+.{digits}f}"


def table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def risk_at(curve: dict, target: float) -> float:
    coverage = np.asarray(curve["coverage"], dtype=float)
    risk = np.asarray(curve["risk"], dtype=float)
    index = int(np.searchsorted(coverage, target))
    index = min(index, len(coverage) - 1)
    return float(risk[index])


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", default="result/results/evaluation.json")
    parser.add_argument("--calibration", default="result/results/calibration.json")
    parser.add_argument("--run-meta", default="result/results/run_meta.json")
    parser.add_argument("--history", default="result/results/history.json")
    parser.add_argument("--output", default="result/report/run_summary.md")
    args = parser.parse_args()

    evaluation = load(Path(args.evaluation))
    calibration = load(Path(args.calibration))
    run_meta = load(Path(args.run_meta)) if Path(args.run_meta).exists() else {}
    history = load(Path(args.history)) if Path(args.history).exists() else []

    primary = evaluation.get("primary_error_unit", "any_label")
    units = [primary] + [u for u in evaluation.get("secondary_error_units", []) if u != primary]
    baseline = evaluation.get("baseline", {})
    accepted = evaluation.get("accepted", {})

    lines = []
    lines.append("# PTB-XL ECG-iAIM 运行总结")
    lines.append("")
    lines.append("> 本报告由 `code/scripts/summarize.py` 从 `evaluation.json`、`calibration.json`、`run_meta.json` 自动生成；数字口径以主终点 any-label 为准。")
    lines.append("")

    lines.append("## 1. 运行配置")
    lines.append("")
    rows = [
        ["随机种子", run_meta.get("seed", "—")],
        ["训练 epoch", run_meta.get("epochs", "—")],
        ["训练 / 验证样本", f"{run_meta.get('n_train', '—')} / {run_meta.get('n_validation', '—')}"],
        ["测试样本", evaluation.get("n_test", "—")],
        ["测试患者数", evaluation.get("n_patients", "—")],
        ["设备", run_meta.get("device", "—")],
        ["最优 epoch / val loss", f"{run_meta.get('best_epoch', '—')} / {fmt(run_meta.get('best_validation_loss'))}"],
        ["波形缓存", run_meta.get("cache") or "未使用"],
    ]
    lines.append(table(["项", "值"], rows))
    lines.append("")
    if history:
        lines.append("### 训练曲线")
        lines.append("")
        rows = [[item.get("epoch"), fmt(item.get("train_loss")), fmt(item.get("validation_loss"))] for item in history]
        lines.append(table(["epoch", "train loss", "val loss"], rows))
        lines.append("")

    lines.append("## 2. 校准结果")
    lines.append("")
    before = calibration.get("metrics_uncalibrated", {})
    after = calibration.get("metrics_calibrated", {})
    rows = []
    for key, label in [("nll", "NLL"), ("brier", "Brier"), ("ece", "ECE")]:
        rows.append([label, fmt(before.get(key)), fmt(after.get(key)), signed(after.get(key, 0) - before.get(key, 0))])
    lines.append(table(["指标（fold 9）", "未校准", "校准后", "变化"], rows))
    lines.append("")
    lines.append(
        f"- 温度：**{fmt(calibration.get('temperature'), 4)}**；熵阈值：**{fmt(calibration.get('entropy_threshold'), 4)}**"
    )
    lines.append(
        f"- 目标 coverage：{pct(calibration.get('target_coverage'))}；验证集实际 coverage：{pct(calibration.get('val_coverage'))}"
    )
    lines.append(f"- 校准来源：{calibration.get('fit_split', '—')} / {calibration.get('n', '—')} 条")
    lines.append("")

    lines.append("## 3. 主终点：80% coverage 下的 accepted-set error")
    lines.append("")
    matched = baseline.get("matched_coverage", {})
    baseline_coverage = matched.get("coverage", baseline.get("coverage"))
    rows = []
    for unit in units:
        base_error = matched.get("accepted_error", {}).get(unit, baseline.get("accepted_error", {}).get(unit))
        cal_error = accepted.get(unit)
        delta = None if base_error is None or cal_error is None else cal_error - base_error
        rows.append(
            [
                f"ERM 基线（{unit}）",
                pct(baseline_coverage),
                fmt(base_error),
                "—",
            ]
        )
        rows.append(
            [
                f"校准 + 拒答（{unit}）",
                pct(accepted.get("coverage")),
                fmt(cal_error),
                signed(delta),
            ]
        )
    lines.append(table(["方案", "coverage", "accepted error", "相对基线"], rows))
    lines.append("")
    lines.append("> 负的「相对基线」表示校准 + 拒答的 accepted error 更低。基线 coverage 与校准方案 coverage 可能因阈值在 fold 9 上分别选择而略有差异。")
    lines.append("")
    curve_base = baseline.get("risk_coverage", {}).get(primary)
    curve_cal = evaluation.get("risk_coverage", {}).get(primary)
    if curve_base and curve_cal:
        lines.append(
            f"- 固定 80% coverage 的曲线读数：ERM 基线 {fmt(risk_at(curve_base, 0.8))}，"
            f"校准 + 拒答 {fmt(risk_at(curve_cal, 0.8))}。"
        )
        lines.append("")
    boot = evaluation.get("bootstrap", {}).get("selective_risk", {})
    lines.append(
        f"- 校准 + 拒答 accepted error 的 patient-level bootstrap 95% CI："
        f"[{fmt(boot.get('lower'))}, {fmt(boot.get('upper'))}]，点估计 {fmt(boot.get('estimate'))}，"
        f"重采样 {boot.get('n_resamples', '—')} 次 / {boot.get('n_patients', '—')} 名患者"
    )
    lines.append("")

    lines.append("## 4. 整体与分类指标（fold 10）")
    lines.append("")
    overall = evaluation.get("overall", {})
    rows = [
        ["NLL", fmt(overall.get("nll"))],
        ["Brier", fmt(overall.get("brier"))],
        ["ECE", fmt(overall.get("ece"))],
        ["label accuracy", pct(overall.get("label_accuracy"))],
        ["exact match", pct(overall.get("exact_match"))],
        ["any-label error", pct(overall.get("any_label_error"))],
        ["macro-label error", pct(overall.get("macro_label_error"))],
        ["macro AUROC", fmt(evaluation.get("auroc", {}).get("macro"))],
        ["macro AUPRC", fmt(evaluation.get("auprc", {}).get("macro"))],
    ]
    lines.append(table(["指标", "值"], rows))
    lines.append("")
    per_class = evaluation.get("per_class", [])
    if per_class:
        rows = [
            [
                row.get("class"),
                row.get("n_positive"),
                pct(row.get("prevalence")),
                fmt(row.get("auroc"), 3),
                fmt(row.get("auprc"), 3),
                fmt(row.get("nll")),
                fmt(row.get("brier")),
                pct(row.get("accuracy")),
            ]
            for row in per_class
        ]
        lines.append(table(["类别", "阳性数", "患病率", "AUROC", "AUPRC", "NLL", "Brier", "accuracy"], rows))
    lines.append("")

    lines.append("## 5. 亚组")
    lines.append("")
    subgroups = evaluation.get("subgroups", {})
    for column, rows_data in subgroups.items():
        lines.append(f"### {column}")
        lines.append("")
        if not isinstance(rows_data, list):
            lines.append(f"未计算：{rows_data}")
            lines.append("")
            continue
        rows = []
        for row in rows_data:
            rows.append(
                [
                    row.get("group"),
                    row.get("n"),
                    row.get("n_accepted", "—"),
                    pct(row.get("coverage")),
                    fmt(row.get("accepted_error")),
                    pct(row.get("exact_match")),
                    pct(row.get("macro_label_error")),
                    fmt(row.get("ece")),
                ]
            )
        lines.append(table(["组", "n", "n accepted", "coverage", "accepted error", "exact match", "macro error", "ECE"], rows))
        lines.append("")

    lines.append("## 6. 设备 / 站点构成偏移（train vs test）")
    lines.append("")
    shift = evaluation.get("shift", {})
    for column, payload in shift.items():
        lines.append(f"### {column}")
        lines.append("")
        rows_data = sorted(payload.get("rows", []), key=lambda row: abs(row.get("diff_pp", 0)), reverse=True)[:6]
        rows = [[row.get("value"), pct(row.get("reference_share")), pct(row.get("target_share")), signed(row.get("diff_pp"), 2)] for row in rows_data]
        lines.append(f"总变差距离 TVD = **{fmt(payload.get('tvd'))}**")
        lines.append("")
        lines.append(table(["取值", "train 占比", "test 占比", "差值（百分点）"], rows))
        lines.append("")

    lines.append("## 7. risk-coverage 摘要")
    lines.append("")
    rows = []
    for target in [0.5, 0.7, 0.8, 0.9, 1.0]:
        row = [pct(target)]
        for unit in units:
            base_curve = baseline.get("risk_coverage", {}).get(unit)
            cal_curve = evaluation.get("risk_coverage", {}).get(unit)
            row.append(fmt(risk_at(base_curve, target)) if base_curve else "—")
            row.append(fmt(risk_at(cal_curve, target)) if cal_curve else "—")
        rows.append(row)
    headers = ["coverage"]
    for unit in units:
        headers.extend([f"基线 {unit}", f"校准+拒答 {unit}"])
    lines.append(table(headers, rows))
    lines.append("")

    lines.append("## 8. 图")
    lines.append("")
    lines.append("![Training curve](figures/training_curve.png)")
    lines.append("")
    lines.append("![Risk-coverage](figures/risk_coverage.png)")
    lines.append("")

    lines.append("## 9. 自动摘要与限制")
    lines.append("")
    temp = calibration.get("temperature")
    ece_before = before.get("ece")
    ece_after = after.get("ece")
    lines.append(f"- 温度缩放：T = {fmt(temp)}；ECE {fmt(ece_before)} → {fmt(ece_after)}（{signed((ece_after or 0) - (ece_before or 0))}）。")
    base_error = matched.get("accepted_error", {}).get(primary, baseline.get("accepted_error", {}).get(primary))
    cal_error = accepted.get(primary)
    if base_error is not None and cal_error is not None:
        lines.append(
            f"- 主终点：ERM 基线 {pct(base_error)} → 校准 + 拒答 {pct(cal_error)}，"
            f"绝对变化 {signed((cal_error - base_error) * 100, 2)} 个百分点（coverage {pct(baseline_coverage)} vs {pct(accepted.get('coverage'))}）。"
        )
        if cal_error < base_error:
            lines.append("- 方向：校准 + 拒答低于基线，方向支持 H3；但仍需结合区间估计判断显著性。")
        elif cal_error > base_error:
            lines.append("- 方向：校准 + 拒答不低于基线，本 run 未复现 H3（H3：80% coverage 下校准 + 拒答的 accepted error 低于 ERM 基线）。")
        if boot and boot.get("lower") is not None and boot.get("lower") <= base_error <= boot.get("upper"):
            lines.append("- 基线的点估计落在校准方案 95% CI 内，当前 run 不能判定两者存在显著差异。")
    lines.append("- 注意：ECE/NLL/Brier 的改善不等于 selective risk 改善；本 run 中两者方向可以不一致。")
    lines.append(f"- macro AUROC = {fmt(evaluation.get('auroc', {}).get('macro'), 3)}；macro AUPRC = {fmt(evaluation.get('auprc', {}).get('macro'), 3)}。")
    if shift:
        worst = max(shift.items(), key=lambda item: item[1].get("tvd", 0))
        lines.append(f"- 最大构成偏移：{worst[0]}，TVD = {fmt(worst[1].get('tvd'))}。")
    lines.append("- 限制：单 seed、固定 20 epoch、未做外部数据集验证；device/site 偏移是观察性构成差异，不是受控干预。")
    lines.append("- 限制：全零标签记录保留，all-zero 仅表示五个 superclass 均未命中，不等于 NORM。")
    lines.append("- 限制：本报告不含 SHAP/LIME；主终点与亚组结论不构成临床有效性证据。")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**边界声明**：所有数字来自当前 checkpoint 与 fold 10 的单次评估；不得用于临床诊断或治疗建议。")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入 {output}（{len(lines)} 行）")


if __name__ == "__main__":
    main()
