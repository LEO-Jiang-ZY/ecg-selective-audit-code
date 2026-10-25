"""汇总风险目标阈值与逐标签拒答结果。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def fmt(value, digits=4):
    if pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def pct(value, digits=1):
    if pd.isna(value):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def main():
    risk = pd.read_csv("result/analysis/risk_target.csv")
    per_summary = pd.read_csv("result/analysis/per_label_abstention.summary.csv")
    per_detail = pd.read_csv("result/analysis/per_label_abstention.csv")
    reference = json.loads(Path("result/results/evaluation.json").read_text(encoding="utf-8"))

    figures = Path("result/report/figures")
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=160)
    ax.plot(risk["test_coverage"], risk["test_error"], color="#185FA5", marker="o", linewidth=1.4, label="max-prob + risk target")
    for _, row in risk.iterrows():
        ax.annotate(f"α={row['target_risk']:.2f}", (row["test_coverage"], row["test_error"]), textcoords="offset points", xytext=(5, 5), fontsize=7, color="#185FA5")
    ax.scatter(
        [reference["accepted"]["coverage"]],
        [reference["accepted"]["any_label"]],
        color="#BA7517",
        marker="s",
        s=28,
        label="current entropy @80%",
        zorder=3,
    )
    ax.set_xlabel("test coverage")
    ax.set_ylabel("accepted any-label error")
    ax.set_title("Risk-target operating points vs current 80% coverage")
    ax.grid(True, color="#E5E5E5", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures / "risk_target.png", facecolor="white")
    plt.close(fig)

    lines = ["# 决策层优化：风险目标阈值与逐标签拒答", ""]
    lines.append("> seed 42；max-prob 与逐标签阈值均在 fold 9 上选择，fold 10 只评估一次。单 seed 探索性结果，不是最终结论。")
    lines.append("")

    lines.append("## 1. max-prob + 风险目标阈值")
    lines.append("")
    rows = []
    for _, row in risk.iterrows():
        rows.append(
            [
                f"α={row['target_risk']:.2f}",
                "是" if bool(row["achievable"]) else "否",
                pct(row["val_coverage"]),
                f"{fmt(row['val_error'])} / UCB {fmt(row['val_ucb'])}",
                pct(row["test_coverage"]),
                fmt(row["test_error"]),
                f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]",
            ]
        )
    lines.append(table(["目标风险", "可达", "val coverage", "val error / 95% UCB", "test coverage", "test accepted error", "95% CI"], rows))
    lines.append("")
    lines.append(
        f"当前主流程（熵 + 80% coverage）：coverage {pct(reference['accepted']['coverage'])}，"
        f"accepted any-label error {pct(reference['accepted']['any_label'])}。"
    )
    lines.append("")
    lines.append("![risk-target](figures/risk_target.png)")
    lines.append("")
    lines.append("**读法**：目标风险越低，能自动签发的比例越低。α=0.20 时 test coverage 约 45.5%，accepted error 16.5%；α=0.30 时 coverage 约 70.0%，error 26.6%。这不是模型变差，而是把 operating point 从“固定 80% 覆盖率”换成了“固定风险”。")
    lines.append("")

    lines.append("## 2. 逐标签拒答")
    lines.append("")
    rows = []
    for _, row in per_summary.iterrows():
        rows.append(
            [
                row["scheme"],
                pct(row["label_coverage"]),
                pct(row["label_error"]),
                pct(row["full_accept_rate"]),
                pct(row["any_error_full_accept"]),
                fmt(row["mean_deferred_labels"], 2),
            ]
        )
    lines.append(table(["方案", "标签覆盖率", "标签错误率", "整条记录全接受率", "全接受记录的 any-label error", "平均转人工标签数"], rows))
    lines.append("")
    lines.append("**读法**：`label_coverage` 是 5×N 个标签中被自动接受的比例；`label_error` 是这些被接受标签的错误率；`full_accept_rate` 是五个标签全部被接受的记录比例。逐标签拒答的代价是记录很少能“整条全自动”，但人工只需复核被 defer 的少数标签。")
    lines.append("")

    lines.append("### 逐标签阈值（tuned_10pct）")
    lines.append("")
    subset = per_detail[per_detail["scheme"] == "tuned_10pct"]
    rows = [
        [
            row["label"],
            fmt(row["low"], 2),
            fmt(row["high"], 2),
            pct(row["val_coverage"]),
            pct(row["val_error"]),
            pct(row["test_coverage"]),
            pct(row["test_error"]),
        ]
        for _, row in subset.iterrows()
    ]
    lines.append(table(["标签", "低阈值", "高阈值", "val 覆盖率", "val 错误率", "test 覆盖率", "test 错误率"], rows))
    lines.append("")

    lines.append("## 3. 与当前记录级 80% 拒答的对比")
    lines.append("")
    lines.append(
        f"- 当前记录级：accepted coverage {pct(reference['accepted']['coverage'])}，"
        f"any-label error {pct(reference['accepted']['any_label'])}，macro-label error {pct(reference['accepted']['macro_label'])}。"
    )
    lines.append("- 风险目标 α=0.20：coverage 45.5%，any-label error 16.5%，适合“宁可少签发、也要控制错误”的场景。")
    lines.append("- 逐标签 tuned_10pct：标签覆盖率 76.4%，标签错误率 5.6%；但整条记录全接受率只有 39.8%，平均每条记录仍有 1.18 个标签需要人工复核。")
    lines.append("- 两者不是同一单位：前者是记录级，后者是标签级；不能直接说“5.6% 比 9.9% 更好”，应结合人工复核负担一起看。")
    lines.append("")

    lines.append("## 4. 结论与限制")
    lines.append("")
    lines.append("- 固定 80% coverage 不是安全目标；它对应约 31% 的记录级错误。若临床目标是 20% 错误，只能自动签发约 45% 的记录。")
    lines.append("- 逐标签拒答把“整条记录退回”变成“只退回不确定标签”，在标签覆盖率约 76% 时把标签错误压到 5.6%；代价是整条全自动率降到 39.8%。")
    lines.append("- 两项都依赖 fold 9 选阈值；逐标签阈值有 5×2 个参数，过拟合风险高于单阈值，需多 seed/交叉拟合复核。")
    lines.append("- 当前仍是单 seed、单 test 的探索性分析；不能据此在 test 上挑选 operating point。")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**边界声明**：数字来自 seed 42 的 fold 10 单次评估，不构成临床有效性证据。")

    output = Path("result/report/decision_layer.md")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入 {output} 与 {figures / 'risk_target.png'}")


if __name__ == "__main__":
    main()
