"""多标签 selective-classification gap 的误差预算分解。

在固定 coverage 下把 accepted any-label error 拆成两部分：

    R_current(c) = R_oracle(c) + ε_bayes(c) + ε_rank(c)

其中
- R_oracle(c)：完美 correctness 排序下的风险（不可实现的下界，仅作标尺）
- R_bayes(c)：用多模型 ensemble 后验近似的最优风险，即不可约噪声
- ε_bayes = R_bayes - R_oracle，ε_rank = R_current - R_bayes

另报告两项敏感性：ε_stat（患者级 bootstrap 标准差）与 ε_shift（按训练设备/站点
分布重加权后的风险变化）。这是近似分解，不是严格可加的上界；论文中需如此表述。
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
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptbxl.inference import load_run
from ptbxl.utils import jsonable

from ptbxl.palette import (  # noqa: E402
    EDGE_FLOOR as EDGE_BAYES,
    EDGE_ORACLE,
    EDGE_RANK,
    FILL_FLOOR as FILL_BAYES,
    FILL_ORACLE,
    FILL_RANK,
    GRID,
    INK,
    MUTED,
)

# 参考排序 = 同架构三 seed ensemble。用四个不同架构的 ensemble 会低于最佳单模型，
# 那样 reference 就不再是「比 current 更好」的标尺（实测 80% coverage 下 30.60 vs 30.15）。
RUNS = {
    "resnet1d": "result/runs/seed_42/config.yaml",
    "seed_43": "result/runs/seed_43/config.yaml",
    "seed_44": "result/runs/seed_44/config.yaml",
}


def correct_indicator(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return np.all((p > threshold) == (y > threshold), axis=1).astype(float)


def all_correct_probability(p: np.ndarray) -> np.ndarray:
    return np.prod(np.where(p > 0.5, p, 1 - p), axis=1)


def curve_by_order(values: np.ndarray, coverages: np.ndarray) -> np.ndarray:
    """values 为已按可信度降序排列的逐记录“正确性”；返回 1 - 前缀均值。

    取整规则与 ptbxl.budget 保持一致（round），否则 oracle 会与
    一致性下界 (E − (n−k)) / k 对不上。
    """
    cumulative = np.cumsum(values)
    k_n = len(values)
    risks = []
    for c in coverages:
        k = int(min(k_n, max(1, round(c * k_n))))
        risks.append(1.0 - cumulative[k - 1] / k)
    return np.array(risks)


def decompose(y: np.ndarray, p_single: np.ndarray, p_ensemble: np.ndarray, coverages: np.ndarray) -> pd.DataFrame:
    """风险标尺：oracle（真实标签排序）≤ reference（ensemble 排序）≤ current（单模型排序）。

    reference 用 ensemble 分数排序、再用真实 0/1 计算风险，因此它一定不低于 oracle，
    也不依赖概率乘积假设。它是 Bayes 后验的代理，故 epsilon_reference 是 Bayes 噪声的
    上界估计，而非精确值。
    """
    correct = correct_indicator(y, p_single)

    oracle_order = np.argsort(-correct)
    risk_oracle = curve_by_order(correct[oracle_order], coverages)

    # reference 与 current 同族（max-prob），差异只来自 ensemble 与单模型
    reference_score = np.maximum(p_ensemble, 1 - p_ensemble).mean(axis=1)
    reference_order = np.argsort(-reference_score)
    risk_reference = curve_by_order(correct[reference_order], coverages)

    score = np.maximum(p_single, 1 - p_single).mean(axis=1)
    current_order = np.argsort(-score)
    risk_current = curve_by_order(correct[current_order], coverages)

    q = np.clip(p_single, 1e-7, 1 - 1e-7)
    entropy = -(q * np.log(q) + (1 - q) * np.log1p(-q)).mean(axis=1)
    entropy_order = np.argsort(entropy)
    risk_entropy = curve_by_order(correct[entropy_order], coverages)

    table = pd.DataFrame(
        {
            "coverage": coverages,
            "risk_oracle": risk_oracle,
            "risk_reference": risk_reference,
            "risk_current_maxprob": risk_current,
            "risk_current_entropy": risk_entropy,
            "epsilon_reference": risk_reference - risk_oracle,
            "epsilon_rank": risk_current - risk_reference,
            "epsilon_rank_entropy": risk_entropy - risk_reference,
            "gap_total": risk_current - risk_oracle,
        }
    )
    return table


def bootstrap_stat(y, p_single, patient_ids, coverages, n_resamples: int = 200, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique = np.unique(patient_ids)
    groups = {pid: np.where(patient_ids == pid)[0] for pid in unique}
    draws = []
    for _ in range(n_resamples):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([groups[pid] for pid in chosen])
        score = np.maximum(p_single[index], 1 - p_single[index]).mean(axis=1)
        order = np.argsort(-score)
        correct = correct_indicator(y[index], p_single[index])
        draws.append(curve_by_order(correct[order], coverages))
    return np.std(np.array(draws), axis=0)


def shift_risk(y, p_single, frame, coverages) -> tuple[np.ndarray, dict]:
    """按训练集设备分布对测试集做重要性重加权后的风险。"""
    device = frame["device_group"].astype(str).to_numpy()
    train_share = frame.attrs.get("train_device_share", {})
    if not train_share:
        return np.full(len(coverages), np.nan), {"note": "缺少训练设备分布"}
    test_share = pd.Series(device).value_counts(normalize=True).to_dict()
    weights = np.array([train_share.get(d, 0.0) / max(test_share.get(d, 1e-9), 1e-9) for d in device])
    weights = np.clip(weights, 0.0, np.quantile(weights, 0.95))
    weights = weights / weights.mean()

    score = np.maximum(p_single, 1 - p_single).mean(axis=1)
    order = np.argsort(-score)
    correct = correct_indicator(y, p_single)[order]
    w = weights[order]
    cumulative_w = np.cumsum(w)
    cumulative_wc = np.cumsum(w * correct)
    risks = []
    for c in coverages:
        target = c * cumulative_w[-1]
        k = int(np.searchsorted(cumulative_w, target))
        k = min(max(k, 1), len(w))
        risks.append(1.0 - cumulative_wc[k - 1] / cumulative_w[k - 1])
    return np.array(risks), {"max_weight": float(weights.max()), "ess": float(weights.sum() ** 2 / (weights ** 2).sum())}


def plot(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.3), dpi=300)
    c = table["coverage"]
    ax.stackplot(
        c,
        100 * table["risk_oracle"],
        100 * table["epsilon_reference"],
        100 * table["epsilon_rank"].clip(lower=0),
        colors=[FILL_ORACLE, FILL_BAYES, FILL_RANK],
        edgecolor=[EDGE_ORACLE, EDGE_BAYES, EDGE_RANK],
        linewidth=0.8,
        labels=[
            "oracle risk (perfect ordering)",
            "floor above oracle (irreducible + reference)",
            "ranking error (single model)",
        ],
    )
    ax.plot(c, 100 * table["risk_current_maxprob"], color=INK, linewidth=1.2, label="observed risk (max-prob)")
    ax.axvline(0.80, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
    ax.text(0.79, 38, "80%", fontsize=6.5, color=MUTED, ha="right", va="top")
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0, 45)
    ax.set_xlabel("coverage", fontsize=8, color=INK)
    ax.set_ylabel("accepted any-label error (%)", fontsize=8, color=INK)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(MUTED)
        ax.spines[spine].set_linewidth(0.6)
    ax.tick_params(colors=INK, labelsize=7.5, width=0.6, length=3)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.0, frameon=False, loc="upper left")
    fig.tight_layout(pad=0.5)
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--output-dir", default="result/analysis")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    predictions, labels, frame = {}, None, None
    for name, config in RUNS.items():
        run = load_run(config, "test", device=args.device)
        cal_path = Path(config).parent / "calibration.json"
        cal = json.loads(cal_path.read_text(encoding="utf-8")) if cal_path.exists() else None
        temperature = float(cal["temperature"]) if cal else 1.0
        predictions[name] = expit(run["logits"].astype(float) / temperature)
        if labels is None:
            labels = run["labels"].astype(float)
            frame = run["split_frame"].copy()
        print(f"loaded {name} n={len(run['labels'])}", flush=True)

    p_ensemble = np.mean([predictions[k] for k in RUNS], axis=0)
    p_single = predictions["resnet1d"]
    patient_ids = frame["patient_id"].to_numpy()

    coverages = np.round(np.arange(0.20, 1.0001, 0.02), 2)
    table = decompose(labels, p_single, p_ensemble, coverages)
    table["epsilon_stat"] = bootstrap_stat(labels, p_single, patient_ids, coverages, args.bootstrap)
    table["risk_shift"] = np.nan

    # 用训练集设备分布做重加权
    import yaml

    from ptbxl.cohort import derive_cohort_columns
    from ptbxl.labels import load_and_label
    from ptbxl.split import patient_fold_split

    cfg = yaml.safe_load(Path(RUNS["resnet1d"]).read_text(encoding="utf-8"))
    root = Path(cfg["data"]["root"])
    full = load_and_label(
        root / cfg["data"]["metadata"], root / cfg["data"]["scp_statements"],
        cfg["data"]["classes"],
        min_scp_probability=cfg.get("labeling", {}).get("min_scp_probability"),
        drop_all_zero=bool(cfg.get("labeling", {}).get("drop_all_zero", False)),
    )
    full = derive_cohort_columns(full)
    split = patient_fold_split(full, cfg["data"]["folds"])
    train_share = split["train"]["device_group"].astype(str).value_counts(normalize=True).to_dict()
    frame.attrs["train_device_share"] = train_share
    shift_values, shift_meta = shift_risk(labels, p_single, frame, coverages)
    table["risk_shift"] = shift_values
    table["epsilon_shift"] = shift_values - table["risk_current_maxprob"]

    table.to_csv(out / "gap_decomposition.csv", index=False)

    summary_rows = []
    for c in [0.50, 0.70, 0.80, 0.90]:
        row = table.iloc[int(np.argmin(np.abs(coverages - c)))]
        summary_rows.append(
            {
                "coverage": float(row["coverage"]),
                "observed_risk": float(row["risk_current_maxprob"]),
                "reference_floor": float(row["risk_reference"]),
                "epsilon_reference": float(row["epsilon_reference"]),
                "epsilon_rank": float(row["epsilon_rank"]),
                "rank_identifiable": bool(abs(row["epsilon_rank"]) > row["epsilon_stat"]),
                "epsilon_reference_share": float(row["epsilon_reference"] / max(row["gap_total"], 1e-9)),
                "epsilon_stat": float(row["epsilon_stat"]),
                "epsilon_shift": float(row["epsilon_shift"]),
                "risk_entropy": float(row["risk_current_entropy"]),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "gap_summary.csv", index=False)

    (out / "gap_meta.json").write_text(
        json.dumps(jsonable({"ensemble_members": list(RUNS), "shift": shift_meta,
                             "n_test": int(len(labels))}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot(table, out / "gap_decomposition.png")

    print(summary.round(4).to_string(index=False))
    print("shift meta", shift_meta)


if __name__ == "__main__":
    main()
