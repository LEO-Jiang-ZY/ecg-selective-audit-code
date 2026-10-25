"""亚组分析、bootstrap 置信区间与分布偏移表。"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .metrics import accepted_set_error, predictive_entropy, summarize_metrics


def subgroup_metrics(
    metadata: pd.DataFrame,
    y_true,
    logits,
    group_col: str,
    probability_threshold: float = 0.5,
    error_unit: str = "any_label",
    accepted_mask=None,
) -> pd.DataFrame:
    """按 group_col 分组计算指标；accepted_mask 为全局保留掩码。"""
    if group_col not in metadata:
        raise KeyError(f"metadata 缺少亚组列 {group_col!r}")
    metadata = metadata.reset_index(drop=True)
    y_true = np.asarray(y_true)
    logits = np.asarray(logits)
    mask = None if accepted_mask is None else np.asarray(accepted_mask, dtype=bool)
    if mask is not None and len(mask) != len(metadata):
        raise ValueError(f"accepted_mask 长度 {len(mask)} 与 metadata {len(metadata)} 不一致")

    rows = []
    for value, index in metadata.groupby(group_col, dropna=False, observed=True).groups.items():
        idx = np.asarray(list(index), dtype=int)
        metrics = summarize_metrics(y_true[idx], logits[idx], threshold=probability_threshold)
        row = {"group": "unknown" if pd.isna(value) else str(value), "n": int(len(idx)), **metrics}
        if mask is not None:
            group_mask = mask[idx]
            row["n_accepted"] = int(group_mask.sum())
            row["coverage"] = float(group_mask.mean())
            row["accepted_error"] = accepted_set_error(
                y_true[idx],
                logits[idx],
                group_mask,
                error_unit=error_unit,
                threshold=probability_threshold,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _draw_indices(rng: np.random.Generator, n: int, patient_ids=None) -> np.ndarray:
    if patient_ids is None:
        return rng.integers(0, n, size=n)
    patients = np.asarray(patient_ids)
    unique = np.unique(patients)
    groups = {patient: np.where(patients == patient)[0] for patient in unique}
    chosen = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([groups[patient] for patient in chosen])


def bootstrap_metric(
    y_true,
    logits,
    metric: str | Callable = "exact_match",
    n_resamples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
    patient_ids=None,
) -> Mapping[str, float]:
    """bootstrap 置信区间；提供 patient_ids 时按患者整群重采样。"""
    y_true = np.asarray(y_true)
    logits = np.asarray(logits)
    rng = np.random.default_rng(seed)

    def evaluate(indices: np.ndarray) -> float:
        if callable(metric):
            return float(metric(y_true[indices], logits[indices]))
        return float(summarize_metrics(y_true[indices], logits[indices])[metric])

    point = evaluate(np.arange(len(y_true)))
    sampled = np.array([evaluate(_draw_indices(rng, len(y_true), patient_ids)) for _ in range(n_resamples)], dtype=float)
    sampled = sampled[np.isfinite(sampled)]
    alpha = (1 - confidence) / 2
    return {
        "estimate": point,
        "lower": float(np.quantile(sampled, alpha)) if sampled.size else float("nan"),
        "upper": float(np.quantile(sampled, 1 - alpha)) if sampled.size else float("nan"),
        "n_resamples": int(n_resamples),
        "n_patients": int(len(np.unique(patient_ids))) if patient_ids is not None else None,
        "unit": "patient" if patient_ids is not None else "record",
    }


def bootstrap_selective_risk(
    y_true,
    logits,
    entropy_threshold: float,
    error_unit: str = "any_label",
    patient_ids=None,
    n_resamples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> Mapping[str, float]:
    """固定熵阈值下，对 accepted-set error 做 bootstrap。"""
    y_true = np.asarray(y_true)
    logits = np.asarray(logits)
    entropy = predictive_entropy(logits, from_logits=True)
    rng = np.random.default_rng(seed)

    def risk(indices: np.ndarray) -> float:
        mask = entropy[indices] <= float(entropy_threshold)
        return accepted_set_error(y_true[indices], logits[indices], mask, error_unit=error_unit)

    point = risk(np.arange(len(y_true)))
    sampled = np.array([risk(_draw_indices(rng, len(y_true), patient_ids)) for _ in range(n_resamples)], dtype=float)
    sampled = sampled[np.isfinite(sampled)]
    alpha = (1 - confidence) / 2
    return {
        "estimate": point,
        "lower": float(np.quantile(sampled, alpha)) if sampled.size else float("nan"),
        "upper": float(np.quantile(sampled, 1 - alpha)) if sampled.size else float("nan"),
        "n_resamples": int(n_resamples),
        "n_patients": int(len(np.unique(patient_ids))) if patient_ids is not None else None,
        "unit": "patient" if patient_ids is not None else "record",
    }


def distribution_shift_table(reference: pd.DataFrame, target: pd.DataFrame, columns: Sequence[str]) -> dict:
    """比较 reference 与 target 在指定列上的构成差异，返回各列 TVD 与逐类别占比。"""
    output = {}
    for column in columns:
        if column not in reference or column not in target:
            output[column] = {"tvd": float("nan"), "rows": [], "error": "column missing"}
            continue
        ref = reference[column].map(lambda value: "unknown" if pd.isna(value) else str(value)).value_counts(normalize=True)
        tgt = target[column].map(lambda value: "unknown" if pd.isna(value) else str(value)).value_counts(normalize=True)
        values = sorted(set(ref.index) | set(tgt.index), key=lambda value: tgt.get(value, 0.0), reverse=True)
        rows = []
        for value in values:
            ref_share = float(ref.get(value, 0.0))
            tgt_share = float(tgt.get(value, 0.0))
            rows.append(
                {
                    "value": value,
                    "reference_share": ref_share,
                    "target_share": tgt_share,
                    "diff_pp": round((tgt_share - ref_share) * 100, 2),
                }
            )
        tvd = 0.5 * sum(abs(row["target_share"] - row["reference_share"]) for row in rows)
        output[column] = {"tvd": float(tvd), "rows": rows}
    return output
