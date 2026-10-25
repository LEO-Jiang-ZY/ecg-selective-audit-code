"""分类、校准与选择性预测指标。"""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

ERROR_UNITS = ("any_label", "macro_label")


def _arrays(y_true, logits_or_probs, from_logits=True):
    y = np.asarray(y_true, dtype=float)
    x = np.asarray(logits_or_probs, dtype=float)
    p = expit(x) if from_logits else np.clip(x, 1e-7, 1 - 1e-7)
    if y.shape != p.shape:
        raise ValueError(f"形状不一致: y={y.shape}, prediction={p.shape}")
    return y, p


def _validate_error_unit(error_unit: str) -> str:
    if error_unit not in ERROR_UNITS:
        raise ValueError(f"未知 error_unit={error_unit!r}，可选 {ERROR_UNITS}")
    return error_unit


def multilabel_nll(y_true, logits_or_probs, from_logits=True) -> float:
    y, p = _arrays(y_true, logits_or_probs, from_logits)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def brier_score(y_true, logits_or_probs, from_logits=True) -> float:
    y, p = _arrays(y_true, logits_or_probs, from_logits)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y_true, logits_or_probs, n_bins: int = 15, from_logits=True) -> float:
    """将每个标签的概率视作二分类预测并按置信度分箱。"""
    y, p = _arrays(y_true, logits_or_probs, from_logits)
    confidence = np.maximum(p, 1 - p).ravel()
    correctness = (p >= 0.5).astype(float).ravel() == y.ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (confidence >= left) & ((confidence < right) if right < 1 else (confidence <= right))
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(correctness[mask].mean()))
    return float(ece)


def predictive_entropy(logits_or_probs, from_logits=True) -> np.ndarray:
    x = np.asarray(logits_or_probs, dtype=float)
    p = expit(x) if from_logits else np.clip(x, 1e-7, 1 - 1e-7)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p)).mean(axis=1)


def error_vector(y_true, logits_or_probs, error_unit: str = "any_label", threshold: float = 0.5, from_logits=True) -> np.ndarray:
    """返回逐记录的误差。

    any_label  : 任一标签错误记为 1，等价于 1 - exact_match。
    macro_label: 该记录五个标签错误的平均值，取值 [0, 1]。
    """
    _validate_error_unit(error_unit)
    y, p = _arrays(y_true, logits_or_probs, from_logits)
    wrong = (p >= threshold).astype(int) != y
    if error_unit == "any_label":
        return wrong.any(axis=1).astype(float)
    return wrong.mean(axis=1)


def accepted_set_error(
    y_true,
    logits_or_probs,
    accepted_mask=None,
    error_unit: str = "any_label",
    threshold: float = 0.5,
    from_logits=True,
) -> float:
    """在保留样本上计算 selective risk。mask=None 时使用全部样本。"""
    errors = error_vector(y_true, logits_or_probs, error_unit, threshold, from_logits)
    if accepted_mask is None:
        return float(errors.mean())
    mask = np.asarray(accepted_mask, dtype=bool)
    if mask.shape != errors.shape:
        raise ValueError(f"accepted_mask 形状 {mask.shape} 与样本数 {errors.shape} 不一致")
    if not mask.any():
        return float("nan")
    return float(errors[mask].mean())


def risk_coverage(
    y_true,
    logits_or_probs,
    coverages=None,
    error_unit: str = "any_label",
    threshold: float = 0.5,
    from_logits=True,
) -> dict[str, np.ndarray]:
    """按熵从低到高保留样本，返回 selective risk 曲线。"""
    _validate_error_unit(error_unit)
    y, p = _arrays(y_true, logits_or_probs, from_logits)
    order = np.argsort(predictive_entropy(p, from_logits=False))
    errors = error_vector(y, p, error_unit, threshold, from_logits=False)[order]
    cumulative = np.cumsum(errors)
    if coverages is None:
        coverages = np.linspace(1 / len(y), 1.0, min(len(y), 100))
    coverages = np.asarray(coverages, dtype=float)
    risks = np.array(
        [
            cumulative[max(1, int(np.ceil(c * len(y)))) - 1] / max(1, int(np.ceil(c * len(y))))
            for c in coverages
        ]
    )
    return {"coverage": coverages, "risk": risks, "order": order, "error_unit": error_unit}


def aurc(risk, coverage) -> float:
    """Risk-coverage 曲线下面积；越低越好。"""
    risk = np.asarray(risk, dtype=float)
    coverage = np.asarray(coverage, dtype=float)
    return float(np.trapz(risk, coverage))


def coverage_at_error(risk, coverage, alpha: float) -> float:
    """经验前沿上 accepted error <= alpha 的最大 coverage。"""
    risk = np.asarray(risk, dtype=float)
    coverage = np.asarray(coverage, dtype=float)
    valid = np.where(risk <= float(alpha))[0]
    return float(coverage[valid[-1]]) if valid.size else float("nan")


def macro_auroc(y_true, logits_or_probs, from_logits=True, class_names=None) -> dict:
    """逐标签 AUROC 及其宏平均；单一类别标签返回 NaN。"""
    y, p = _arrays(y_true, logits_or_probs, from_logits)
    names = list(class_names) if class_names is not None else [f"label_{i}" for i in range(y.shape[1])]
    if len(names) != y.shape[1]:
        raise ValueError(f"class_names 数量 {len(names)} 与标签数 {y.shape[1]} 不一致")
    per_class = {}
    for k, name in enumerate(names):
        if len(np.unique(y[:, k])) < 2:
            per_class[name] = float("nan")
        else:
            per_class[name] = float(roc_auc_score(y[:, k], p[:, k]))
    valid = [v for v in per_class.values() if np.isfinite(v)]
    return {"macro": float(np.mean(valid)) if valid else float("nan"), "per_class": per_class}


def macro_auprc(y_true, logits_or_probs, from_logits=True, class_names=None) -> dict:
    """逐标签 average precision 及其宏平均。"""
    y, p = _arrays(y_true, logits_or_probs, from_logits)
    names = list(class_names) if class_names is not None else [f"label_{i}" for i in range(y.shape[1])]
    if len(names) != y.shape[1]:
        raise ValueError(f"class_names 数量 {len(names)} 与标签数 {y.shape[1]} 不一致")
    per_class = {}
    for k, name in enumerate(names):
        if len(np.unique(y[:, k])) < 2:
            per_class[name] = float("nan")
        else:
            per_class[name] = float(average_precision_score(y[:, k], p[:, k]))
    valid = [v for v in per_class.values() if np.isfinite(v)]
    return {"macro": float(np.mean(valid)) if valid else float("nan"), "per_class": per_class}


def per_class_metrics(y_true, logits_or_probs, threshold: float = 0.5, from_logits=True, class_names=None) -> list[dict]:
    y, p = _arrays(y_true, logits_or_probs, from_logits)
    names = list(class_names) if class_names is not None else [f"label_{i}" for i in range(y.shape[1])]
    if len(names) != y.shape[1]:
        raise ValueError(f"class_names 数量 {len(names)} 与标签数 {y.shape[1]} 不一致")
    rows = []
    for k, name in enumerate(names):
        yk, pk = y[:, k], p[:, k]
        rows.append(
            {
                "class": name,
                "n_positive": int(yk.sum()),
                "prevalence": float(yk.mean()),
                "auroc": float(roc_auc_score(yk, pk)) if len(np.unique(yk)) > 1 else float("nan"),
                "auprc": float(average_precision_score(yk, pk)) if len(np.unique(yk)) > 1 else float("nan"),
                "nll": float(-np.mean(yk * np.log(pk) + (1 - yk) * np.log1p(-pk))),
                "brier": float(np.mean((pk - yk) ** 2)),
                "accuracy": float(np.mean((pk >= threshold).astype(int) == yk)),
            }
        )
    return rows


def summarize_metrics(y_true, logits_or_probs, threshold: float = 0.5, from_logits=True) -> dict[str, float]:
    y, p = _arrays(y_true, logits_or_probs, from_logits)
    pred = (p >= threshold).astype(int)
    wrong = pred != y
    return {
        "nll": multilabel_nll(y, logits_or_probs, from_logits),
        "brier": brier_score(y, logits_or_probs, from_logits),
        "ece": expected_calibration_error(y, logits_or_probs, from_logits=from_logits),
        "label_accuracy": float(np.mean(pred == y)),
        "exact_match": float(np.mean(np.all(pred == y, axis=1))),
        "any_label_error": float(wrong.any(axis=1).mean()),
        "macro_label_error": float(wrong.mean()),
    }
