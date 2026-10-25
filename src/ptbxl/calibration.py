"""多标签 logits 的单一温度缩放与熵拒识。"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit

from .metrics import multilabel_nll, predictive_entropy


class TemperatureScaler:
    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(self, logits, y_true, bounds=(0.05, 20.0), max_iter: int = 100):
        logits = np.asarray(logits, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        objective = lambda log_t: multilabel_nll(y_true, logits / np.exp(log_t), from_logits=True)
        result = minimize_scalar(objective, bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded", options={"maxiter": max_iter})
        self.temperature = float(np.exp(result.x))
        return self

    def transform_logits(self, logits):
        return np.asarray(logits) / self.temperature

    def predict_proba(self, logits):
        return expit(self.transform_logits(logits))


def select_entropy_threshold(logits_or_probs, target_coverage: float = 0.80, from_logits: bool = True) -> float:
    """在给定样本上选取熵阈值，使保留比例不低于 target_coverage。

    只能在验证集（fold 9）上调用；不要在测试集上重新选阈。
    """
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError(f"target_coverage 必须在 (0, 1] 内，实际为 {target_coverage}")
    entropy = predictive_entropy(logits_or_probs, from_logits)
    return float(np.quantile(entropy, target_coverage, method="higher"))


def coverage_at_threshold(logits_or_probs, threshold: float | None, from_logits: bool = True) -> float:
    """返回熵阈值对应的保留比例；threshold=None 表示不拒识。"""
    if threshold is None:
        return 1.0
    entropy = predictive_entropy(logits_or_probs, from_logits)
    return float((entropy <= float(threshold)).mean())


def entropy_abstention(logits_or_probs, threshold: float | None, from_logits: bool = True) -> np.ndarray:
    """返回保留掩码；threshold=None 时不拒识。"""
    entropy = predictive_entropy(logits_or_probs, from_logits)
    if threshold is None:
        return np.ones(len(entropy), dtype=bool)
    return entropy <= float(threshold)


def fit_temperature_and_threshold(
    logits,
    labels,
    target_coverage: float = 0.80,
    max_iter: int = 100,
) -> tuple[TemperatureScaler, float, float]:
    """在验证集上拟合温度并按 coverage 选熵阈值。"""
    scaler = TemperatureScaler().fit(logits, labels, max_iter=max_iter)
    calibrated = scaler.transform_logits(logits)
    threshold = select_entropy_threshold(calibrated, target_coverage, from_logits=True)
    coverage = coverage_at_threshold(calibrated, threshold, from_logits=True)
    return scaler, threshold, coverage
