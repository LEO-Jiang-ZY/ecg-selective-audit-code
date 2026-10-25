"""Correctness head：用特征预测"这条记录/这个标签是否会被判对"，作为拒答分数。

与熵、max-prob 等 post-hoc 分数不同，correctness head 是非单调、特征感知的
打分器，可以突破 logit 派生分数的排序上界（Rabanser & Papernot, 2025）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import error_vector


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p))


def _metadata_columns(frame: pd.DataFrame) -> np.ndarray:
    age = frame["age"].to_numpy(dtype=float)
    age = np.where(age > 89, 90.0, age)
    age = np.where(np.isfinite(age), age, np.nanmedian(age))
    age = (age - 60.0) / 20.0
    sex = frame["sex_label"].astype(str).to_numpy()
    sex_code = np.where(sex == "male", 1.0, np.where(sex == "female", 0.0, -1.0))
    quality = frame["quality_any"].astype(str).to_numpy()
    quality_code = np.where(quality == "noise", 1.0, 0.0)
    return np.column_stack([age, sex_code, quality_code]).astype(np.float64)


def record_features(backbone: np.ndarray, logits: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    """记录级设计矩阵：backbone + logits + 概率统计 + 元数据。"""
    p = expit(logits)
    ent = binary_entropy(p)
    confident = np.maximum(p, 1 - p)
    stats = np.column_stack(
        [
            ent.mean(axis=1),
            ent.max(axis=1),
            ent.min(axis=1),
            confident.mean(axis=1),
            confident.min(axis=1),
            np.prod(confident, axis=1),
            (p > 0.5).sum(axis=1).astype(float),
            np.abs(p - 0.5).mean(axis=1),
            (ent > 0.3).sum(axis=1).astype(float),
            (ent > 0.6).sum(axis=1).astype(float),
        ]
    )
    return np.column_stack([backbone, logits, p, ent, stats, _metadata_columns(frame)])


def label_features(
    backbone: np.ndarray,
    logits: np.ndarray,
    n_labels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """标签级设计矩阵：每个 (记录, 标签) 一行；返回特征与标签索引。"""
    p = expit(logits)
    ent = binary_entropy(p)
    confident = np.maximum(p, 1 - p)
    n = len(logits)
    blocks, indices = [], []
    for k in range(n_labels):
        one_hot = np.zeros((n, n_labels), dtype=np.float64)
        one_hot[:, k] = 1.0
        block = np.column_stack(
            [
                backbone,
                logits,
                p,
                ent,
                confident,
                confident.mean(axis=1),
                confident.min(axis=1),
                np.prod(confident, axis=1),
                logits[:, k],
                p[:, k],
                ent[:, k],
                confident[:, k],
                one_hot,
            ]
        )
        blocks.append(block)
        indices.append(np.full(n, k, dtype=np.int64))
    return np.vstack(blocks), np.concatenate(indices)


def grouped_folds(groups: np.ndarray, stratify: np.ndarray, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """患者级分层折：按患者正例率排序后蛇形分配，保证每折正例率接近。"""
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({"g": groups, "s": np.asarray(stratify, dtype=float)})
    means = table.groupby("g")["s"].mean()
    order = rng.permutation(means.index.to_numpy())
    ordered = means.loc[order].sort_values(kind="stable").index.to_numpy()
    fold_of: dict = {}
    for position, group in enumerate(ordered):
        cycle, slot = divmod(position, n_splits)
        fold_of[group] = n_splits - 1 - slot if cycle % 2 else slot
    return np.array([fold_of[g] for g in groups], dtype=np.int64)


@dataclass
class CorrectnessHead:
    kind: str = "gbdt"
    seed: int = 42
    model: object | None = None
    fold_scores: np.ndarray | None = None
    oof_auc: float = float("nan")
    fold_auc: list = field(default_factory=list)

    def _make(self):
        if self.kind == "logreg":
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
            )
        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=self.seed,
        )

    def fit_oof(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5, seed: int = 42) -> np.ndarray:
        """患者级交叉拟合，返回折外分数（越大越可信）。"""
        folds = grouped_folds(groups, y, n_splits=n_splits, seed=seed)
        scores = np.full(len(y), np.nan)
        for fold in range(n_splits):
            train = folds != fold
            valid = folds == fold
            model = self._make()
            model.fit(X[train], y[train])
            scores[valid] = model.predict_proba(X[valid])[:, 1]
        self.fold_scores = scores
        self.folds = folds
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y)) > 1:
            self.oof_auc = float(roc_auc_score(y, scores))
        return scores

    def fit_full(self, X: np.ndarray, y: np.ndarray) -> "CorrectnessHead":
        self.model = self._make()
        self.model.fit(X, y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("先调用 fit_full")
        return self.model.predict_proba(X)[:, 1]


def risk_coverage_from_score(y: np.ndarray, p: np.ndarray, score: np.ndarray, n_points: int = 200) -> dict:
    """score 越大越可信（correctness head 的方向）；返回等距 coverage 上的 selective risk。"""
    order = np.argsort(-np.asarray(score, dtype=float))
    errors = error_vector(y, p, "any_label", 0.5, from_logits=False)[order]
    cumulative = np.cumsum(errors)
    coverages = np.linspace(1 / len(y), 1.0, n_points)
    risks = np.array(
        [
            cumulative[max(1, int(np.ceil(c * len(y)))) - 1] / max(1, int(np.ceil(c * len(y))))
            for c in coverages
        ]
    )
    return {"coverage": coverages, "risk": risks}


def lowest_score_threshold(score: np.ndarray, coverage: float) -> float:
    """选阈值使保留比例最接近目标 coverage。"""
    target = int(round(coverage * len(score)))
    ordered = np.sort(score)[::-1]
    target = min(max(target, 1), len(ordered))
    return float(ordered[target - 1])
