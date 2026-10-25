"""误差预算的唯一定义。

论文里凡出现「6.9 pp」「23.1%」的地方，都必须由这里的函数生成，避免两个脚本
各用一套取整规则（此前 decompose_gap 用 ceil、gap_closure 用 round，导致
oracle 在 23.1% 与 23.6% 之间摆动）。

约定：
    correct[i]  —— 第 i 条记录是否**所有**标签都判对（多标签的 any-label 口径）
    score[i]    —— 置信度，越大越可信；「越低越可信」的分数取负后再传入
    coverage c  —— 取置信度最高的 ceil/round(c·n) 条

    R(θ, c)     = 1 − mean(correct[top_k(θ)])
    R_oracle(c) = R(按 correct 排序, c)      任何排序都无法再降低的下界
    budget(c)   = R_current(c) − R_oracle(c)
    recovered   = (R_current − R_θ) / budget
"""
from __future__ import annotations

import numpy as np


def top_k(coverage: float, n: int) -> int:
    """保留条数。统一用 round，并在文档中声明。"""
    return int(min(n, max(1, round(coverage * n))))


def risk_at_coverage(score: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    k = top_k(coverage, len(correct))
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    return float(1.0 - np.asarray(correct, dtype=float)[order[:k]].mean())


def oracle_risk(correct: np.ndarray, coverage: float) -> float:
    return risk_at_coverage(np.asarray(correct, dtype=float), correct, coverage)


def budget(correct: np.ndarray, current_score: np.ndarray, coverage: float) -> float:
    return risk_at_coverage(current_score, correct, coverage) - oracle_risk(correct, coverage)


def recovered(correct: np.ndarray, current_score: np.ndarray, score: np.ndarray, coverage: float) -> float:
    r_oracle = oracle_risk(correct, coverage)
    r_current = risk_at_coverage(current_score, correct, coverage)
    r_score = risk_at_coverage(score, correct, coverage)
    denom = r_current - r_oracle
    return float((r_current - r_score) / denom) if denom > 1e-12 else float("nan")


def error_count_bounds(correct: np.ndarray, coverage: float) -> dict:
    """一致性自检：由错误总数直接推出的 oracle 下界，应与 oracle_risk 相等。"""
    n = len(correct)
    k = top_k(coverage, n)
    errors = int((1 - np.asarray(correct, dtype=float)).sum())
    excluded = n - k
    closed_form = max(0.0, (errors - excluded) / k)
    return {"n": n, "k": k, "errors": errors, "closed_form": closed_form,
            "oracle": oracle_risk(correct, coverage)}
