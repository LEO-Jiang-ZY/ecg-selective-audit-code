"""比较熵、max-prob、margin 等拒答分数在 80% coverage 下的表现。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptbxl.inference import calibrated_logits, load_run
from ptbxl.metrics import accepted_set_error


def entropy_score(logits: np.ndarray) -> np.ndarray:
    p = expit(logits)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p)).mean(axis=1)


def max_prob_score(logits: np.ndarray) -> np.ndarray:
    p = expit(logits)
    return np.maximum(p, 1 - p).mean(axis=1)


def margin_score(logits: np.ndarray) -> np.ndarray:
    p = expit(logits)
    return np.abs(p - 0.5).mean(axis=1)


def min_confidence_score(logits: np.ndarray) -> np.ndarray:
    p = expit(logits)
    return np.maximum(p, 1 - p).min(axis=1)


def all_correct_score(logits: np.ndarray) -> np.ndarray:
    """五个标签全部正确的独立近似概率；越高越可信。"""
    p = expit(logits)
    return np.prod(np.maximum(p, 1 - p), axis=1)


def expected_error_score(logits: np.ndarray) -> np.ndarray:
    """期望标签错误数之和；越低越可信。"""
    p = expit(logits)
    return np.minimum(p, 1 - p).sum(axis=1)


def max_label_entropy_score(logits: np.ndarray) -> np.ndarray:
    """最不确定标签的熵；越低越可信。"""
    p = expit(logits)
    entropy = -(p * np.log(p) + (1 - p) * np.log1p(-p))
    return entropy.max(axis=1)


def n_uncertain_score(logits: np.ndarray, low: float = 0.3, high: float = 0.7) -> np.ndarray:
    """落在 [low, high] 的标签个数；越少越可信。"""
    p = expit(logits)
    return ((p > low) & (p < high)).sum(axis=1).astype(float)


SCORES = {
    "entropy": (entropy_score, "lower"),
    "max_prob": (max_prob_score, "higher"),
    "margin": (margin_score, "higher"),
    "min_confidence": (min_confidence_score, "higher"),
    "all_correct_prob": (all_correct_score, "higher"),
    "expected_errors": (expected_error_score, "lower"),
    "max_label_entropy": (max_label_entropy_score, "lower"),
    "n_uncertain": (n_uncertain_score, "lower"),
}


def threshold_for(val_scores: np.ndarray, direction: str, coverage: float) -> float:
    if direction == "lower":
        return float(np.quantile(val_scores, coverage, method="higher"))
    return float(np.quantile(val_scores, 1 - coverage, method="lower"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="result/runs/seed_42/config.yaml")
    parser.add_argument("--calibration", default="result/runs/seed_42/calibration.json")
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="result/analysis/abstention_compare.csv")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    val_path = Path(cfg["training"]["validation_logits"])
    val = np.load(val_path)
    val_logits, val_labels = val["logits"].astype(float), val["labels"].astype(float)
    val_logits, _ = calibrated_logits(val_logits, args.calibration)

    test = load_run(args.config, "test", device=args.device)
    test_logits, _ = calibrated_logits(test["logits"].astype(float), args.calibration)
    test_labels = test["labels"].astype(float)

    rows = []
    for name, (function, direction) in SCORES.items():
        val_scores = function(val_logits)
        test_scores = function(test_logits)
        threshold = threshold_for(val_scores, direction, args.coverage)
        val_mask = val_scores <= threshold if direction == "lower" else val_scores >= threshold
        test_mask = test_scores <= threshold if direction == "lower" else test_scores >= threshold
        rows.append(
            {
                "score": name,
                "direction": direction,
                "threshold": threshold,
                "val_coverage": float(val_mask.mean()),
                "test_coverage": float(test_mask.mean()),
                "n_accepted": int(test_mask.sum()),
                "accepted_any_label": accepted_set_error(test_labels, test_logits, test_mask, "any_label"),
                "accepted_macro_label": accepted_set_error(test_labels, test_logits, test_mask, "macro_label"),
            }
        )

    frame = pd.DataFrame(rows).sort_values("accepted_any_label")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"已写入 {output}")


if __name__ == "__main__":
    main()
