"""风险目标阈值：用 max-prob 分数，在 fold 9 上选满足目标风险的阈值。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptbxl.inference import calibrated_logits, load_run
from ptbxl.metrics import accepted_set_error, error_vector


def max_prob_score(logits: np.ndarray) -> np.ndarray:
    p = expit(logits)
    return np.maximum(p, 1 - p).mean(axis=1)


def cp_upper(errors: int, total: int, confidence: float = 0.95) -> float:
    """单侧 Clopper-Pearson 上界。"""
    if total <= 0:
        return 1.0
    if errors >= total:
        return 1.0
    return float(beta.ppf(confidence, errors + 1, total - errors))


def choose_threshold(scores: np.ndarray, errors: np.ndarray, alpha: float, confidence: float = 0.95) -> dict:
    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99), method="lower"))
    best = None
    fallback = None
    for threshold in candidates:
        mask = scores >= threshold
        total = int(mask.sum())
        if total == 0:
            continue
        count = int(errors[mask].sum())
        ucb = cp_upper(count, total, confidence)
        candidate = {
            "threshold": float(threshold),
            "val_coverage": float(mask.mean()),
            "val_error": float(count / total),
            "val_ucb": ucb,
            "val_accepted": total,
        }
        if fallback is None or ucb < fallback["val_ucb"]:
            fallback = candidate
        if ucb <= alpha and (best is None or candidate["val_coverage"] > best["val_coverage"]):
            best = candidate
    if best is None:
        best = dict(fallback)
        best["achievable"] = False
    else:
        best["achievable"] = True
    return best


def bootstrap_error(y_true, logits, mask, patient_ids, n_resamples: int = 1000, seed: int = 42):
    errors = error_vector(y_true, logits, "any_label")
    patients = np.asarray(patient_ids)
    unique = np.unique(patients)
    groups = {patient: np.where(patients == patient)[0] for patient in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([groups[patient] for patient in chosen])
        selected = mask[indices]
        if selected.any():
            values.append(float(errors[indices][selected].mean()))
    values = np.asarray(values)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="result/runs/seed_42/config.yaml")
    parser.add_argument("--calibration", default="result/runs/seed_42/calibration.json")
    parser.add_argument("--targets", nargs="+", type=float, default=[0.15, 0.20, 0.25, 0.30])
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="result/analysis/risk_target.csv")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    val = np.load(Path(cfg["training"]["validation_logits"]))
    val_logits, _ = calibrated_logits(val["logits"].astype(float), args.calibration)
    val_labels = val["labels"].astype(float)
    val_scores = max_prob_score(val_logits)
    val_errors = error_vector(val_labels, val_logits, "any_label")

    test = load_run(args.config, "test", device=args.device)
    test_logits, _ = calibrated_logits(test["logits"].astype(float), args.calibration)
    test_labels = test["labels"].astype(float)
    test_scores = max_prob_score(test_logits)
    patient_ids = test["split_frame"]["patient_id"].to_numpy()

    rows = []
    for alpha in args.targets:
        chosen = choose_threshold(val_scores, val_errors, alpha, args.confidence)
        mask = test_scores >= chosen["threshold"]
        test_error = accepted_set_error(test_labels, test_logits, mask, "any_label")
        test_macro = accepted_set_error(test_labels, test_logits, mask, "macro_label")
        low, high = bootstrap_error(test_labels, test_logits, mask, patient_ids)
        rows.append(
            {
                "target_risk": alpha,
                "achievable": chosen["achievable"],
                "threshold": chosen["threshold"],
                "val_coverage": chosen["val_coverage"],
                "val_error": chosen["val_error"],
                "val_ucb": chosen["val_ucb"],
                "test_coverage": float(mask.mean()),
                "test_error": test_error,
                "test_macro": test_macro,
                "ci_low": low,
                "ci_high": high,
                "n_accepted": int(mask.sum()),
            }
        )

    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    reference = json.loads(Path("result/results/evaluation.json").read_text(encoding="utf-8"))
    payload = {
        "score": "max_prob",
        "confidence": args.confidence,
        "targets": rows,
        "reference_80pct_entropy": {
            "coverage": reference["accepted"]["coverage"],
            "any_label": reference["accepted"]["any_label"],
            "macro_label": reference["accepted"]["macro_label"],
            "ci_low": reference["bootstrap"]["selective_risk"]["lower"],
            "ci_high": reference["bootstrap"]["selective_risk"]["upper"],
        },
    }
    output.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"已写入 {output} 与 {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
