"""在 fold 9 上调整逐类概率阈值，重点观察 HYP。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptbxl.inference import calibrated_logits, load_run
from ptbxl.metrics import accepted_set_error


def probabilities(logits: np.ndarray) -> np.ndarray:
    return expit(np.asarray(logits, dtype=float))


def predict_with(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (probs >= thresholds).astype(int)


def record_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    wrong = pred != y
    return {
        "exact_match": float(np.mean(np.all(~wrong, axis=1))),
        "any_label_error": float(wrong.any(axis=1).mean()),
        "macro_label_error": float(wrong.mean()),
    }


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate(y: np.ndarray, probs: np.ndarray, thresholds: np.ndarray, classes: list[str]) -> dict:
    pred = predict_with(probs, thresholds)
    output = record_metrics(y, pred)
    output["per_class"] = {
        name: binary_metrics(y[:, index], pred[:, index]) for index, name in enumerate(classes)
    }
    return output


def tune_hyp(y: np.ndarray, probs: np.ndarray, classes: list[str], index: int, grid: np.ndarray) -> tuple[float, dict]:
    best = None
    for threshold in grid:
        thresholds = np.full(len(classes), 0.5)
        thresholds[index] = threshold
        metrics = record_metrics(y, predict_with(probs, thresholds))
        hyp = binary_metrics(y[:, index], (probs[:, index] >= threshold).astype(int))
        key = (metrics["any_label_error"], -hyp["f1"])
        if best is None or key < best[0]:
            best = (key, {"threshold": float(threshold), **metrics, "hyp": hyp})
    return best[1]["threshold"], best[1]


def tune_coordinate(y: np.ndarray, probs: np.ndarray, classes: list[str], grid: np.ndarray, passes: int = 3) -> tuple[np.ndarray, dict]:
    thresholds = np.full(len(classes), 0.5)
    best = record_metrics(y, predict_with(probs, thresholds))
    for _ in range(passes):
        improved = False
        for index in range(len(classes)):
            local = (best["any_label_error"], best["macro_label_error"])
            for threshold in grid:
                candidate = thresholds.copy()
                candidate[index] = threshold
                metrics = record_metrics(y, predict_with(probs, candidate))
                key = (metrics["any_label_error"], metrics["macro_label_error"])
                if key < local:
                    local = key
                    thresholds = candidate
                    best = metrics
                    improved = True
        if not improved:
            break
    return thresholds, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="result/runs/seed_42/config.yaml")
    parser.add_argument("--calibration", default="result/runs/seed_42/calibration.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="result/analysis/class_thresholds.json")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    classes = cfg["data"]["classes"]
    val = np.load(Path(cfg["training"]["validation_logits"]))
    val_logits, _ = calibrated_logits(val["logits"].astype(float), args.calibration)
    val_labels = val["labels"].astype(float)
    val_probs = probabilities(val_logits)

    test = load_run(args.config, "test", device=args.device)
    test_logits, _ = calibrated_logits(test["logits"].astype(float), args.calibration)
    test_labels = test["labels"].astype(float)
    test_probs = probabilities(test_logits)

    grid = np.round(np.arange(0.05, 0.951, 0.05), 2)
    hyp_index = classes.index("HYP") if "HYP" in classes else len(classes) - 1

    baseline_thresholds = np.full(len(classes), 0.5)
    hyp_threshold, hyp_val = tune_hyp(val_labels, val_probs, classes, hyp_index, grid)
    hyp_thresholds = baseline_thresholds.copy()
    hyp_thresholds[hyp_index] = hyp_threshold
    coordinate_thresholds, coordinate_val = tune_coordinate(val_labels, val_probs, classes, grid)

    schemes = {
        "baseline_0.5": baseline_thresholds,
        "hyp_tuned": hyp_thresholds,
        "coordinate": coordinate_thresholds,
    }
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    entropy_threshold = calibration.get("entropy_threshold")
    entropy = -(np.clip(test_probs, 1e-7, 1 - 1e-7) * np.log(np.clip(test_probs, 1e-7, 1 - 1e-7))
                + (1 - np.clip(test_probs, 1e-7, 1 - 1e-7)) * np.log1p(-np.clip(test_probs, 1e-7, 1 - 1e-7))).mean(axis=1)
    accepted_mask = entropy <= float(entropy_threshold) if entropy_threshold is not None else np.ones(len(entropy), dtype=bool)

    output = {"grid": grid.tolist(), "hyp_threshold": hyp_threshold, "schemes": {}}
    for name, thresholds in schemes.items():
        val_metrics = evaluate(val_labels, val_probs, thresholds, classes)
        test_metrics = evaluate(test_labels, test_probs, thresholds, classes)
        test_pred = predict_with(test_probs, thresholds)
        errors = (test_pred != test_labels)
        accepted_any = float(errors[accepted_mask].any(axis=1).mean()) if accepted_mask.any() else float("nan")
        accepted_macro = float(errors[accepted_mask].mean()) if accepted_mask.any() else float("nan")
        output["schemes"][name] = {
            "thresholds": {cls: float(value) for cls, value in zip(classes, thresholds)},
            "val": val_metrics,
            "test": test_metrics,
            "test_accepted": {
                "coverage": float(accepted_mask.mean()),
                "any_label": accepted_any,
                "macro_label": accepted_macro,
            },
        }

    output["hyp_tuning"] = hyp_val
    output["coordinate_val"] = coordinate_val
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for name, payload in output["schemes"].items():
        rows.append(
            {
                "scheme": name,
                "hyp_threshold": payload["thresholds"].get("HYP"),
                "val_any": payload["val"]["any_label_error"],
                "test_any": payload["test"]["any_label_error"],
                "test_exact": payload["test"]["exact_match"],
                "test_macro": payload["test"]["macro_label_error"],
                "test_accepted_any": payload["test_accepted"]["any_label"],
                "test_accepted_macro": payload["test_accepted"]["macro_label"],
                "hyp_f1": payload["test"]["per_class"]["HYP"]["f1"],
                "hyp_precision": payload["test"]["per_class"]["HYP"]["precision"],
                "hyp_recall": payload["test"]["per_class"]["HYP"]["recall"],
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(path.with_suffix(".csv"), index=False)
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"已写入 {path} 与 {path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
