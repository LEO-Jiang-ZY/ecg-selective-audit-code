"""按 device_group / site_group 做分组温度缩放与分组熵阈值。"""
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
from ptbxl.calibration import TemperatureScaler, entropy_abstention, select_entropy_threshold
from ptbxl.inference import load_run
from ptbxl.metrics import accepted_set_error, predictive_entropy


def group_fit(logits: np.ndarray, labels: np.ndarray, coverage: float) -> dict:
    scaler = TemperatureScaler().fit(logits, labels)
    calibrated = scaler.transform_logits(logits)
    threshold = select_entropy_threshold(calibrated, coverage, from_logits=True)
    mask = entropy_abstention(calibrated, threshold, from_logits=True)
    return {
        "temperature": scaler.temperature,
        "threshold": threshold,
        "coverage": float(mask.mean()),
        "accepted_any": accepted_set_error(labels, calibrated, mask, "any_label"),
        "accepted_macro": accepted_set_error(labels, calibrated, mask, "macro_label"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="result/runs/seed_42/config.yaml")
    parser.add_argument("--calibration", default="result/runs/seed_42/calibration.json")
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--min-val", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="result/analysis/group_calibration.csv")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    global_cal = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    global_t = float(global_cal["temperature"])
    global_threshold = float(global_cal["entropy_threshold"])

    val = load_run(args.config, "validation", device=args.device)
    test = load_run(args.config, "test", device=args.device)
    val_frame, test_frame = val["split_frame"], test["split_frame"]
    val_logits, val_labels = val["logits"].astype(float), val["labels"].astype(float)
    test_logits, test_labels = test["logits"].astype(float), test["labels"].astype(float)

    global_test_cal = test_logits / global_t
    global_test_mask = entropy_abstention(global_test_cal, global_threshold, from_logits=True)

    rows = []
    for column in ["device_group", "site_group"]:
        if column not in val_frame or column not in test_frame:
            continue
        grouped_t = np.full(len(test_frame), global_t)
        grouped_threshold = np.full(len(test_frame), global_threshold)
        for value in sorted(val_frame[column].astype(str).unique()):
            val_mask = val_frame[column].astype(str).to_numpy() == value
            test_mask = test_frame[column].astype(str).to_numpy() == value
            n_val, n_test = int(val_mask.sum()), int(test_mask.sum())
            if n_test == 0:
                continue
            if n_val >= args.min_val:
                fit = group_fit(val_logits[val_mask], val_labels[val_mask], args.coverage)
                grouped_t[test_mask] = fit["temperature"]
                grouped_threshold[test_mask] = fit["threshold"]
                status = "fitted"
            else:
                fit = {"temperature": global_t, "threshold": global_threshold, "coverage": float("nan")}
                status = "global_fallback"

            group_logits = test_logits[test_mask] / grouped_t[test_mask][0]
            entropy = predictive_entropy(group_logits, from_logits=True)
            group_mask = entropy <= grouped_threshold[test_mask][0]
            group_any = accepted_set_error(test_labels[test_mask], group_logits, group_mask, "any_label")
            global_group_cal = test_logits[test_mask] / global_t
            global_group_mask = global_test_mask[test_mask]
            global_any = accepted_set_error(test_labels[test_mask], global_group_cal, global_group_mask, "any_label")

            rows.append(
                {
                    "column": column,
                    "group": value,
                    "status": status,
                    "n_val": n_val,
                    "n_test": n_test,
                    "temperature": grouped_t[test_mask][0],
                    "threshold": grouped_threshold[test_mask][0],
                    "group_coverage": float(group_mask.mean()),
                    "global_any": global_any,
                    "group_any": group_any,
                    "delta_any": group_any - global_any,
                }
            )

    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    grouped_cal = test_logits / grouped_t[:, None]
    grouped_mask = predictive_entropy(grouped_cal, from_logits=True) <= grouped_threshold
    summary = {
        "global": {
            "coverage": float(global_test_mask.mean()),
            "accepted_any": accepted_set_error(test_labels, global_test_cal, global_test_mask, "any_label"),
            "accepted_macro": accepted_set_error(test_labels, global_test_cal, global_test_mask, "macro_label"),
        },
        "grouped": {
            "coverage": float(grouped_mask.mean()),
            "accepted_any": accepted_set_error(test_labels, grouped_cal, grouped_mask, "any_label"),
            "accepted_macro": accepted_set_error(test_labels, grouped_cal, grouped_mask, "macro_label"),
        },
    }
    output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"已写入 {output} 与 {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
