"""在 fold 9 验证集 logits 上拟合温度并按目标 coverage 选择熵阈值。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptbxl.calibration import (
    TemperatureScaler,
    coverage_at_threshold,
    entropy_abstention,
    select_entropy_threshold,
)
from ptbxl.metrics import accepted_set_error, summarize_metrics
from ptbxl.utils import jsonable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="code/configs/default.yaml")
    parser.add_argument("--val-logits", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--target-coverage", type=float, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cal_cfg = cfg.get("calibration", {})
    val_path = Path(args.val_logits or cfg["training"].get("validation_logits", "result/results/val_logits.npz"))
    if not val_path.exists():
        raise FileNotFoundError(f"缺少验证集 logits: {val_path}；请先运行 scripts/train.py")

    payload = np.load(val_path)
    required = {"logits", "labels"}
    missing = required - set(payload.files)
    if missing:
        raise ValueError(f"{val_path} 缺少数组 {sorted(missing)}")
    logits = payload["logits"].astype(float)
    labels = payload["labels"].astype(float)
    if logits.shape != labels.shape:
        raise ValueError(f"logits {logits.shape} 与 labels {labels.shape} 形状不一致")
    split_name = str(payload["split"]) if "split" in payload.files else "unknown"
    if split_name not in {"validation", "fold9_validation"}:
        raise ValueError(f"校准必须使用 fold 9 验证集 logits，当前 split={split_name!r}")

    target_coverage = float(args.target_coverage if args.target_coverage is not None else cal_cfg.get("target_coverage", 0.80))
    scaler = TemperatureScaler().fit(logits, labels, max_iter=int(cal_cfg.get("max_iter", 100)))
    calibrated = scaler.transform_logits(logits)
    threshold = select_entropy_threshold(calibrated, target_coverage, from_logits=True)
    coverage = coverage_at_threshold(calibrated, threshold, from_logits=True)
    mask = entropy_abstention(calibrated, threshold, from_logits=True)

    threshold_uncalibrated = select_entropy_threshold(logits, target_coverage, from_logits=True)
    coverage_uncalibrated = coverage_at_threshold(logits, threshold_uncalibrated, from_logits=True)
    mask_uncalibrated = entropy_abstention(logits, threshold_uncalibrated, from_logits=True)

    report = {
        "fit_split": split_name,
        "source": str(val_path),
        "n": int(len(labels)),
        "target_coverage": target_coverage,
        "temperature": scaler.temperature,
        "entropy_threshold": threshold,
        "val_coverage": coverage,
        "accepted_fraction": float(mask.mean()),
        "accepted_error": {
            "any_label": accepted_set_error(labels, calibrated, mask, error_unit="any_label"),
            "macro_label": accepted_set_error(labels, calibrated, mask, error_unit="macro_label"),
        },
        "entropy_threshold_uncalibrated": threshold_uncalibrated,
        "val_coverage_uncalibrated": coverage_uncalibrated,
        "accepted_error_uncalibrated": {
            "any_label": accepted_set_error(labels, logits, mask_uncalibrated, error_unit="any_label"),
            "macro_label": accepted_set_error(labels, logits, mask_uncalibrated, error_unit="macro_label"),
        },
        "metrics_uncalibrated": summarize_metrics(labels, logits),
        "metrics_calibrated": summarize_metrics(labels, calibrated),
    }

    output = Path(args.output or cal_cfg.get("output", "result/results/calibration.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(jsonable(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
