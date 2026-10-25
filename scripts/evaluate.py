"""在 fold 10 上评估：主终点 any-label、校准、拒答、亚组与 device/site 偏移。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptbxl.analysis import (
    bootstrap_metric,
    bootstrap_selective_risk,
    distribution_shift_table,
    subgroup_metrics,
)
from ptbxl.cache import cache_rows_for, load_cache
from ptbxl.calibration import TemperatureScaler, entropy_abstention
from ptbxl.cohort import derive_cohort_columns
from ptbxl.labels import load_and_label
from ptbxl.metrics import (
    accepted_set_error,
    macro_auprc,
    macro_auroc,
    per_class_metrics,
    risk_coverage,
    summarize_metrics,
)
from ptbxl.model import build_model
from ptbxl.split import patient_fold_split
from ptbxl.train import make_loader, predict
from ptbxl.utils import jsonable


def _load_calibration(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "temperature" not in payload:
        raise ValueError(f"{path} 缺少 temperature")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="code/configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--limit", type=int, default=None, help="仅评估前 N 条测试记录，用于冒烟测试")
    parser.add_argument("--output", default=None)
    parser.add_argument("--allow-uncalibrated", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=None)
    parser.add_argument("--group-columns", nargs="+", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["training"]
    eval_cfg = cfg.get("evaluation", {})
    labeling = cfg.get("labeling", {})
    root = Path(data_cfg["root"])

    frame = load_and_label(
        root / data_cfg["metadata"],
        root / data_cfg["scp_statements"],
        data_cfg["classes"],
        min_scp_probability=labeling.get("min_scp_probability"),
        drop_all_zero=bool(labeling.get("drop_all_zero", False)),
    )
    frame = derive_cohort_columns(frame)
    splits = patient_fold_split(frame, data_cfg["folds"])
    test_full = splits["test"]
    test = test_full.head(args.limit) if args.limit else test_full

    cache = None
    test_cache_rows = None
    cache_path = data_cfg.get("cache")
    cache_index_path = data_cfg.get("cache_index")
    if cache_path and cache_index_path and Path(cache_path).exists() and Path(cache_index_path).exists():
        cache = load_cache(cache_path)
        cache_index = pd.read_csv(cache_index_path)
        test_cache_rows = cache_rows_for(test, cache_index)
        print(f"使用波形缓存 {cache_path}（{len(cache)} 条）", flush=True)

    loader = make_loader(test, root, data_cfg["classes"], data_cfg["batch_size"], data_cfg["target_length"], data_cfg["sampling_rate"], False, data_cfg.get("num_workers", 0), cache=cache, cache_rows=test_cache_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg, num_classes=len(data_cfg["classes"])).to(device)

    checkpoint_value = args.checkpoint or train_cfg.get("checkpoint")
    if not checkpoint_value:
        raise ValueError("必须提供 --checkpoint 或 training.checkpoint")
    checkpoint = Path(checkpoint_value)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}；拒绝用随机初始化模型评估")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    expected = state.get("num_classes")
    if expected is not None and int(expected) != len(data_cfg["classes"]):
        raise ValueError(f"checkpoint 类别数 {expected} 与配置 {len(data_cfg['classes'])} 不一致")
    model.load_state_dict(state["model"])
    y_true, logits = predict(model, loader, device)

    calibration_path = Path(args.calibration or cfg.get("calibration", {}).get("output", "result/results/calibration.json"))
    calibration = None
    if calibration_path.exists():
        calibration = _load_calibration(calibration_path)
        temperature = float(calibration["temperature"])
        entropy_threshold = calibration.get("entropy_threshold")
    elif args.allow_uncalibrated:
        temperature = 1.0
        entropy_threshold = None
    else:
        raise FileNotFoundError(
            f"缺少校准文件 {calibration_path}；请先运行 scripts/fit_calibration.py，"
            "或显式加 --allow-uncalibrated 做冒烟测试"
        )

    calibrated = TemperatureScaler(temperature).transform_logits(logits)
    accepted_mask = entropy_abstention(calibrated, entropy_threshold, from_logits=True)

    probability_threshold = float(eval_cfg.get("probability_threshold", 0.5))
    primary = str(eval_cfg.get("primary_error_unit", "any_label"))
    secondary = [str(unit) for unit in eval_cfg.get("secondary_error_units", ["macro_label"])]
    units = [primary] + [unit for unit in secondary if unit != primary]

    overall = summarize_metrics(y_true, calibrated, threshold=probability_threshold)
    baseline_metrics = summarize_metrics(y_true, logits, threshold=probability_threshold)
    baseline = {
        "metrics": baseline_metrics,
        "coverage": 1.0,
        "accepted_error": {
            unit: accepted_set_error(y_true, logits, None, error_unit=unit, threshold=probability_threshold)
            for unit in units
        },
        "risk_coverage": {},
    }
    for unit in units:
        baseline_curve = risk_coverage(y_true, logits, error_unit=unit, threshold=probability_threshold)
        baseline["risk_coverage"][unit] = {
            "coverage": baseline_curve["coverage"].tolist(),
            "risk": baseline_curve["risk"].tolist(),
        }
    if calibration and calibration.get("entropy_threshold_uncalibrated") is not None:
        baseline_mask = entropy_abstention(
            logits,
            calibration["entropy_threshold_uncalibrated"],
            from_logits=True,
        )
        baseline["matched_coverage"] = {
            "coverage": float(baseline_mask.mean()),
            "n_accepted": int(baseline_mask.sum()),
            "accepted_error": {
                unit: accepted_set_error(
                    y_true,
                    logits,
                    baseline_mask,
                    error_unit=unit,
                    threshold=probability_threshold,
                )
                for unit in units
            },
        }

    accepted = {
        "coverage": float(accepted_mask.mean()),
        "n_accepted": int(accepted_mask.sum()),
        "error_unit": primary,
    }
    for unit in units:
        accepted[unit] = accepted_set_error(
            y_true,
            calibrated,
            accepted_mask,
            error_unit=unit,
            threshold=probability_threshold,
        )

    curves = {}
    for unit in units:
        curve = risk_coverage(y_true, calibrated, error_unit=unit, threshold=probability_threshold)
        curves[unit] = {"coverage": curve["coverage"].tolist(), "risk": curve["risk"].tolist()}

    bootstrap_cfg = cfg.get("bootstrap", {})
    n_resamples = int(args.bootstrap_resamples if args.bootstrap_resamples is not None else bootstrap_cfg.get("n_resamples", 1000))
    seed = int(bootstrap_cfg.get("seed", 42))
    confidence = float(bootstrap_cfg.get("confidence", 0.95))
    patient_ids = test["patient_id"].to_numpy() if "patient_id" in test else None
    bootstrap = {
        "selective_risk": bootstrap_selective_risk(
            y_true,
            calibrated,
            float(entropy_threshold) if entropy_threshold is not None else float("inf"),
            error_unit=primary,
            patient_ids=patient_ids,
            n_resamples=n_resamples,
            seed=seed,
            confidence=confidence,
        ),
        "exact_match": bootstrap_metric(
            y_true,
            calibrated,
            metric="exact_match",
            n_resamples=n_resamples,
            seed=seed,
            confidence=confidence,
            patient_ids=patient_ids,
        ),
    }

    group_columns = args.group_columns or cfg.get("subgroups", {}).get("columns", [])
    subgroups = {}
    for column in group_columns:
        if column not in test:
            subgroups[column] = {"error": "column missing"}
            continue
        table = subgroup_metrics(
            test.reset_index(drop=True),
            y_true,
            calibrated,
            column,
            probability_threshold=probability_threshold,
            error_unit=primary,
            accepted_mask=accepted_mask,
        )
        subgroups[column] = table.to_dict(orient="records")

    shift_columns = cfg.get("shift", {}).get("columns", [])
    shift = distribution_shift_table(splits["train"], test_full, shift_columns) if shift_columns else {}

    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": state.get("epoch"),
        "checkpoint_validation_loss": state.get("validation_loss"),
        "calibrated": calibration is not None,
        "calibration_source": str(calibration_path) if calibration is not None else None,
        "temperature": temperature,
        "entropy_threshold": entropy_threshold,
        "target_coverage": calibration.get("target_coverage") if calibration else None,
        "n_test": int(len(y_true)),
        "n_test_full": int(len(test_full)),
        "test_limited": args.limit is not None,
        "n_patients": int(test["patient_id"].nunique()) if "patient_id" in test else None,
        "primary_error_unit": primary,
        "secondary_error_units": [unit for unit in units if unit != primary],
        "probability_threshold": probability_threshold,
        "overall": overall,
        "baseline": baseline,
        "accepted": accepted,
        "auroc": macro_auroc(y_true, calibrated, class_names=data_cfg["classes"]),
        "auprc": macro_auprc(y_true, calibrated, class_names=data_cfg["classes"]),
        "per_class": per_class_metrics(y_true, calibrated, threshold=probability_threshold, class_names=data_cfg["classes"]),
        "risk_coverage": curves,
        "bootstrap": bootstrap,
        "subgroups": subgroups,
        "shift": shift,
    }

    output = Path(args.output or eval_cfg.get("output", "result/results/evaluation.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "n_test": report["n_test"],
        "temperature": temperature,
        "coverage": accepted["coverage"],
        "accepted_error": {unit: accepted[unit] for unit in units},
        "auroc_macro": report["auroc"]["macro"],
        "auprc_macro": report["auprc"]["macro"],
    }
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))
    print(f"已写入 {output}")


if __name__ == "__main__":
    main()
