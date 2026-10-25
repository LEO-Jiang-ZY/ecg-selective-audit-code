"""A：学习型 correctness head，与 entropy / max-prob / all-correct 对比。

协议：
- 训练：fold 1-8 的**折外**特征（由 oof_models.py 产出），患者级 5 折交叉拟合估计泛化
- 选阈：fold 9，固定 80% coverage
- 评估：fold 10，仅当 fold 9 判定为正才执行
"""
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
from ptbxl.cohort import derive_cohort_columns
from ptbxl.labels import load_and_label
from ptbxl.metrics import aurc, coverage_at_error
from ptbxl.selective import (
    CorrectnessHead,
    binary_entropy,
    label_features,
    lowest_score_threshold,
    record_features,
    risk_coverage_from_score,
)
from ptbxl.utils import jsonable


def load_frame(config_path: str) -> pd.DataFrame:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    root = Path(cfg["data"]["root"])
    frame = load_and_label(
        root / cfg["data"]["metadata"], root / cfg["data"]["scp_statements"],
        cfg["data"]["classes"],
        min_scp_probability=cfg.get("labeling", {}).get("min_scp_probability"),
        drop_all_zero=bool(cfg.get("labeling", {}).get("drop_all_zero", False)),
    )
    return derive_cohort_columns(frame)


def aligned_metadata(frame: pd.DataFrame, ecg_ids: np.ndarray) -> pd.DataFrame:
    return frame.loc[ecg_ids].reset_index(drop=True)


def evaluate_scores(y: np.ndarray, p: np.ndarray, scores: dict[str, np.ndarray], coverages=(0.15, 0.20, 0.80)) -> pd.DataFrame:
    rows = []
    for name, score in scores.items():
        curve = risk_coverage_from_score(y, p, score)
        rows.append(
            {
                "score": name,
                "aurc": aurc(curve["risk"], curve["coverage"]),
                "coverage_at_15": coverage_at_error(curve["risk"], curve["coverage"], 0.15),
                "coverage_at_20": coverage_at_error(curve["risk"], curve["coverage"], 0.20),
                "error_at_80": float(np.interp(0.80, curve["coverage"], curve["risk"])),
            }
        )
    return pd.DataFrame(rows)


def baseline_scores(logits: np.ndarray) -> dict[str, np.ndarray]:
    p = expit(logits)
    confident = np.maximum(p, 1 - p)
    return {
        "entropy": -binary_entropy(p).mean(axis=1),
        "max_prob": confident.mean(axis=1),
        "all_correct": np.prod(confident, axis=1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="result/runs/seed_42/config.yaml")
    parser.add_argument("--oof-dir", default="result/analysis/oof")
    parser.add_argument("--output-dir", default="result/analysis/selective")
    parser.add_argument("--kind", default="gbdt", choices=["gbdt", "logreg"])
    parser.add_argument("--target-coverage", type=float, default=0.80)
    parser.add_argument("--test", action="store_true", help="通过 fold 9 判定后再跑 fold 10")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = load_frame(args.config)

    oof = np.load(Path(args.oof_dir) / "oof_features.npz")
    train_frame = aligned_metadata(frame, oof["ecg_id"])
    X_train = record_features(oof["features"].astype(float), oof["logits"].astype(float), train_frame)
    y_train = oof["labels"].astype(float)
    p_train = expit(oof["logits"].astype(float))
    target_train = np.all((p_train > 0.5) == (y_train > 0.5), axis=1).astype(int)
    groups = oof["patient_id"]

    head = CorrectnessHead(kind=args.kind, seed=42)
    oof_score = head.fit_oof(X_train, target_train, groups, n_splits=5, seed=42)
    print(f"[train] n={len(y_train)} positive_rate={target_train.mean():.3f} oof_auc={head.oof_auc:.4f}", flush=True)
    head.fit_full(X_train, target_train)

    # label-level head：label_features 为 label-major 顺序（k * n + i），故需转置后展平
    Xl_train, _ = label_features(oof["features"].astype(float), oof["logits"].astype(float), 5)
    yl_train = y_train.T.reshape(-1)
    pl_train = p_train.T.reshape(-1)
    target_label = (yl_train == (pl_train > 0.5)).astype(int)
    group_label = np.tile(groups, 5)
    label_head = CorrectnessHead(kind=args.kind, seed=42)
    label_oof = label_head.fit_oof(Xl_train, target_label, group_label, n_splits=5, seed=42)
    print(f"[label] n={len(target_label)} positive_rate={target_label.mean():.3f} oof_auc={label_head.oof_auc:.4f}", flush=True)
    label_head.fit_full(Xl_train, target_label)

    split_name = "test" if args.test else "validation"
    payload = np.load(Path(args.oof_dir) / f"{split_name}_features.npz")
    split_frame = aligned_metadata(frame, payload["ecg_id"])
    y = payload["labels"].astype(float)
    logits = payload["logits"].astype(float)
    p = expit(logits)
    X = record_features(payload["features"].astype(float), logits, split_frame)

    scores = baseline_scores(logits)
    scores["correctness_head"] = head.score(X)
    table = evaluate_scores(y, p, scores)
    table["split"] = split_name
    table["n"] = len(y)
    table.to_csv(out / f"{split_name}_score_comparison.csv", index=False)

    # 固定 80% coverage 的阈值法对照
    threshold_rows = []
    for name, score in scores.items():
        tau = lowest_score_threshold(score, args.target_coverage)
        mask = score >= tau
        errors = np.any((p > 0.5) != (y > 0.5), axis=1)
        threshold_rows.append(
            {
                "score": name,
                "threshold": tau,
                "coverage": float(mask.mean()),
                "accepted_any_error": float(errors[mask].mean()),
            }
        )
    threshold_table = pd.DataFrame(threshold_rows)
    threshold_table["split"] = split_name
    threshold_table.to_csv(out / f"{split_name}_threshold_comparison.csv", index=False)

    # label-level 对照：阈值在 fold 9 选、fold 10 复用
    Xl = label_features(payload["features"].astype(float), logits, 5)[0]
    label_score = label_head.score(Xl).reshape(5, len(y)).T
    np.savez_compressed(out / f"{split_name}_label_scores.npz", label_score=label_score)

    label_rows = []
    for coverage_target in (0.70, 0.76, 0.80):
        tau = float(np.quantile(label_score, 1 - coverage_target, method="lower"))
        accepted = label_score >= tau
        correct = (y == (p > 0.5))
        label_rows.append(
            {
                "score": "correctness_head_label",
                "threshold": tau,
                "label_coverage": float(accepted.mean()),
                "label_error": float((~correct)[accepted].mean()),
                "full_accept_rate": float(accepted.all(axis=1).mean()),
                "deferred_per_record": float((~accepted).sum(axis=1).mean()),
            }
        )
    label_table = pd.DataFrame(label_rows)
    label_table["split"] = split_name
    label_table.to_csv(out / f"{split_name}_label_head.csv", index=False)

    verdict = {}
    base = table.set_index("score")
    verdict["delta_aurc"] = float(base.loc["correctness_head", "aurc"] - base.loc["max_prob", "aurc"])
    verdict["delta_cov20"] = float(base.loc["correctness_head", "coverage_at_20"] - base.loc["max_prob", "coverage_at_20"])
    verdict["delta_err80"] = float(base.loc["correctness_head", "error_at_80"] - base.loc["max_prob", "error_at_80"])
    verdict["positive"] = bool(verdict["delta_aurc"] <= -0.002 and verdict["delta_cov20"] >= 0.01)
    verdict["oof_auc_record"] = head.oof_auc
    verdict["oof_auc_label"] = label_head.oof_auc

    (out / f"{split_name}_verdict.json").write_text(
        json.dumps(jsonable(verdict), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(table.round(4).to_string(index=False))
    print(threshold_table.round(4).to_string(index=False))
    print(label_table.round(4).to_string(index=False))
    print("verdict", json.dumps(jsonable(verdict), ensure_ascii=False))


if __name__ == "__main__":
    main()
