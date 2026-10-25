"""A（正式版）：以主模型自身特征训练 correctness head，并与 post-hoc 分数对比。

与 correctness_head.py 的区别：
- correctness_head.py 用 OOF 模型特征训练（严格 cross-fitting），但特征分布与部署模型不一致
- 本脚本用**主模型**对 fold 1-8 的特征训练，学的是"这个模型在哪些样本上会错"，
  与部署场景一致。fold 9 只用于选阈值，fold 10 只评估一次。

两套结果都要报告：cross-fitting 版本是"无偏但失配"，in-sample 版本是"匹配但可能过拟合"。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ptbxl.cache import cache_rows_for, load_cache
from ptbxl.cohort import derive_cohort_columns
from ptbxl.features import extract_penultimate
from ptbxl.labels import load_and_label
from ptbxl.metrics import aurc, coverage_at_error
from ptbxl.model import build_model
from ptbxl.selective import (
    CorrectnessHead,
    binary_entropy,
    label_features,
    lowest_score_threshold,
    record_features,
    risk_coverage_from_score,
)
from ptbxl.split import patient_fold_split
from ptbxl.train import make_loader
from ptbxl.utils import jsonable


def build_frame(cfg: dict) -> pd.DataFrame:
    root = Path(cfg["data"]["root"])
    frame = load_and_label(
        root / cfg["data"]["metadata"], root / cfg["data"]["scp_statements"],
        cfg["data"]["classes"],
        min_scp_probability=cfg.get("labeling", {}).get("min_scp_probability"),
        drop_all_zero=bool(cfg.get("labeling", {}).get("drop_all_zero", False)),
    )
    return derive_cohort_columns(frame)


def extract_split(model, split_frame, cfg, cache, index, device) -> dict:
    loader = make_loader(
        split_frame, Path(cfg["data"]["root"]), cfg["data"]["classes"],
        cfg["data"]["batch_size"], cfg["data"]["target_length"], cfg["data"]["sampling_rate"],
        False, cfg["data"].get("num_workers", 0),
        cache=cache, cache_rows=cache_rows_for(split_frame, index),
    )
    payload = extract_penultimate(model, loader, device)
    payload["frame"] = split_frame.reset_index(drop=True)
    return payload


def score_table(y: np.ndarray, p: np.ndarray, scores: dict, split: str) -> pd.DataFrame:
    rows = []
    for name, score in scores.items():
        curve = risk_coverage_from_score(y, p, score)
        rows.append(
            {
                "split": split,
                "score": name,
                "aurc": float(aurc(curve["risk"], curve["coverage"])),
                "coverage_at_15": float(coverage_at_error(curve["risk"], curve["coverage"], 0.15)),
                "coverage_at_20": float(coverage_at_error(curve["risk"], curve["coverage"], 0.20)),
                "error_at_80": float(np.interp(0.80, curve["coverage"], curve["risk"])),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="result/runs/seed_42/config.yaml")
    parser.add_argument("--output-dir", default="result/analysis/selective")
    parser.add_argument("--device", default=None)
    parser.add_argument("--reuse-features", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    feature_path = out / "main_model_features.npz"

    frame = build_frame(cfg)
    splits = patient_fold_split(frame, cfg["data"]["folds"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cache = load_cache(cfg["data"]["cache"])
    index = pd.read_csv(cfg["data"]["cache_index"])

    if args.reuse_features and feature_path.exists():
        stored = np.load(feature_path)
        payloads = {k: {"features": stored[f"{k}_features"], "logits": stored[f"{k}_logits"],
                        "labels": stored[f"{k}_labels"], "frame": splits[k].reset_index(drop=True)}
                    for k in ("train", "validation", "test")}
    else:
        model = build_model(cfg["model"], num_classes=len(cfg["data"]["classes"])).to(device)
        state = torch.load(cfg["training"]["checkpoint"], map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        payloads = {k: extract_split(model, splits[k], cfg, cache, index, device)
                    for k in ("train", "validation", "test")}
        np.savez_compressed(
            feature_path,
            **{f"{k}_{f}": payloads[k][f] for k in payloads for f in ("features", "logits", "labels")},
        )

    def design(payload):
        return record_features(payload["features"].astype(float), payload["logits"].astype(float), payload["frame"])

    train = payloads["train"]
    X_train = design(train)
    y_train = train["labels"].astype(float)
    p_train = expit(train["logits"].astype(float))
    target = np.all((p_train > 0.5) == (y_train > 0.5), axis=1).astype(int)
    head = CorrectnessHead(kind="gbdt", seed=42).fit_full(X_train, target)

    # label head
    Xl_train, _ = label_features(train["features"].astype(float), train["logits"].astype(float), 5)
    yl = y_train.T.reshape(-1)
    pl = p_train.T.reshape(-1)
    label_head = CorrectnessHead(kind="gbdt", seed=42).fit_full(Xl_train, (yl == (pl > 0.5)).astype(int))

    results, threshold_rows, label_rows = [], [], []
    for split in ("validation", "test"):
        payload = payloads[split]
        y = payload["labels"].astype(float)
        logits = payload["logits"].astype(float)
        p = expit(logits)
        X = design(payload)
        confident = np.maximum(p, 1 - p)
        scores = {
            "entropy": -binary_entropy(p).mean(axis=1),
            "max_prob": confident.mean(axis=1),
            "all_correct": np.prod(confident, axis=1),
            "correctness_head": head.score(X),
        }
        results.append(score_table(y, p, scores, split))

        for name, score in scores.items():
            tau = lowest_score_threshold(score, 0.80)
            mask = score >= tau
            errors = np.any((p > 0.5) != (y > 0.5), axis=1)
            threshold_rows.append(
                {"split": split, "score": name, "threshold": tau,
                 "coverage": float(mask.mean()), "accepted_any_error": float(errors[mask].mean())}
            )

        Xl = label_features(payload["features"].astype(float), logits, 5)[0]
        label_score = label_head.score(Xl).reshape(5, len(y)).T
        np.savez_compressed(out / f"{split}_label_scores_insample.npz", label_score=label_score)
        correct = (y == (p > 0.5))
        for coverage_target in (0.70, 0.76, 0.80):
            tau = float(np.quantile(label_score, 1 - coverage_target, method="lower"))
            accepted = label_score >= tau
            label_rows.append(
                {"split": split, "target_coverage": coverage_target, "threshold": tau,
                 "label_coverage": float(accepted.mean()),
                 "label_error": float((~correct)[accepted].mean()),
                 "full_accept_rate": float(accepted.all(axis=1).mean()),
                 "deferred_per_record": float((~accepted).sum(axis=1).mean())}
            )

    comparison = pd.concat(results, ignore_index=True)
    comparison.to_csv(out / "insample_comparison.csv", index=False)
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(out / "insample_thresholds.csv", index=False)
    labels = pd.DataFrame(label_rows)
    labels.to_csv(out / "insample_label_head.csv", index=False)

    base = comparison[comparison.split == "test"].set_index("score")
    self_curve = risk_coverage_from_score(y_train, p_train, head.score(X_train))
    verdict = {
        "delta_aurc_vs_maxprob": float(base.loc["correctness_head", "aurc"] - base.loc["max_prob", "aurc"]),
        "delta_aurc_vs_allcorrect": float(base.loc["correctness_head", "aurc"] - base.loc["all_correct", "aurc"]),
        "delta_cov20_vs_maxprob": float(base.loc["correctness_head", "coverage_at_20"] - base.loc["max_prob", "coverage_at_20"]),
        "delta_err80_vs_maxprob": float(base.loc["correctness_head", "error_at_80"] - base.loc["max_prob", "error_at_80"]),
        "train_self_aurc": float(aurc(self_curve["risk"], self_curve["coverage"])),
    }
    (out / "insample_verdict.json").write_text(json.dumps(jsonable(verdict), ensure_ascii=False, indent=2), encoding="utf-8")

    print(comparison.round(4).to_string(index=False))
    print(thresholds.round(4).to_string(index=False))
    print(labels.round(4).to_string(index=False))
    print("verdict", json.dumps(jsonable(verdict), ensure_ascii=False))


if __name__ == "__main__":
    main()
