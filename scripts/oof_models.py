"""交叉拟合主模型，产出 fold 1-8 的折外特征，供 correctness head 训练。

动机：主模型在 fold 1-8 上是样本内的，logits 过拟合，直接在这些 logits 上训练
correctness head 会学到"高置信=正确"的假映射。这里把 fold 1-8 按患者分 5 折，
每折训练一个独立模型，只用来预测被留出折，得到近似无泄漏的特征。

每折固定训练 14 个 epoch（与主模型的最佳 epoch 一致），不使用验证集选权重，
避免任何按折外表现挑 epoch 的泄漏。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptbxl.cache import cache_rows_for, load_cache
from ptbxl.cohort import derive_cohort_columns
from ptbxl.features import extract_penultimate
from ptbxl.labels import load_and_label
from ptbxl.model import build_model
from ptbxl.selective import grouped_folds
from ptbxl.split import patient_fold_split
from ptbxl.train import fit_model, make_loader, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="result/runs/seed_42/config.yaml")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="result/analysis/oof")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["training"]
    classes = data_cfg["classes"]
    labeling = cfg.get("labeling", {})
    root = Path(data_cfg["root"])

    frame = load_and_label(
        root / data_cfg["metadata"],
        root / data_cfg["scp_statements"],
        classes,
        min_scp_probability=labeling.get("min_scp_probability"),
        drop_all_zero=bool(labeling.get("drop_all_zero", False)),
    )
    frame = derive_cohort_columns(frame)
    splits = patient_fold_split(frame, data_cfg["folds"])
    train_frame = splits["train"]

    cache = load_cache(data_cfg["cache"]) if data_cfg.get("cache") else None
    cache_index = pd.read_csv(data_cfg["cache_index"]) if data_cfg.get("cache_index") else None
    if cache is None or cache_index is None:
        raise SystemExit("需要可用的波形缓存，先运行 build_cache.py")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    labels_truth = train_frame[classes].to_numpy(dtype=np.float32)
    folds = grouped_folds(
        train_frame["patient_id"].to_numpy(),
        labels_truth.mean(axis=1),
        n_splits=args.n_splits,
        seed=args.seed,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    oof_features, oof_logits, oof_labels, oof_ecg, oof_patient, oof_fold = [], [], [], [], [], []
    fold_meta = []

    for fold in range(args.n_splits):
        started = time.perf_counter()
        fit_frame = train_frame.iloc[folds != fold]
        hold_frame = train_frame.iloc[folds == fold]

        set_seed(args.seed + fold)
        model = build_model(model_cfg, num_classes=len(classes)).to(device)
        train_loader = make_loader(
            fit_frame, root, classes, data_cfg["batch_size"], data_cfg["target_length"],
            data_cfg["sampling_rate"], True, data_cfg.get("num_workers", 0),
            cache=cache, cache_rows=cache_rows_for(fit_frame, cache_index),
        )
        monitor = make_loader(
            hold_frame.head(256), root, classes, data_cfg["batch_size"], data_cfg["target_length"],
            data_cfg["sampling_rate"], False, data_cfg.get("num_workers", 0),
            cache=cache, cache_rows=cache_rows_for(hold_frame.head(256), cache_index),
        )
        history = fit_model(
            train_loader, monitor, model, args.epochs,
            train_cfg["learning_rate"], train_cfg["weight_decay"], str(device),
            None,
        )

        eval_loader = make_loader(
            hold_frame, root, classes, data_cfg["batch_size"], data_cfg["target_length"],
            data_cfg["sampling_rate"], False, data_cfg.get("num_workers", 0),
            cache=cache, cache_rows=cache_rows_for(hold_frame, cache_index),
        )
        payload = extract_penultimate(model, eval_loader, device)
        oof_features.append(payload["features"])
        oof_logits.append(payload["logits"])
        oof_labels.append(payload["labels"])
        oof_ecg.append(hold_frame.index.to_numpy())
        oof_patient.append(hold_frame["patient_id"].to_numpy())
        oof_fold.append(np.full(len(hold_frame), fold, dtype=np.int64))
        fold_meta.append(
            {
                "fold": fold,
                "n_fit": int(len(fit_frame)),
                "n_hold": int(len(hold_frame)),
                "final_train_loss": float(history[-1]["train_loss"]),
                "seconds": round(time.perf_counter() - started, 1),
            }
        )
        print(f"[fold {fold + 1}/{args.n_splits}] n={len(hold_frame)} "
              f"val_loss={history[-1]['validation_loss']:.4f} "
              f"{fold_meta[-1]['seconds']:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()

    np.savez_compressed(
        out_dir / "oof_features.npz",
        features=np.concatenate(oof_features).astype(np.float32),
        logits=np.concatenate(oof_logits).astype(np.float32),
        labels=np.concatenate(oof_labels).astype(np.float32),
        ecg_id=np.concatenate(oof_ecg),
        patient_id=np.concatenate(oof_patient),
        fold=np.concatenate(oof_fold),
    )

    # fold 9 / fold 10 用主模型提取，作为部署时的特征分布
    main_model = build_model(model_cfg, num_classes=len(classes)).to(device)
    state = torch.load(train_cfg["checkpoint"], map_location=device, weights_only=True)
    main_model.load_state_dict(state["model"])
    for name in ("validation", "test"):
        split_frame = splits[name]
        loader = make_loader(
            split_frame, root, classes, data_cfg["batch_size"], data_cfg["target_length"],
            data_cfg["sampling_rate"], False, data_cfg.get("num_workers", 0),
            cache=cache, cache_rows=cache_rows_for(split_frame, cache_index),
        )
        payload = extract_penultimate(main_model, loader, device)
        np.savez_compressed(
            out_dir / f"{name}_features.npz",
            features=payload["features"].astype(np.float32),
            logits=payload["logits"].astype(np.float32),
            labels=payload["labels"].astype(np.float32),
            ecg_id=split_frame.index.to_numpy(),
            patient_id=split_frame["patient_id"].to_numpy(),
        )
        print(f"[{name}] n={len(split_frame)} feature_dim={payload['features'].shape[1]}", flush=True)

    (out_dir / "oof_meta.json").write_text(
        json.dumps({"folds": fold_meta, "n_splits": args.n_splits, "epochs": args.epochs,
                    "seed": args.seed, "feature_dim": int(oof_features[0].shape[1])},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("saved to", out_dir)


if __name__ == "__main__":
    main()
