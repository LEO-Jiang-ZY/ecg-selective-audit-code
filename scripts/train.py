"""训练 PTB-XL 轻量 1D ResNet，并导出验证集 logits。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptbxl.cache import cache_rows_for, load_cache
from ptbxl.labels import load_and_label
from ptbxl.model import build_model
from ptbxl.split import patient_fold_split
from ptbxl.train import export_logits, fit_model, make_loader, set_seed
from ptbxl.utils import jsonable


def _limit(frame, limit: int | None):
    return frame.head(limit) if limit else frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="code/configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的 epochs，用于冒烟测试")
    parser.add_argument("--learning-rate", type=float, default=None, help="覆盖配置中的 learning_rate")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-validation", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--logits", default=None, help="验证集 logits 输出路径；默认取配置或 checkpoint 同目录")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["training"]
    labeling = cfg.get("labeling", {})
    set_seed(int(train_cfg["seed"]))
    root = Path(data_cfg["root"])
    frame = load_and_label(
        root / data_cfg["metadata"],
        root / data_cfg["scp_statements"],
        data_cfg["classes"],
        min_scp_probability=labeling.get("min_scp_probability"),
        drop_all_zero=bool(labeling.get("drop_all_zero", False)),
    )
    splits = patient_fold_split(frame, data_cfg["folds"])
    train_frame = _limit(splits["train"], args.limit_train)
    validation_frame = _limit(splits["validation"], args.limit_validation)
    if train_frame.empty or validation_frame.empty:
        raise ValueError("训练集或验证集为空，请检查 fold 配置与 limit 参数")

    cache = None
    train_cache_rows = None
    validation_cache_rows = None
    cache_path = data_cfg.get("cache")
    cache_index_path = data_cfg.get("cache_index")
    if cache_path and cache_index_path and Path(cache_path).exists() and Path(cache_index_path).exists():
        cache = load_cache(cache_path)
        cache_index = pd.read_csv(cache_index_path)
        train_cache_rows = cache_rows_for(train_frame, cache_index)
        validation_cache_rows = cache_rows_for(validation_frame, cache_index)
        print(f"使用波形缓存 {cache_path}（{len(cache)} 条）", flush=True)
    else:
        print("未找到波形缓存，回退到 WFDB 逐条读取", flush=True)

    train_loader = make_loader(train_frame, root, data_cfg["classes"], data_cfg["batch_size"], data_cfg["target_length"], data_cfg["sampling_rate"], True, data_cfg.get("num_workers", 0), cache=cache, cache_rows=train_cache_rows)
    validation_loader = make_loader(validation_frame, root, data_cfg["classes"], data_cfg["batch_size"], data_cfg["target_length"], data_cfg["sampling_rate"], False, data_cfg.get("num_workers", 0), cache=cache, cache_rows=validation_cache_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg, num_classes=len(data_cfg["classes"])).to(device)

    epochs = int(args.epochs if args.epochs is not None else train_cfg["epochs"])
    if epochs < 1:
        raise ValueError("epochs 必须 >= 1")
    checkpoint = args.checkpoint or train_cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError("必须提供 training.checkpoint 或 --checkpoint")
    history = fit_model(
        train_loader,
        validation_loader,
        model,
        epochs,
        float(args.learning_rate if args.learning_rate is not None else train_cfg["learning_rate"]),
        float(train_cfg["weight_decay"]),
        device,
        checkpoint,
    )

    output_dir = Path(checkpoint).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    logits_info = None
    if train_cfg.get("save_validation_logits", True):
        if args.logits:
            logits_path = args.logits
        elif args.checkpoint:
            logits_path = str(output_dir / "val_logits.npz")
        else:
            logits_path = train_cfg.get("validation_logits", str(output_dir / "val_logits.npz"))
        logits_info = export_logits(model, validation_loader, validation_frame, logits_path, device, split_name="validation")

    meta = {
        "seed": int(train_cfg["seed"]),
        "epochs": epochs,
        "device": str(device),
        "n_train": int(len(train_frame)),
        "n_validation": int(len(validation_frame)),
        "checkpoint": str(checkpoint),
        "best_epoch": int(state.get("epoch", -1)),
        "best_validation_loss": float(state.get("validation_loss", float("nan"))),
        "validation_logits": logits_info,
        "cache": str(cache_path) if cache is not None else None,
        "config": cfg,
    }
    (output_dir / "run_meta.json").write_text(json.dumps(jsonable(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(jsonable({"history_last": history[-1], "run_meta": meta}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
