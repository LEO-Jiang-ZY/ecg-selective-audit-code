"""按配置加载某个 split 的模型输出，供多种后处理分析复用。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .cache import cache_rows_for, load_cache
from .cohort import derive_cohort_columns
from .labels import load_and_label
from .model import build_model
from .split import patient_fold_split
from .train import make_loader, predict


def load_run(config_path: str | Path, split_name: str = "test", limit: int | None = None, device: str | None = None, return_signals: bool = False) -> dict:
    """返回 labels、logits、split 表、完整 frame、模型与配置。"""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["training"]
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
    split_frame = patient_fold_split(frame, data_cfg["folds"])[split_name]
    if limit:
        split_frame = split_frame.head(limit)
    split_ids = split_frame.index.to_numpy()

    cache = None
    rows = None
    cache_path = data_cfg.get("cache")
    cache_index_path = data_cfg.get("cache_index")
    if cache_path and cache_index_path and Path(cache_path).exists() and Path(cache_index_path).exists():
        cache = load_cache(cache_path)
        rows = cache_rows_for(split_frame, pd.read_csv(cache_index_path))

    loader = make_loader(
        split_frame,
        root,
        data_cfg["classes"],
        data_cfg["batch_size"],
        data_cfg["target_length"],
        data_cfg["sampling_rate"],
        False,
        data_cfg.get("num_workers", 0),
        cache=cache,
        cache_rows=rows,
    )
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(model_cfg, num_classes=len(data_cfg["classes"])).to(device)
    checkpoint = Path(train_cfg["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    labels, logits = predict(model, loader, device)
    signals = None
    if return_signals:
        if cache is None or rows is None:
            raise ValueError("return_signals=True 需要可用的波形缓存")
        signals = np.asarray([cache[row] for row in rows], dtype=np.float32)
    return {
        "config": cfg,
        "frame": frame,
        "split_frame": split_frame.reset_index(drop=True),
        "split_ids": split_ids,
        "labels": labels,
        "logits": logits,
        "signals": signals,
        "cache_rows": rows,
        "model": model,
        "device": device,
        "checkpoint": checkpoint,
    }


def calibrated_logits(logits: np.ndarray, calibration_path: str | Path) -> tuple[np.ndarray, dict]:
    import json

    payload = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    temperature = float(payload.get("temperature", 1.0))
    return np.asarray(logits, dtype=float) / temperature, payload
