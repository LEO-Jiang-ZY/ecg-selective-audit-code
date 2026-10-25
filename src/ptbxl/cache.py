"""100 Hz 波形的 memmap 缓存，用于加速多 epoch 训练与重复评估。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .io import pad_or_crop, read_ecg_100hz, resolve_record_path


def build_cache(
    frame: pd.DataFrame,
    root: str | Path,
    output: str | Path,
    index_output: str | Path,
    target_length: int = 1000,
    sampling_rate: int = 100,
    n_channels: int = 12,
    dtype: str = "float32",
    log_every: int = 500,
) -> dict:
    """把 frame 中每条记录写成 (n, channels, length) 的 memmap。

    行顺序与 frame 一致，index_output 保存 ecg_id -> row 的映射。
    """
    root = Path(root)
    output = Path(output)
    index_output = Path(index_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    index_output.parent.mkdir(parents=True, exist_ok=True)

    n = len(frame)
    cache = np.lib.format.open_memmap(output, mode="w+", dtype=dtype, shape=(n, n_channels, target_length))
    rows = []
    started = time.perf_counter()
    for i, (ecg_id, row) in enumerate(frame.iterrows()):
        signal, _ = read_ecg_100hz(
            resolve_record_path(row, root, sampling_rate),
            target_hz=sampling_rate,
        )
        signal = pad_or_crop(signal, target_length)
        if signal.shape != (n_channels, target_length):
            raise ValueError(f"ecg_id={ecg_id} 形状 {signal.shape}，期望 {(n_channels, target_length)}")
        cache[i] = signal.astype(dtype, copy=False)
        rows.append(
            {
                "ecg_id": int(ecg_id),
                "row": i,
                "patient_id": row.get("patient_id"),
                "strat_fold": row.get("strat_fold"),
            }
        )
        if log_every and (i + 1) % log_every == 0:
            elapsed = time.perf_counter() - started
            print(f"cache {i + 1}/{n}  {elapsed:.1f}s  {elapsed / (i + 1) * 1000:.1f} ms/record", flush=True)

    cache.flush()
    del cache
    pd.DataFrame(rows).to_csv(index_output, index=False)
    meta = {
        "output": str(output),
        "index": str(index_output),
        "n_records": int(n),
        "shape": [int(n), int(n_channels), int(target_length)],
        "dtype": dtype,
        "sampling_rate": int(sampling_rate),
        "source_root": str(root),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    output.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_cache(path: str | Path, mmap_mode: str = "r") -> np.ndarray:
    return np.load(Path(path), mmap_mode=mmap_mode)


def cache_rows_for(frame: pd.DataFrame, index: pd.DataFrame) -> np.ndarray:
    """按 frame 的 ecg_id 顺序生成 cache 行号。"""
    mapping = dict(zip(index["ecg_id"].astype(int), index["row"].astype(int)))
    missing = [int(ecg_id) for ecg_id in frame.index if int(ecg_id) not in mapping]
    if missing:
        raise KeyError(f"{len(missing)} 条记录不在缓存索引中，例如 {missing[:5]}")
    return np.asarray([mapping[int(ecg_id)] for ecg_id in frame.index], dtype=int)
