"""PTB-XL WFDB 100 Hz 记录读取。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


def read_ecg_100hz(record: str | Path, root: str | Path | None = None, target_hz: int = 100, normalize: bool = True) -> tuple[np.ndarray, dict]:
    """读取 WFDB 记录，返回 shape=(channels, samples) 的 float32 数组。"""
    import wfdb

    path = Path(record)
    if root is not None:
        path = Path(root) / path
    signal, fields = wfdb.rdsamp(str(path))
    signal = np.asarray(signal, dtype=np.float32)
    source_hz = int(round(float(fields["fs"])))
    if source_hz != target_hz:
        from math import gcd
        g = gcd(source_hz, target_hz)
        signal = resample_poly(signal, target_hz // g, source_hz // g, axis=0).astype(np.float32)
    signal = signal.T
    if normalize:
        mean = np.nanmean(signal, axis=1, keepdims=True)
        std = np.nanstd(signal, axis=1, keepdims=True)
        signal = (np.nan_to_num(signal, nan=0.0) - mean) / np.maximum(std, np.finfo(np.float32).eps)
    fields = dict(fields)
    fields["fs_original"] = source_hz
    fields["fs"] = target_hz
    return signal, fields


def resolve_record_path(row, root: str | Path, sampling_rate: int = 100) -> Path:
    """根据 metadata 的 filename_lr/filename_hr 解析相对 WFDB 路径。"""
    column = "filename_lr" if sampling_rate == 100 else "filename_hr"
    if column not in row:
        raise KeyError(f"metadata 缺少 {column}")
    return Path(root) / str(row[column])


def pad_or_crop(signal: np.ndarray, length: int) -> np.ndarray:
    """将单条 ECG 调整到固定样本数。"""
    if signal.shape[-1] >= length:
        return signal[..., :length]
    return np.pad(signal, [(0, 0), (0, length - signal.shape[-1])], mode="constant")
