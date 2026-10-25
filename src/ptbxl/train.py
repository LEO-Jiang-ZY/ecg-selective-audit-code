"""PTB-XL 数据集、训练与预测循环。"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .io import pad_or_crop, read_ecg_100hz, resolve_record_path
from .model import ECGResNet1D


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PTBXLDataset(Dataset):
    def __init__(self, frame, root, classes, target_length: int = 1000, sampling_rate: int = 100, cache=None, cache_rows=None):
        self.frame = frame.reset_index(drop=True)
        self.root = Path(root)
        self.classes = list(classes)
        self.target_length = target_length
        self.sampling_rate = sampling_rate
        self.cache = cache
        self.cache_rows = None if cache_rows is None else np.asarray(cache_rows, dtype=int)
        if (cache is None) != (cache_rows is None):
            raise ValueError("cache 与 cache_rows 必须同时提供")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        if self.cache is not None:
            signal = np.array(self.cache[self.cache_rows[index]], dtype=np.float32, copy=True)
        else:
            signal, _ = read_ecg_100hz(resolve_record_path(row, self.root, self.sampling_rate), target_hz=self.sampling_rate)
            signal = pad_or_crop(signal, self.target_length)
        labels = row[self.classes].to_numpy(dtype=np.float32)
        return torch.from_numpy(signal), torch.from_numpy(labels)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * len(x)
        count += len(x)
    return total / max(count, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(logits)


@torch.no_grad()
def export_logits(model, loader, frame, path, device, split_name: str = "validation") -> dict:
    """导出某个 split 的 logits，供验证集温度拟合与阈值选择使用。

    frame 的行顺序必须与 loader 一致（shuffle=False），其 index 视为 ecg_id。
    """
    labels, logits = predict(model, loader, device)
    if len(frame) != len(labels):
        raise ValueError(f"frame 行数 {len(frame)} 与预测结果 {len(labels)} 不一致")
    payload = {
        "logits": logits.astype(np.float32),
        "labels": labels.astype(np.int8),
        "ecg_id": np.asarray(frame.index),
        "split": np.array(split_name),
    }
    if "patient_id" in frame:
        payload["patient_id"] = frame["patient_id"].to_numpy()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **payload)
    return {"path": str(target), "n": int(len(labels)), "num_classes": int(logits.shape[1])}


def fit_model(train_loader, validation_loader, model, epochs: int, learning_rate: float, weight_decay: float, device, checkpoint: str | Path | None = None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    best = float("inf")
    history = []
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        model.eval()
        val_loss, count = 0.0, 0
        with torch.no_grad():
            for x, y in validation_loader:
                loss = criterion(model(x.to(device)), y.to(device))
                val_loss += float(loss.item()) * len(x)
                count += len(x)
        val_loss /= max(count, 1)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss})
        print(f"epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}", flush=True)
        if checkpoint and val_loss < best:
            best = val_loss
            Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "validation_loss": val_loss,
                    "num_classes": int(model.head.out_features),
                },
                checkpoint,
            )
    return history


def make_loader(frame, root, classes, batch_size, target_length, sampling_rate, shuffle, num_workers=0, cache=None, cache_rows=None):
    return DataLoader(
        PTBXLDataset(frame, root, classes, target_length, sampling_rate, cache=cache, cache_rows=cache_rows),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
