"""GradientSHAP 与分段 LIME 的轻量实现。"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import Ridge

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def segment_attribution(values: np.ndarray, n_windows: int = 10) -> np.ndarray:
    """把 (leads, samples) 的归因按时间窗求和，返回 (leads, n_windows)。"""
    leads, samples = values.shape
    width = samples // n_windows
    trimmed = values[:, : width * n_windows]
    return trimmed.reshape(leads, n_windows, width).sum(axis=2)


def make_gradient_explainer(model, background: torch.Tensor):
    import shap

    model.eval()
    return shap.GradientExplainer(model, background)


def lime_segments(
    model,
    signal: torch.Tensor,
    label: int,
    n_windows: int = 10,
    n_samples: int = 1000,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[np.ndarray, float]:
    """分段 LIME：12 导联 × n_windows 个二值段，置零等价于替换为导联均值。"""
    if signal.dim() != 3 or signal.shape[0] != 1:
        raise ValueError(f"signal 形状应为 (1, leads, samples)，实际 {tuple(signal.shape)}")
    leads, samples = signal.shape[1], signal.shape[2]
    width = samples // n_windows
    usable = width * n_windows
    n_segments = leads * n_windows
    rng = np.random.default_rng(seed)
    masks = (rng.random((n_samples, n_segments)) < 0.5).astype(np.float32)
    masks[0] = 1.0

    segments = torch.from_numpy(masks.reshape(n_samples, leads, n_windows)).unsqueeze(-1)
    segments = segments.expand(-1, -1, -1, width).reshape(n_samples, leads, usable).to(device)
    x = signal[:, :, :usable].to(device)

    predictions = []
    with torch.no_grad():
        for start in range(0, n_samples, 128):
            batch = segments[start : start + 128] * x
            predictions.append(torch.sigmoid(model(batch))[:, label].cpu().numpy())
    predictions = np.concatenate(predictions)

    distance = np.sum((masks - 1.0) ** 2, axis=1)
    weights = np.exp(-distance / (0.25 * n_segments))
    ridge = Ridge(alpha=1.0).fit(masks, predictions, sample_weight=weights)
    fitted = ridge.predict(masks)
    ss_res = float(np.sum(weights * (predictions - fitted) ** 2))
    ss_tot = float(np.sum(weights * (predictions - np.average(predictions, weights=weights)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return ridge.coef_.reshape(leads, n_windows), r2


def deletion_curve(
    model,
    signal: torch.Tensor,
    importance: np.ndarray,
    label: int,
    n_steps: int = 20,
    device: str = "cpu",
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """按段重要性从高到低置零，返回删除比例、目标概率与曲线 AUC。"""
    leads, n_windows = importance.shape
    flat = importance.reshape(-1).astype(float)
    if random_state is not None:
        order = np.arange(len(flat))
        np.random.default_rng(random_state).shuffle(order)
    else:
        order = np.argsort(flat)[::-1]
    width = signal.shape[-1] // n_windows
    usable = width * n_windows
    base = signal[:, :, :usable].to(device)
    probabilities = []
    for step in range(n_steps + 1):
        remove = int(round(step / n_steps * len(order)))
        keep = np.ones(len(order), dtype=np.float32)
        keep[order[:remove]] = 0.0
        mask = torch.from_numpy(keep.reshape(1, leads, n_windows, 1)).expand(-1, -1, -1, width).reshape(1, leads, usable).to(device)
        with torch.no_grad():
            probabilities.append(float(torch.sigmoid(model(base * mask))[0, label]))
    fractions = np.linspace(0.0, 1.0, n_steps + 1)
    return fractions, np.asarray(probabilities), float(np.trapz(probabilities, fractions))


def top_segments(importance: np.ndarray, k: int = 10) -> list[tuple[int, int, float]]:
    flat = importance.reshape(-1)
    order = np.argsort(np.abs(flat))[::-1][:k]
    n_windows = importance.shape[1]
    return [(int(index // n_windows), int(index % n_windows), float(flat[index])) for index in order]
