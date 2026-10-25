"""提取 penultimate 特征与 logits，供 correctness head 与误差分解使用。

所有架构的 forward 末段都是 ``self.head(features)``，因此对 ``head`` 注册
forward pre-hook 即可拿到全局池化后的表征，无需改模型定义。
"""
from __future__ import annotations

import numpy as np
import torch


class HeadInputCapture:
    """捕获 ``model.head`` 的输入张量。"""

    def __init__(self, model: torch.nn.Module):
        if not hasattr(model, "head"):
            raise AttributeError("模型缺少 head 属性，无法捕获 penultimate 特征")
        self._buffer: torch.Tensor | None = None
        self._handle = model.head.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        self._buffer = inputs[0].detach()

    def pop(self) -> np.ndarray:
        if self._buffer is None:
            raise RuntimeError("hook 未触发，检查模型是否调用了 head")
        value = self._buffer.cpu().numpy().astype(np.float32)
        self._buffer = None
        return value

    def close(self) -> None:
        self._handle.remove()


@torch.no_grad()
def extract_penultimate(model: torch.nn.Module, loader, device) -> dict[str, np.ndarray]:
    """返回 backbone 特征、logits 与标签。"""
    model.eval()
    capture = HeadInputCapture(model)
    features, logits, labels = [], [], []
    try:
        for x, y in loader:
            output = model(x.to(device))
            features.append(capture.pop())
            logits.append(output.cpu().numpy().astype(np.float32))
            labels.append(y.numpy().astype(np.float32))
    finally:
        capture.close()
    return {
        "features": np.concatenate(features),
        "logits": np.concatenate(logits),
        "labels": np.concatenate(labels),
    }
