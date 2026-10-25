"""通用小工具。"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def jsonable(value):
    """把 numpy / pandas 标量与容器转成标准 JSON 可序列化对象。"""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, Path):
        return str(value)
    return value
