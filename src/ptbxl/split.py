"""患者级 train/validation/test 划分。"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def patient_fold_split(
    table: pd.DataFrame,
    folds: Mapping[str, Sequence[int]],
    patient_col: str = "patient_id",
    fold_col: str = "strat_fold",
) -> dict[str, pd.DataFrame]:
    """优先使用 PTB-XL 官方 strat_fold，并断言患者不跨 split。"""
    if patient_col not in table or fold_col not in table:
        raise ValueError(f"需要列 {patient_col!r} 和 {fold_col!r}")
    assigned: dict[str, pd.DataFrame] = {}
    seen: dict[object, str] = {}
    for name, fold_values in folds.items():
        frame = table[table[fold_col].isin(list(fold_values))].copy()
        for patient in frame[patient_col].dropna().unique():
            previous = seen.get(patient)
            if previous is not None and previous != name:
                raise ValueError(f"患者 {patient} 同时出现在 {previous} 和 {name}")
            seen[patient] = name
        assigned[name] = frame
    return assigned


def deterministic_patient_split(
    table: pd.DataFrame,
    fractions: Sequence[float] = (0.8, 0.1, 0.1),
    patient_col: str = "patient_id",
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """没有官方 fold 时按患者随机分割；fractions 对应 train/validation/test。"""
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions 必须是和为 1 的三个比例")
    patients = table[patient_col].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    cut1 = int(len(patients) * fractions[0])
    cut2 = cut1 + int(len(patients) * fractions[1])
    groups = {"train": patients[:cut1], "validation": patients[cut1:cut2], "test": patients[cut2:]}
    return {name: table[table[patient_col].isin(ids)].copy() for name, ids in groups.items()}
