"""PTB-XL SCP statements 到 superclass 多标签的转换。"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

DEFAULT_CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def _is_true(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value) and not (isinstance(value, float) and pd.isna(value))


def parse_scp_codes(value: object) -> dict[str, float]:
    """解析 ptbxl_database.csv 的 scp_codes 字符串。"""
    if isinstance(value, Mapping):
        return {str(k): float(v) for k, v in value.items()}
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, dict):
        raise ValueError(f"scp_codes 必须是 dict，实际为 {type(parsed)}")
    return {str(k): float(v) for k, v in parsed.items()}


def load_statement_map(path: str | Path, classes: Sequence[str] = DEFAULT_CLASSES) -> dict[str, str]:
    """读取 SCP 表，将诊断代码映射到配置的 superclass。"""
    table = pd.read_csv(path)
    required = {"diagnostic", "diagnostic_class"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"scp_statements.csv 缺少列: {sorted(missing)}")
    class_set = set(classes)
    result: dict[str, str] = {}
    rows = table[table["diagnostic"].map(_is_true)]
    for _, row in rows.iterrows():
        label = row.get("diagnostic_class")
        if pd.notna(label) and str(label) in class_set:
            result[str(row["Unnamed: 0"] if "Unnamed: 0" in row else row.name)] = str(label)
    # PTB-XL 常将 SCP code 放在 index；兼容显式 statement/code 列。
    if "Unnamed: 0" in table.columns:
        return result
    index = pd.read_csv(path, index_col=0)
    for code, row in index.iterrows():
        label = row.get("diagnostic_class")
        if pd.notna(label) and str(label) in class_set and _is_true(row.get("diagnostic", False)):
            result[str(code)] = str(label)
    return result


def build_superclass_labels(
    database: pd.DataFrame,
    statements: pd.DataFrame,
    classes: Sequence[str] = DEFAULT_CLASSES,
    min_scp_probability: float | None = None,
    drop_all_zero: bool = False,
) -> pd.DataFrame:
    """为每条记录生成 superclass 多热标签列。

    drop_all_zero=False 时保留五个 superclass 均未命中的记录（all-zero 不等于 NORM）。
    """
    if "scp_codes" not in database:
        raise ValueError("database 必须包含 scp_codes 列")
    if statements.index.name is None and "Unnamed: 0" in statements.columns:
        statements = statements.set_index("Unnamed: 0")
    required = {"diagnostic", "diagnostic_class"}
    missing = required - set(statements.columns)
    if missing:
        raise ValueError(f"scp_statements 缺少列: {sorted(missing)}")
    class_set = tuple(classes)
    code_to_class = {
        str(code): str(row["diagnostic_class"])
        for code, row in statements.iterrows()
        if _is_true(row.get("diagnostic", False)) and pd.notna(row.get("diagnostic_class"))
        and str(row["diagnostic_class"]) in class_set
    }
    output = database.copy()
    values = []
    for raw in output["scp_codes"]:
        codes = parse_scp_codes(raw)
        labels = set()
        for code, score in codes.items():
            if min_scp_probability is None or score >= min_scp_probability:
                label = code_to_class.get(code)
                if label:
                    labels.add(label)
        values.append([int(label in labels) for label in class_set])
    label_frame = pd.DataFrame(values, index=output.index, columns=list(class_set), dtype="int8")
    result = pd.concat([output, label_frame], axis=1)
    if drop_all_zero:
        result = result[result[list(class_set)].sum(axis=1) > 0]
    return result


def load_and_label(
    database_path: str | Path,
    statements_path: str | Path,
    classes: Iterable[str] = DEFAULT_CLASSES,
    min_scp_probability: float | None = None,
    drop_all_zero: bool = False,
) -> pd.DataFrame:
    database = pd.read_csv(database_path, index_col=0)
    statements = pd.read_csv(statements_path, index_col=0)
    return build_superclass_labels(
        database,
        statements,
        tuple(classes),
        min_scp_probability=min_scp_probability,
        drop_all_zero=drop_all_zero,
    )
