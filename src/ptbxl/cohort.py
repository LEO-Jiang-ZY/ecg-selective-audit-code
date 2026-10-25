"""队列派生列：年龄、性别、原生信号质量、设备与站点分组。"""
from __future__ import annotations

import numpy as np
import pandas as pd

QUALITY_COLUMNS = ("baseline_drift", "static_noise", "burst_noise", "electrodes_problems")
AGE_LABELS = ("<18", "18-44", "45-64", "65-89", ">89", "unknown")


def _present(series: pd.Series) -> pd.Series:
    """非空且不是占位字符串，视为该标注存在。"""
    text = series.astype("string").str.strip()
    return text.notna() & ~text.str.lower().isin(["", "nan", "none", "null"])


def _quality_type(flags: pd.DataFrame) -> pd.Series:
    def label(row) -> str:
        names = [column for column in flags.columns if bool(row[column])]
        if not names:
            return "none"
        if len(names) == 1:
            return names[0]
        return "multiple"

    return flags.apply(label, axis=1)


def derive_cohort_columns(frame: pd.DataFrame, device_top_n: int = 6, site_top_n: int = 4) -> pd.DataFrame:
    """返回附加派生列的副本，原始列保持不变。

    派生列：
      age_group   : <18 / 18-44 / 45-64 / 65-89 / >89 / unknown（age=300 归入 >89）
      sex_label   : male / female / unknown
      quality_any : noise / clean
      quality_type: 单一噪声类型 / multiple / none
      device_group: 高频设备名 / other / unknown
      site_group  : 高频站点 / site_other / unknown
    """
    out = frame.copy()

    age = pd.to_numeric(out["age"], errors="coerce") if "age" in out else pd.Series(np.nan, index=out.index)
    age = age.where(age <= 89, 90)
    age_group = pd.cut(
        age,
        bins=[-np.inf, 18, 45, 65, 90, np.inf],
        right=False,
        labels=["<18", "18-44", "45-64", "65-89", ">89"],
    )
    out["age_group"] = age_group.cat.add_categories(["unknown"]).fillna("unknown")

    sex = pd.to_numeric(out["sex"], errors="coerce") if "sex" in out else pd.Series(np.nan, index=out.index)
    out["sex_label"] = sex.map({0: "male", 1: "female"}).fillna("unknown").astype(str)

    available = [column for column in QUALITY_COLUMNS if column in out.columns]
    if available:
        flags = pd.DataFrame({column: _present(out[column]) for column in available}, index=out.index)
        out["quality_any"] = np.where(flags.any(axis=1), "noise", "clean")
        out["quality_type"] = _quality_type(flags).astype(str)
    else:
        out["quality_any"] = "unknown"
        out["quality_type"] = "unknown"

    if "device" in out.columns:
        device = out["device"].map(lambda value: "unknown" if pd.isna(value) else " ".join(str(value).split()))
        top_devices = device.value_counts().head(device_top_n).index
        out["device_group"] = device.where(device.isin(top_devices), "other").astype(str)
    else:
        out["device_group"] = "unknown"

    if "site" in out.columns:
        site = pd.to_numeric(out["site"], errors="coerce")
        top_sites = set(site.value_counts().head(site_top_n).index)
        out["site_group"] = site.map(
            lambda value: "unknown"
            if pd.isna(value)
            else (f"site_{int(value)}" if value in top_sites else "site_other")
        ).astype(str)
    else:
        out["site_group"] = "unknown"

    return out
