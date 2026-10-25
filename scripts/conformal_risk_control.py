"""E6：一致性风险控制（Conformal Risk Control）与经验阈值的对照。

CRC（Angelopoulos et al., ICLR 2023）控制的不是「接受集里的错误率」，而是
**边际风险** E[L]，其中 L 为被接受且判错的记录数（记录级）或错收标签数（标签级）。
两者不同：边际风险可以对覆盖率做交换，条件风险不能。本脚本把两种口径并排放，
并检验同分布假设在设备亚组上是否成立。

用法：
    python code/scripts/conformal_risk_control.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptbxl.inference import load_run  # noqa: E402

ALPHAS = (0.10, 0.15, 0.20, 0.25, 0.30)


def max_prob(p: np.ndarray) -> np.ndarray:
    return np.maximum(p, 1 - p).mean(1)


def crc_threshold(scores: np.ndarray, losses: np.ndarray, alpha: float, bound: float = 1.0) -> float:
    """返回满足 CRC 上界的最小阈值（即最大覆盖）。

    R̂(λ)·n/(n+1) + bound/(n+1) ≤ α，R̂ 为 λ 以上的平均损失。
    """
    n = len(scores)
    order = np.argsort(-scores, kind="stable")
    s_sorted = scores[order]
    l_sorted = losses[order]
    cum = np.cumsum(l_sorted) / np.arange(1, n + 1)
    bound_seq = cum * n / (n + 1) + bound / (n + 1)
    ok = bound_seq <= alpha
    if not ok.any():
        return float("inf")
    k = int(np.max(np.where(ok)[0]))          # 最大的可行前缀
    return float(s_sorted[k])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--run", default="result/runs/seed_42")
    ap.add_argument("--output-dir", default="result/analysis")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cal = json.loads((Path(args.run) / "calibration.json").read_text(encoding="utf-8"))
    T = float(cal["temperature"])

    val = np.load(Path(args.run) / "val_logits.npz")
    val_p = expit(val["logits"].astype(float) / T)
    val_y = val["labels"].astype(float)

    test = load_run(Path(args.run) / "config.yaml", "test", device=args.device)
    test_p = expit(test["logits"].astype(float) / T)
    test_y = test["labels"].astype(float)
    frame = test["split_frame"]

    # ---- 记录级 ----
    v_score, t_score = max_prob(val_p), max_prob(test_p)
    v_wrong = ((val_p > 0.5) != (val_y > 0.5)).any(axis=1).astype(float)
    t_wrong = ((test_p > 0.5) != (test_y > 0.5)).any(axis=1).astype(float)

    rows = []
    for a in ALPHAS:
        lam = crc_threshold(v_score, v_wrong, a, bound=1.0)
        acc = t_score >= lam
        cov = float(acc.mean())
        cond = float(t_wrong[acc].mean()) if acc.any() else float("nan")
        marg = float(t_wrong[acc].mean() * cov) if acc.any() else float("nan")
        rows.append({"alpha": a, "lambda": lam, "coverage": cov,
                     "accepted_error": cond, "marginal_risk": marg,
                     "guarantee_holds": bool(marg <= a)})
    rec = pd.DataFrame(rows)
    print("== 记录级 CRC ==")
    print(rec.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- 标签级：控制每记录错收标签数的期望 ----
    B = val_y.shape[1]                     # 损失上界 = 每条记录的标签数
    val_conf = np.maximum(val_p, 1 - val_p)
    test_conf = np.maximum(test_p, 1 - test_p)
    label_rows = []
    for a in ALPHAS:
        lam_v = crc_per_label(val_p, val_conf, val_y, a, B)
        acc = test_conf >= lam_v           # 逐标签接受（形状 n×5）
        pred = test_p > 0.5
        wrong_lab = ((pred != (test_y > 0.5)) & acc)
        n_acc = acc.sum(axis=1)
        full = float((n_acc == B).mean())
        lab_cov = float(acc.mean())
        lab_err = float(wrong_lab.sum() / max(1, acc.sum()))
        marg = float(wrong_lab.sum() / len(test_y))
        label_rows.append({"alpha": a, "label_coverage": lab_cov, "label_error": lab_err,
                           "full_record_issuance": full,
                           "deferred_labels_per_record": float(B - n_acc.mean()),
                           "marginal_risk": marg, "guarantee_holds": bool(marg <= a)})
    lab = pd.DataFrame(label_rows)
    print("\n== 标签级 CRC ==")
    print(lab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- 交换性检验：按设备亚组 ----
    shift_rows = []
    if "device" in frame.columns:
        dev = frame["device"].fillna("unknown").astype(str).to_numpy()
        major = pd.Series(dev).value_counts().index[0]
        for name, mask in [(f"device={major}", dev == major), (f"device!={major}", dev != major)]:
            if mask.sum() < 20:
                continue
            lam = crc_threshold(v_score, v_wrong, 0.20, bound=1.0)
            acc = t_score[mask] >= lam
            if not acc.any():
                continue
            marg = float(t_wrong[mask][acc].mean() * acc.mean())
            shift_rows.append({"group": name, "n": int(mask.sum()), "coverage": float(acc.mean()),
                               "accepted_error": float(t_wrong[mask][acc].mean()),
                               "marginal_risk": marg, "guarantee_holds": bool(marg <= 0.20)})
    shift = pd.DataFrame(shift_rows)
    print("\n== alpha=0.20 下的分组检验 ==")
    print(shift.to_string(index=False, float_format=lambda v: f"{v:.4f}") if len(shift) else "（无 device 列）")

    rec.to_csv(out / "crc_record_level.csv", index=False)
    lab.to_csv(out / "crc_label_level.csv", index=False)
    if len(shift):
        shift.to_csv(out / "crc_shift_check.csv", index=False)
    summary = {
        "temperature": T,
        "n_calibration": int(len(val_y)),
        "n_test": int(len(test_y)),
        "record_level": rec.to_dict(orient="records"),
        "label_level": lab.to_dict(orient="records"),
        "shift_check": shift.to_dict(orient="records"),
    }
    (out / "crc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n写出:", out / "crc_summary.json")


def crc_per_label(p: np.ndarray, conf: np.ndarray, y: np.ndarray, alpha: float, bound: float) -> float:
    """单一共享阈值：损失 = 每记录被接受且判错的标签数，上界为标签数。

    按置信度降序排所有 (记录, 标签) 槽位，前缀即"阈值以上的接受集"，
    因此可一次排序求出最大可行前缀。
    """
    n = y.shape[0]
    wrong = ((p > 0.5) != (y > 0.5))
    order = np.argsort(-conf, axis=None, kind="stable")
    flat_wrong = wrong.ravel()[order]
    cum = np.cumsum(flat_wrong) / n              # 每记录平均错收标签数
    bound_seq = cum * n / (n + 1) + bound / (n + 1)
    ok = np.where(bound_seq <= alpha)[0]
    if len(ok) == 0:
        return float("inf")
    k = int(ok.max())
    return float(conf.ravel()[order][k])


if __name__ == "__main__":
    main()
