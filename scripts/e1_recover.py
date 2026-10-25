"""E1：把 6.88pp 的 oracle 缺口摊开，报告每类打分器回收了多少。

recovered(θ) = (R_current − R_θ) / (R_current − R_oracle)

只使用已保存的中间产物（不重训、不推理）：
  result/analysis/selective/main_model_features.npz    seed 42 的 train/val/test 特征与 logits
  result/analysis/selective/test_label_scores_insample.npz  逐标签 head 的分数
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptbxl.metrics import error_vector  # noqa: E402


def scores_from_logits(logits: np.ndarray) -> dict[str, np.ndarray]:
    p = expit(logits)
    ent = -(p * np.log(np.clip(p, 1e-7, 1)) + (1 - p) * np.log(np.clip(1 - p, 1e-7, 1))).mean(1)
    maxp = np.maximum(p, 1 - p).mean(1)
    allc = np.prod(np.maximum(p, 1 - p), axis=1)
    return {"entropy": -ent, "max_prob": maxp, "all_correct": allc}   # 统一成"越大越可信"


def risk_at_coverage(is_error: np.ndarray, score: np.ndarray, coverage: float) -> float:
    order = np.argsort(-score)
    k = max(1, int(round(coverage * len(score))))
    return float(is_error[order[:k]].mean())


def oracle_risk(is_error: np.ndarray, coverage: float) -> float:
    k = max(1, int(round(coverage * len(is_error))))
    return float(np.sort(is_error)[:k].mean())


def fit_head(X: np.ndarray, y: np.ndarray, seed: int = 42):
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=15, random_state=seed)
    clf.fit(X, y)
    return clf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="result/analysis/selective/main_model_features.npz")
    ap.add_argument("--label-scores", default="result/analysis/selective/test_label_scores_insample.npz")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--coverage", type=float, default=0.80)
    ap.add_argument("--output-dir", default="result/analysis/e1")
    args = ap.parse_args()

    d = np.load(args.features)
    Xtr, ytr, ltr = d["train_features"], d["train_labels"], d["train_logits"]
    Xva, yva, lva = d["validation_features"], d["validation_labels"], d["validation_logits"]
    Xte, yte, lte = d["test_features"], d["test_labels"], d["test_logits"]

    # 目标：该记录是否五个标签全对
    def exact(y, lg):
        return np.all((expit(lg) > 0.5) == (y > 0.5), axis=1)

    head = fit_head(Xtr, exact(ytr, ltr).astype(int))
    lab = np.load(args.label_scores) if Path(args.label_scores).exists() else None
    if lab is not None:
        key = [k for k in lab.files if "score" in k.lower()]
        label_scores = lab[key[0]] if key else None
    else:
        label_scores = None

    cand: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, sv in scores_from_logits(lva).items():
        cand[name] = (sv, scores_from_logits(lte)[name])
    cand["head_gbdt"] = (head.predict_proba(Xva)[:, 1], head.predict_proba(Xte)[:, 1])
    if label_scores is not None and label_scores.shape[0] == len(Xte):
        agg_te = label_scores.min(axis=1)
        agg_va = label_scores[: len(Xva)].min(axis=1) if label_scores.shape[0] >= len(Xva) else None
        if agg_va is not None and len(agg_va) == len(Xva):
            cand["head_labelwise"] = (agg_va, agg_te)

    err_te = error_vector(yte, lte, error_unit="any_label")
    r_oracle = oracle_risk(err_te, args.coverage)
    r_current = risk_at_coverage(err_te, cand["max_prob"][1], args.coverage)
    budget = r_current - r_oracle

    rows = []
    for name, (sv_va, sv_te) in cand.items():
        # 用 fold 9 选 80% 覆盖率的阈值（与主协议一致）
        thr = np.quantile(sv_va, 1 - args.coverage, method="lower")
        mask = sv_te >= thr
        r = float(err_te[mask].mean()) if mask.any() else float("nan")
        cov = float(mask.mean())
        rows.append({
            "scorer": name,
            "coverage": round(cov, 4),
            "risk": round(r, 4),
            "recovered_pp": round((r_current - r) * 100, 3),
            "recovered_share": round((r_current - r) / budget, 4) if budget > 0 else float("nan"),
        })

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("risk")
    df.to_csv(out / "e1_recovered.csv", index=False)

    # 记录级 bootstrap（无 patient_id，标注为 record-level）
    rng = np.random.default_rng(42)
    n = len(err_te)
    boot = {k: [] for k in cand}
    for _ in range(args.bootstrap):
        idx = rng.integers(0, n, n)
        ro = oracle_risk(err_te[idx], args.coverage)
        rc = risk_at_coverage(err_te[idx], cand["max_prob"][1][idx], args.coverage)
        b = rc - ro
        for name, (_, sv) in cand.items():
            thr = np.quantile(cand[name][0], 1 - args.coverage, method="lower")
            m = sv[idx] >= thr
            r = err_te[idx][m].mean() if m.any() else np.nan
            boot[name].append((rc - r) / b if b > 0 else np.nan)
    ci = {k: (float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))) for k, v in boot.items()}

    summary = {
        "coverage": args.coverage,
        "n_test": int(n),
        "risk_oracle": round(r_oracle, 4),
        "risk_current_max_prob": round(r_current, 4),
        "budget_pp": round(budget * 100, 3),
        "table": df.to_dict(orient="records"),
        "recovered_share_ci_record_level": {k: [round(a, 4), round(b, 4)] for k, (a, b) in ci.items()},
        "bootstrap_unit": "record",
    }
    (out / "e1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "table"}, ensure_ascii=False, indent=2))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
