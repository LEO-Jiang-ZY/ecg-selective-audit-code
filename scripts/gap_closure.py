"""E1：把 6.9 pp 的误差预算摊开——四类打分器各自能回收多少。

定义（与论文 §5.4 一致）：

    recovered(θ) = (R_current − R_θ) / (R_current − R_oracle)

R 为 80% coverage 下的 accepted any-label error；oracle 按真实对错排序，
因此 R_oracle 是任何排序都不可再降低的部分。

用法：
    python code/scripts/gap_closure.py --device cuda
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
from ptbxl.inference import load_run  # noqa: E402
from ptbxl.metrics import macro_auroc, macro_auprc  # noqa: E402

COVERAGE = 0.80
RUNS = {s: f"result/runs/seed_{s}/config.yaml" for s in (42, 43, 44)}


# ---------- 打分器 ----------
def entropy_score(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-7, 1 - 1e-7)
    return -(q * np.log(q) + (1 - q) * np.log1p(-q)).mean(1)


def max_prob_score(p: np.ndarray) -> np.ndarray:
    return np.maximum(p, 1 - p).mean(1)


def all_correct_score(p: np.ndarray) -> np.ndarray:
    return np.prod(np.maximum(p, 1 - p), axis=1)


def head_scores(train_feat, train_logits, train_labels, test_feat, test_logits, seed=42):
    """记录级 head 与逐标签 head，返回两个分数（越高越可信）。"""
    def correct_any(logits, labels):
        pred = (expit(logits) > 0.5)
        return (~(pred != (labels > 0.5)).any(axis=1)).astype(int)

    y_rec = correct_any(train_logits, train_labels)
    rec = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=None,
                                         random_state=seed)
    rec.fit(np.hstack([train_feat, train_logits]), y_rec)
    rec_score = rec.predict_proba(np.hstack([test_feat, test_logits]))[:, 1]

    probs = []
    p_train = expit(train_logits)
    p_test = expit(test_logits)
    for k in range(train_labels.shape[1]):
        y_k = ((p_train[:, k] > 0.5) == (train_labels[:, k] > 0.5)).astype(int)
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, random_state=seed + k)
        clf.fit(np.column_stack([train_feat, train_logits[:, k]]), y_k)
        probs.append(clf.predict_proba(np.column_stack([test_feat, test_logits[:, k]]))[:, 1])
    label_score = np.prod(np.vstack(probs), axis=0)
    return rec_score, label_score


# ---------- 评估 ----------
def risk_at_coverage(score: np.ndarray, wrong: np.ndarray, coverage: float = COVERAGE) -> float:
    order = np.argsort(-score, kind="stable")
    k = max(1, int(round(coverage * len(score))))
    return float(wrong[order[:k]].mean())


def aurc(score: np.ndarray, wrong: np.ndarray) -> float:
    order = np.argsort(-score, kind="stable")
    err = wrong[order]
    coverages = np.arange(1, len(err) + 1) / len(err)
    return float(np.trapz(np.cumsum(err) / np.arange(1, len(err) + 1), coverages))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--output-dir", default="result/analysis")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) 三个 seed 的测试 logits 与各自温度
    test = {}
    for seed, cfg in RUNS.items():
        run = load_run(cfg, "test", device=args.device)
        cal = json.loads(Path(f"result/runs/seed_{seed}/calibration.json").read_text(encoding="utf-8"))
        p = expit(run["logits"].astype(float) / float(cal["temperature"]))
        test[seed] = {"p": p, "labels": run["labels"].astype(float),
                      "pids": run["split_frame"]["patient_id"].to_numpy()}
        print(f"seed {seed}: n={len(p)} temperature={cal['temperature']:.4f}")

    labels = test[42]["labels"]
    pids = test[42]["pids"]
    p42 = test[42]["p"]
    wrong = ( (p42 > 0.5) != (labels > 0.5) ).any(axis=1).astype(float)   # any-label error
    correct = 1.0 - wrong

    # 2) 特征（seed 42，已缓存）
    npz = np.load("result/analysis/selective/main_model_features.npz")
    head_rec, head_lab = head_scores(npz["train_features"].astype(float), npz["train_logits"].astype(float),
                                     npz["train_labels"].astype(float), npz["test_features"].astype(float),
                                     npz["test_logits"].astype(float))
    print("heads trained")

    # 3) 集成分数
    mean_p = np.mean([test[s]["p"] for s in test], axis=0)
    member_conf = np.stack([max_prob_score(test[s]["p"]) for s in test])
    disagreement = member_conf.std(axis=0)                       # 越低越可信
    ens_score = max_prob_score(mean_p)

    scorers = {
        "entropy (post-hoc)":            -entropy_score(p42),
        "max-prob (post-hoc)":            max_prob_score(p42),
        "all-correct (post-hoc)":         all_correct_score(p42),
        "ensemble mean (3 seeds)":        ens_score,
        "ensemble disagreement":         -disagreement,
        "learned head, record-level":     head_rec,
        "learned head, label-wise":       head_lab,
    }

    # 4) 基准与预算
    r_oracle = risk_at_coverage(correct, wrong)          # 按真实对错排序
    r_current = risk_at_coverage(scorers["max-prob (post-hoc)"], wrong)
    budget = r_current - r_oracle
    print(f"R_oracle {r_oracle:.4f} | R_current {r_current:.4f} | budget {100*budget:.2f} pp")

    rows = []
    for name, s in scorers.items():
        r = risk_at_coverage(s, wrong)
        rows.append({
            "scorer": name,
            "family": ("post-hoc" if "post-hoc" in name else
                       "ensemble" if "ensemble" in name else "feature-aware"),
            "risk_at_80": r,
            "delta_pp_vs_maxprob": 100 * (r - r_current),
            "recovered": (r_current - r) / budget if budget > 0 else np.nan,
            "aurc": aurc(s, wrong),
            "auroc_macro": macro_auroc(labels, p42, from_logits=False)["macro"],
            "auprc_macro": macro_auprc(labels, p42, from_logits=False)["macro"],
        })

    table = pd.DataFrame(rows).sort_values("risk_at_80")
    table.to_csv(out / "gap_closure.csv", index=False)
    print("\n", table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # 5) 最佳特征感知打分器的 recovered 置信区间（患者级 bootstrap）
    best_name = table[table.family == "feature-aware"].iloc[0]["scorer"]
    best = scorers[best_name]
    rng = np.random.default_rng(42)
    groups = {pid: np.where(pids == pid)[0] for pid in np.unique(pids)}
    keys = list(groups)
    vals = []
    for _ in range(args.bootstrap):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys), replace=True)])
        b_oracle = risk_at_coverage(correct[idx], wrong[idx])
        b_cur = risk_at_coverage(scorers["max-prob (post-hoc)"][idx], wrong[idx])
        b_best = risk_at_coverage(best[idx], wrong[idx])
        denom = b_cur - b_oracle
        if denom > 1e-9:
            vals.append((b_cur - b_best) / denom)
    lo, hi = np.quantile(vals, [0.025, 0.975]) if vals else (np.nan, np.nan)
    print(f"\n{best_name}: recovered 95% CI [{lo:.3f}, {hi:.3f}] (n_boot={len(vals)})")

    summary = {
        "coverage": COVERAGE,
        "r_oracle": r_oracle,
        "r_current_maxprob": r_current,
        "budget_pp": 100 * budget,
        "best_feature_aware": best_name,
        "best_recovered": float(table.iloc[0]["recovered"]),
        "best_recovered_ci": [float(lo), float(hi)],
        "posthoc_best_recovered": float(table[table.family == "post-hoc"].iloc[0]["recovered"]),
    }
    (out / "gap_closure_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
