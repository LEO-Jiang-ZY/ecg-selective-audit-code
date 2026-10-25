"""逐标签拒答：高/低置信标签自动接受，中间区间转人工复核。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptbxl.inference import calibrated_logits, load_run

LOW_GRID = np.round(np.arange(0.01, 0.21, 0.01), 2)
HIGH_GRID = np.round(np.arange(0.80, 1.00, 0.01), 2)


def choose_band(prob: np.ndarray, label: np.ndarray, target: float, min_accepted: int = 100) -> dict:
    best = None
    fallback = None
    for low in LOW_GRID:
        for high in HIGH_GRID:
            accept = (prob <= low) | (prob >= high)
            total = int(accept.sum())
            if total < min_accepted:
                continue
            errors = int(((prob >= 0.5).astype(int) != label)[accept].sum())
            candidate = {
                "low": float(low),
                "high": float(high),
                "coverage": float(total / len(prob)),
                "error": float(errors / total),
                "accepted": total,
                "achievable": True,
            }
            if fallback is None or candidate["error"] < fallback["error"]:
                fallback = candidate
            if candidate["error"] <= target and (best is None or candidate["coverage"] > best["coverage"]):
                best = candidate
    if best is None:
        best = dict(fallback)
        best["achievable"] = False
    return best


def apply_bands(probs: np.ndarray, bands: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    accept = np.zeros_like(probs, dtype=bool)
    for index, band in enumerate(bands):
        accept[:, index] = (probs[:, index] <= band["low"]) | (probs[:, index] >= band["high"])
    return accept


def summarize(probs: np.ndarray, labels: np.ndarray, accept: np.ndarray) -> dict:
    pred = (probs >= 0.5).astype(int)
    wrong = pred != labels
    accepted_labels = int(accept.sum())
    wrong_accepted = int((wrong & accept).sum())
    full = accept.all(axis=1)
    full_any_error = float(wrong[full].any(axis=1).mean()) if full.any() else float("nan")
    return {
        "label_coverage": float(accepted_labels / accept.size),
        "label_error": float(wrong_accepted / accepted_labels) if accepted_labels else float("nan"),
        "full_accept_rate": float(full.mean()),
        "n_full_accept": int(full.sum()),
        "any_error_full_accept": full_any_error,
        "mean_deferred_labels": float((~accept).sum(axis=1).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="result/runs/seed_42/config.yaml")
    parser.add_argument("--calibration", default="result/runs/seed_42/calibration.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="result/analysis/per_label_abstention.csv")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    classes = cfg["data"]["classes"]
    val = np.load(Path(cfg["training"]["validation_logits"]))
    val_logits, _ = calibrated_logits(val["logits"].astype(float), args.calibration)
    val_labels = val["labels"].astype(float)
    val_probs = expit(val_logits)

    test = load_run(args.config, "test", device=args.device)
    test_logits, _ = calibrated_logits(test["logits"].astype(float), args.calibration)
    test_labels = test["labels"].astype(float)
    test_probs = expit(test_logits)

    schemes = {
        "fixed_0.10_0.90": [{"low": 0.10, "high": 0.90} for _ in classes],
        "tuned_5pct": [],
        "tuned_10pct": [],
    }
    tuned_meta = {}
    for name in ["tuned_5pct", "tuned_10pct"]:
        target = 0.05 if name == "tuned_5pct" else 0.10
        bands = []
        for index in range(len(classes)):
            band = choose_band(val_probs[:, index], val_labels[:, index], target)
            bands.append(band)
        schemes[name] = bands
        tuned_meta[name] = bands

    rows = []
    summary_rows = []
    for name, bands in schemes.items():
        val_accept = apply_bands(val_probs, bands)
        test_accept = apply_bands(test_probs, bands)
        summary = summarize(test_probs, test_labels, test_accept)
        summary["scheme"] = name
        summary_rows.append(summary)
        for index, label in enumerate(classes):
            rows.append(
                {
                    "scheme": name,
                    "label": label,
                    "low": bands[index]["low"],
                    "high": bands[index]["high"],
                    "val_coverage": bands[index].get("coverage", float("nan")),
                    "val_error": bands[index].get("error", float("nan")),
                    "test_coverage": float(test_accept[:, index].mean()),
                    "test_error": float(
                        (((test_probs[:, index] >= 0.5).astype(int) != test_labels[:, index]) & test_accept[:, index]).sum()
                        / max(1, test_accept[:, index].sum())
                    ),
                    "test_accepted": int(test_accept[:, index].sum()),
                }
            )

    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    summary_frame = pd.DataFrame(summary_rows)[
        ["scheme", "label_coverage", "label_error", "full_accept_rate", "n_full_accept", "any_error_full_accept", "mean_deferred_labels"]
    ]
    summary_frame.to_csv(output.with_suffix(".summary.csv"), index=False)

    reference = json.loads(Path("result/results/evaluation.json").read_text(encoding="utf-8"))
    payload = {
        "protocol": "per-label abstention; thresholds selected on fold 9, evaluated on fold 10",
        "schemes": tuned_meta,
        "summary": summary_rows,
        "reference_record_level_80pct_entropy": {
            "coverage": reference["accepted"]["coverage"],
            "any_label": reference["accepted"]["any_label"],
            "macro_label": reference["accepted"]["macro_label"],
        },
        "note": "label_coverage 是接受标签数占总标签数的比例；full_accept_rate 是五个标签全部接受的记录比例。",
    }
    output.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"已写入 {output}、{output.with_suffix('.summary.csv')} 与 {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
