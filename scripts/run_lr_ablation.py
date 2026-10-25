"""E2：训练侧学习率消融（3 档），复用 train → calibrate → evaluate 全链路。

不改 train.py：通过为每档生成独立 config 覆盖 training.learning_rate。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

PY = sys.executable
ROOT = Path("result/runs/lr_ablation")
BASE = Path("result/runs/seed_42/config.yaml")


def build_config(lr: float, tag: str) -> Path:
    cfg = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    run = ROOT / tag
    run.mkdir(parents=True, exist_ok=True)
    cfg["training"]["learning_rate"] = lr
    cfg["training"]["seed"] = 42
    cfg["training"]["checkpoint"] = str(run / "best.pt")
    cfg["training"]["validation_logits"] = str(run / "val_logits.npz")
    cfg["calibration"]["output"] = str(run / "calibration.json")
    cfg["evaluation"]["output"] = str(run / "evaluation.json")
    path = run / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    lrs = [3e-4, 1e-3, 3e-3]
    summary = []
    for lr in lrs:
        tag = f"lr_{lr:.0e}".replace("e-0", "e-")
        cfg = build_config(lr, tag)
        run([PY, "code/scripts/train.py", "--config", str(cfg)])
        run([PY, "code/scripts/fit_calibration.py", "--config", str(cfg)])
        run([PY, "code/scripts/evaluate.py", "--config", str(cfg)])
        ev = json.loads((ROOT / tag / "evaluation.json").read_text(encoding="utf-8"))
        cal = json.loads((ROOT / tag / "calibration.json").read_text(encoding="utf-8"))
        hist = json.loads((ROOT / tag / "history.json").read_text(encoding="utf-8"))
        best = min(hist, key=lambda h: h["validation_loss"])
        summary.append({
            "learning_rate": lr,
            "best_epoch": best["epoch"],
            "best_val_loss": round(best["validation_loss"], 4),
            "test_exact_match": round(ev["overall"]["exact_match"], 4),
            "macro_auroc": round(ev["auroc"]["macro"], 4),
            "macro_auprc": round(ev["auprc"]["macro"], 4),
            "accepted_any_label": round(ev["accepted"]["any_label"], 4),
            "coverage": round(ev["accepted"]["coverage"], 4),
            "temperature": round(cal["temperature"], 4),
        })
        print(json.dumps(summary[-1], ensure_ascii=False), flush=True)

    (ROOT / "lr_ablation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(f"{r['learning_rate']:.0e}  val {r['best_val_loss']}  auroc {r['macro_auroc']}  acc_err {r['accepted_any_label']}" for r in summary))


if __name__ == "__main__":
    main()
