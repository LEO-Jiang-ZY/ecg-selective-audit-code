"""按 seed 运行完整实验并汇总 mean ± 95% CI。"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "code" / "scripts"


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def prepare_config(base: dict, seed: int, run_dir: Path) -> Path:
    cfg = copy.deepcopy(base)
    cfg["training"]["seed"] = int(seed)
    cfg["training"]["checkpoint"] = str(run_dir / "best.pt")
    cfg["training"]["validation_logits"] = str(run_dir / "val_logits.npz")
    cfg["calibration"]["output"] = str(run_dir / "calibration.json")
    cfg["evaluation"]["output"] = str(run_dir / "evaluation.json")
    path = run_dir / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def summarize(seeds: list[int], runs_root: Path) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        run_dir = runs_root / f"seed_{seed}"
        evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
        calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
        meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
        primary = evaluation["primary_error_unit"]
        baseline = evaluation["baseline"]["matched_coverage"]["accepted_error"][primary]
        accepted = evaluation["accepted"][primary]
        rows.append(
            {
                "seed": seed,
                "temperature": calibration["temperature"],
                "entropy_threshold": calibration["entropy_threshold"],
                "best_epoch": meta.get("best_epoch"),
                "val_loss": meta.get("best_validation_loss"),
                "coverage": evaluation["accepted"]["coverage"],
                "baseline_error": baseline,
                "accepted_error": accepted,
                "delta": accepted - baseline,
                "macro_auroc": evaluation["auroc"]["macro"],
                "macro_auprc": evaluation["auprc"]["macro"],
                "exact_match": evaluation["overall"]["exact_match"],
                "any_label_error": evaluation["overall"]["any_label_error"],
                "macro_label_error": evaluation["overall"]["macro_label_error"],
                "ece": evaluation["overall"]["ece"],
            }
        )
    return pd.DataFrame(rows)


def write_summary(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "summary.csv", index=False)

    numeric = [column for column in frame.columns if column != "seed"]
    rows = []
    for column in numeric:
        values = frame[column].astype(float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        if len(values) > 1:
            half = float(stats.t.ppf(0.975, len(values) - 1) * sd / (len(values) ** 0.5))
        else:
            half = float("nan")
        rows.append([column, mean, sd, mean - half, mean + half, float(values.min()), float(values.max())])
    summary = pd.DataFrame(rows, columns=["metric", "mean", "sd", "ci_low", "ci_high", "min", "max"])
    summary.to_csv(output_dir / "aggregate.csv", index=False)

    lines = ["# 多 seed 汇总", "", f"共 {len(frame)} 个 seed。", ""]
    lines.append("| metric | mean | sd | 95% CI | min | max |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['metric']} | {row['mean']:.4f} | {row['sd']:.4f} | "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] | {row['min']:.4f} | {row['max']:.4f} |"
        )
    lines.append("")
    lines.append("## 逐 seed")
    lines.append("")
    lines.append(frame.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="code/configs/default.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--runs-root", default="result/runs")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    base = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    runs_root = ROOT / args.runs_root
    runs_root.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        for seed in args.seeds:
            run_dir = runs_root / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            required = [run_dir / "evaluation.json", run_dir / "calibration.json"]
            if args.skip_existing and all(path.exists() for path in required):
                print(f"skip seed {seed}（已有结果）", flush=True)
                continue
            config_path = prepare_config(base, seed, run_dir)
            command = [sys.executable, str(SCRIPTS / "train.py"), "--config", str(config_path)]
            if args.epochs is not None:
                command += ["--epochs", str(args.epochs)]
            run(command)
            run([sys.executable, str(SCRIPTS / "fit_calibration.py"), "--config", str(config_path)])
            run([sys.executable, str(SCRIPTS / "evaluate.py"), "--config", str(config_path)])

    frame = summarize(args.seeds, runs_root)
    write_summary(frame, runs_root)
    print(frame.to_string(index=False))
    print(f"已写入 {runs_root / 'summary.csv'} 与 {runs_root / 'summary.md'}")


if __name__ == "__main__":
    main()
