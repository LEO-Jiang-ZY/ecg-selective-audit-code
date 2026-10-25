# Which script produces which number

Read this before quoting a number anywhere: it tells you the artefact and the command behind it.
Every value below is regenerated from `results/`, not transcribed.

| Reported quantity | Value (seed 42) | Artefact | Command |
|---|---|---|---|
| Best validation loss | 0.2993 | `results/seed_42/history.json` | `scripts/train.py` |
| Temperature | 1.1868 | `results/seed_42/calibration.json` | `scripts/fit_calibration.py` |
| Entropy threshold (80 % target) | 0.4264 | `results/seed_42/calibration.json` | `scripts/fit_calibration.py` |
| Fold-9 ECE | 0.0233 → 0.0051 | `results/seed_42/calibration.json` | same |
| Test coverage | 79.62 % | `results/seed_42/evaluation.json` | `scripts/evaluate.py` |
| Accepted any-label error | 31.14 % [28.92, 33.58] | `results/seed_42/evaluation.json` | same |
| Accepted macro-label error | 9.90 % | `results/seed_42/evaluation.json` | same |
| Macro AUROC / AUPRC | 0.9076 / 0.7701 | `results/seed_42/evaluation.json` | same |
| Exact match | 61.10 % [58.95, 63.22] | `results/analyses/fold10_ci.csv` | bootstrap over `evaluation.json` |
| ERM baseline at matched coverage | 30.96 % | `results/seed_42/evaluation.json` → `baseline.matched_coverage` | same |
| Three-seed table | +0.18 / +0.37 / +0.46 | `results/analyses/multiseed_summary.json` | `scripts/run_experiment.py --seeds 42 43 44` |
| AURC by score | 0.1914 / 0.1877 / 0.1872 | `results/analyses/frontier_summary.csv` | `scripts/frontier_analysis.py` |
| Coverage at a 20 % error target | 42.3 / 45.5 / 46.2 % | `results/analyses/frontier_summary.csv` | same |
| Error budget at 80 % coverage | 6.5 pp, floor 23.6 % | `results/analyses/gap_summary.csv` | `scripts/decompose_gap.py` |
| What each scorer recovers | head 6.1 %, all-correct 2.6 % | `results/analyses/gap_closure.csv` | `scripts/gap_closure.py` |
| Learning-rate ablation | AUROC 0.9021 / 0.9076 / 0.9065 | `results/analyses/lr_ablation.csv` | `train.py --learning-rate` |
| Conformal risk control | CRC cov 49.9 % vs CP 45.5 % at α=0.20 | `results/analyses/crc_summary.json` | `scripts/conformal_risk_control.py` |
| Risk-controlled operating points | Table 10 | `results/analyses/risk_target.csv` | `scripts/risk_target.py` |
| Label-level abstention | 5.61 % at 39.8 % full issuance | `results/analyses/per_label_abstention.summary.csv` | `scripts/per_label_abstention.py` |
| Learned head (in-sample) | AURC 0.1837 vs 0.1875 | `results/analyses/` head output | `scripts/selective_head_insample.py` |
| Architecture baseline table | Table 9 | `results/analyses/model_comparison.csv` | `scripts/model_comparison.py` |
| Subgroup error | Figure 7 | `results/seed_42/evaluation.json` → `subgroups` | `scripts/evaluate.py` |
| Device / site shift | TVD 0.326 / 0.252 | `results/seed_42/evaluation.json` → `shift` | same |

## Three things worth knowing before you re-run anything

1. **`evaluation.json` is a single-pass artefact.** Re-running `evaluate.py` on a *different*
   checkpoint overwrites the numbers the report quotes. Keep the per-seed folders intact.
2. **The budget numbers depend on the reference ordering.** `gap_summary.csv` is computed against a
   three-seed ensemble of the same architecture; a four-architecture ensemble is worse than the best
   single model here, so it cannot serve as a reference. The oracle column does not depend on the choice.
3. **Bootstrap resamples patients, not records.** A naive record-level bootstrap gives intervals
   roughly 30 % narrower and is wrong here, because the 2,198 test records come from 1,904 patients.
