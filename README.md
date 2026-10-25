# Calibration Is Not Enough — code, data and tools

Code, derived data and analysis tools for the MD6329 mini project
**"Calibration Is Not Enough: A Three-Level Audit of Selective Prediction for Multi-Label ECG
Classification"** (Zhengyu Jiang, G2605829A, MSc AI in Medicine, NTU).

The question is narrow and practical: *does a better-calibrated ECG classifier give a safer
triage policy?* On PTB-XL the answer is no — temperature scaling improves expected calibration
error by 3.5× while leaving the risk–coverage curve essentially unchanged. The repository contains
everything needed to re-derive that result, plus the three-level audit (calibration → score →
granularity) and the error-budget analysis that followed it.

---

## Headline numbers

| Quantity | Value |
|---|---|
| Model | ECGResNet1D, 51,429 parameters, trained from scratch on folds 1–8 |
| Test fold | fold 10, 2,198 records / 1,904 patients, evaluated once |
| Macro AUROC / AUPRC | 0.9076 [0.8999, 0.9157] / 0.7701 [0.7534, 0.7872] |
| ECE (15 bins) | 0.0238 → 0.0068 after temperature scaling |
| AURC | 0.1900 → 0.1914 (near-invariant) |
| Accepted any-label error @ 80 % coverage | 30.98 % → 31.23 % (+0.25 pp) |
| Budget available to any abstention score | 6.5 pp; a perfect ordering still leaves 23.6 % error |

Intervals are patient-level bootstrap (1,000 resamples over 1,904 patients).

---

## Quick start

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. data — not redistributed, see data/README.md
#    put PTB-XL v1.0.3 at: data/PTB-XL_ECG_dataset_1.0.3/...

# 3. one command per stage
python scripts/build_cache.py     --config configs/default.yaml   # 21,799 × 12 × 1,000 float32, ~1 GB
python scripts/train.py           --config configs/default.yaml   # 20 epochs, ~11 min on one GPU
python scripts/fit_calibration.py --config configs/default.yaml   # temperature + threshold on fold 9
python scripts/evaluate.py        --config configs/default.yaml   # single pass over fold 10
```

`scripts/summarize.py` turns the JSON artefacts into a Markdown report; `scripts/run_experiment.py`
runs the whole pipeline for several seeds.

**Protocol rule that the code enforces.** Temperature, abstention threshold and the correctness
head are fitted on fold 9 only. `fit_calibration.py` refuses any logits file whose `split` field is
not `validation`, and `evaluate.py` raises if the checkpoint is missing instead of silently
evaluating a randomly initialised model.

---

## Repository layout

```
configs/default.yaml        all hyperparameters and paths
src/ptbxl/                  library: io, cache, labels, cohort, split, model, train, budget,
                            calibration, metrics, analysis, selective, features, inference, palette
scripts/                    one file per stage (see table below)
data/splits/                cohort_metadata.csv — derived per-record metadata (no waveforms)
results/seed_{42,43,44}/    calibration.json, evaluation.json, history.json, run_meta.json, best.pt
results/analyses/           budget, frontier, subgroup and per-label tables as CSV
docs/REPRODUCE.md           exact commands, timings and expected outputs
docs/RESULTS_INDEX.md       which script produces which number
```

### Scripts

| Stage | Script | Output |
|---|---|---|
| Cache waveforms | `build_cache.py` | memmap `(21799, 12, 1000)` float32 |
| Train | `train.py` | `best.pt`, `val_logits.npz` (`split="validation"`) |
| Calibrate | `fit_calibration.py` | `calibration.json` (T, τ) |
| Evaluate | `evaluate.py` | `evaluation.json` (metrics, subgroups, shift, bootstrap) |
| Report | `summarize.py` | Markdown run report |
| Multi-seed | `run_experiment.py` | `results/seed_*/` for each seed |
| Error budget | `decompose_gap.py` | `gap_summary.csv`, `gap_decomposition.csv` |
| Frontier | `frontier_analysis.py` | `frontier_summary.csv`, `frontier_paired.csv` |
| Risk targets | `risk_target.py` | `risk_target.csv` |
| Per-label abstention | `per_label_abstention.py` | `per_label_abstention*.csv` |
| Learned scorers | `selective_head_insample.py`, `correctness_head.py`, `oof_models.py` | head metrics |
| Budget closure | `gap_closure.py` | `gap_closure.csv` (seven scorer families) |
| Recipe ablation | `train.py --learning-rate` | `lr_ablation.csv` |
| Risk control | `conformal_risk_control.py` | `crc_summary.json`, `crc_*.csv` |
| Architecture baselines | `model_comparison.py` | `model_comparison.csv` |
| Figures | `paper_figures.py`, `figure_head.py`, `paper_pipeline_figure.py` | PDF/PNG |
| Reference check | `verify_references.py` | Crossref/arXiv verification table |

---

## Data

PTB-XL v1.0.3 is **not** redistributed here: 21,799 records are about 1 GB of waveform data.
Download it from PhysioNet (<https://physionet.org/content/ptb-xl/1.0.3/>) and place it at the path
in `configs/default.yaml`.

`data/splits/cohort_metadata.csv` contains the derived per-record table used in this project —
patient id, official fold, age, sex, device, acquisition site, the four native quality annotations,
human-validation flag and the record path. It contains no signal data. PTB-XL is released under
CC-BY 4.0; keep the attribution in `LICENSE` if you redistribute this table.

Two properties of the metadata matter and are handled in `src/ptbxl/cohort.py`:

- `age = 300` is the HIPAA code for "older than 89" (293 records) and is grouped as `>89`;
- `validated_by_human` is 100 % true in folds 9–10 but about 67 % in training, so it is confounded
  with the split and is never used as a test-time subgroup.

---

## Reproducibility notes

- **Folds are patient-level and fixed**: folds 1–8 train, fold 9 fits every decision, fold 10 is
  touched once. Do not re-split.
- **Seeds**: 42, 43, 44 are all included with their artefacts. Seed-to-seed standard deviation of
  accepted error is 1.0 pp — larger than the 0.47 pp correctness-head effect, which is why that
  effect is reported as suggestive rather than established.
- **Timings** on one consumer GPU: cache build 4 min, 20 epochs 11 min, calibration and evaluation
  under a minute each.
- **What is not included**: the waveform cache (`data/cache.npy`, ~1 GB) and the extracted feature
  matrices (`*.npz`). Both are regenerated by `build_cache.py` and `selective_head_insample.py`
  in a few minutes.

---

## What this code does *not* claim

Single public cohort, one architecture family, no external validation, no reader study, no cost
model. Thresholds for the per-label policy were tuned on all of fold 9, which risks overfitting
five bands to 2,183 records; nested cross-fitting is the remedy and is left as future work.
"Better" throughout means a lower empirical selective risk at a stated coverage, not demonstrated
clinical benefit.

---

## License

Code: MIT (see `LICENSE`). Derived tables: CC-BY 4.0, following PTB-XL.

## Citation

```bibtex
@misc{jiang2026calibration,
  author = {Jiang, Zhengyu},
  title  = {Calibration Is Not Enough: A Three-Level Audit of Selective Prediction
            for Multi-Label ECG Classification},
  year   = {2026},
  note   = {MD6329 Deep Learning for Healthcare AI mini project, NTU}
}
```
