# Reproducing the results

Every number in the report is produced by one of the commands below from the artefacts in
`results/`. Times are for one consumer GPU (RTX-class) with the waveform cache built; CPU runs are
5–8× slower.

## 0. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 1. Build the waveform cache (once, ~4 min)

```bash
python scripts/build_cache.py --config configs/default.yaml
```

Writes a float32 memmap of shape `(21799, 12, 1000)` (~1 GB) plus an index CSV. The loader is
verified record-by-record against the WFDB reference reader (`np.allclose`, 1e-6). Without the
cache, one epoch costs about 5.8 min of pure I/O.

## 2. Train (~11 min per seed)

```bash
python scripts/train.py --config configs/default.yaml
```

Expected on seed 42: best epoch 14, validation loss 0.2993, macro AUROC ≈ 0.908. The script writes
`best.pt` and `val_logits.npz`, and stamps the logits file with `split="validation"`.

## 3. Fit the calibration layer on fold 9 (seconds)

```bash
python scripts/fit_calibration.py --config configs/default.yaml
```

Expected: `T = 1.1868`, entropy threshold `τ = 0.4264` at 80 % target coverage, fold-9 ECE
0.0233 → 0.0051. The script **refuses** any logits file whose `split` is not `validation`, so fold 10
cannot leak into calibration by accident.

## 4. Evaluate fold 10 once (under a minute)

```bash
python scripts/evaluate.py --config configs/default.yaml
```

Expected: coverage 79.62 %, accepted any-label error 31.14 % (patient bootstrap
[28.92, 33.58]), macro AUROC 0.9076, macro AUPRC 0.7701. `evaluate.py` raises if the checkpoint is
missing rather than evaluating a randomly initialised model.

## 5. Multi-seed replication (~11 min per seed)

```bash
python scripts/run_experiment.py --seeds 42 43 44 --skip-existing
```

Expected: accepted error 31.14 / 29.81 / 31.81, i.e. +0.18 / +0.37 / +0.46 points over the
uncalibrated baseline at matched coverage, while macro AUROC stays within 0.0004.

## 6. Decision-layer analyses (no retraining)

```bash
python scripts/abstention_compare.py     --config results/seed_42/config.yaml
python scripts/frontier_analysis.py      --config results/seed_42/config.yaml
python scripts/risk_target.py            --config results/seed_42/config.yaml
python scripts/per_label_abstention.py   --config results/seed_42/config.yaml
python scripts/decompose_gap.py          --device cpu --bootstrap 300
python scripts/selective_head_insample.py
```

Expected headline numbers: AURC 0.1914 (entropy) / 0.1877 (max-prob) / 0.1872 (all-correct);
error budget 6.9 pp at 80 % coverage with a 23.1 % oracle floor; label-level abstention 5.61 %
label error at 39.8 % full-record issuance.

## 7. Budget closure and learning-rate ablation (~30 min)

```bash
python scripts/gap_closure.py --device cuda --bootstrap 500   # seven scorer families
for lr in 3e-4 3e-3; do
  python scripts/train.py --config results/ablation/lr_$lr/config.yaml --learning-rate $lr
  python scripts/fit_calibration.py --config results/ablation/lr_$lr/config.yaml
  python scripts/evaluate.py        --config results/ablation/lr_$lr/config.yaml
done
```

Expected: at 80 % coverage the budget is 6.5 pp; the best post-hoc score recovers 2.6 % of it, a
three-seed ensemble the same, and a feature-aware head 6.1 % (interval spanning zero). The learning-rate
sweep gives macro AUROC 0.9021 / 0.9076 / 0.9065 for 3e-4 / 1e-3 / 3e-3 — note that validation loss
ranks these the other way round.

## 8. Conformal risk control (< 1 min)

```bash
python scripts/conformal_risk_control.py --device cuda
```

Expected at α = 0.20: record-level CRC issues 49.9 % of records at 18.2 % conditional error and
9.1 % marginal risk; the empirical Clopper–Pearson threshold of Table 10 issues 45.5 % at 16.5 %.
Label-level CRC issues 74.1 % of labels at 5.14 % error with 36.7 % of records cleared in full.
The device split keeps the marginal bound satisfied while conditional error ranges 15.1–25.5 %.

## 9. Architecture baselines (~35 min)

```bash
for name in resunet1d resnet50_1d transformer1d; do
  python scripts/train.py          --config results/model_compare/$name/config.yaml
  python scripts/fit_calibration.py --config results/model_compare/$name/config.yaml
  python scripts/evaluate.py        --config results/model_compare/$name/config.yaml
done
python scripts/model_comparison.py
```

Expected: AURC 0.2041 / 0.1952 / 0.2379 against 0.1877 for the 51k model — more capacity does not
improve the frontier under this training budget.

## Protocol guards

Three things the code refuses to do, because each one would silently invalidate a result:

| Guard | Where | Why |
|---|---|---|
| Calibrate on a non-validation split | `fit_calibration.py` | temperature and threshold must come from fold 9 |
| Evaluate without a checkpoint | `evaluate.py` | a random model still produces plausible-looking numbers |
| Re-split the cohort | `src/ptbxl/split.py` | official patient-level folds are part of the protocol |

## Determinism

`set_seed` fixes Python, NumPy, PyTorch and CUDA seeds and disables cuDNN autotuning. Bitwise
reproduction is not guaranteed across GPU models; the three included seeds bound the variation that
matters (1.0 pp on accepted error).
