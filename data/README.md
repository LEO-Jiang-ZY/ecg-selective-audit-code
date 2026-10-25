# PTB-XL v1.0.3 — how to obtain the data

The waveforms are **not** included in this repository (about 1 GB). Everything else needed to
reproduce the results is here, including the derived per-record table.

## 1. Download

```bash
# option A — PhysioNet direct
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/

# option B — browser
# https://physionet.org/content/ptb-xl/1.0.3/  →  "Download the ZIP file"
```

## 2. Place it where the config expects

```
data/PTB-XL_ECG_dataset_1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
├── ptbxl_database.csv
├── scp_statements.csv
└── records100/
```

`configs/default.yaml` → `data.root` points at that directory. Adjust the path if you keep the
dataset elsewhere; nothing else needs to change.

## 3. Use version 1.0.3, at 100 Hz

- **v1.0.3 only.** v1.0.1 has a different record count and label distribution; mixing versions
  silently changes prevalence and every downstream number.
- **`records100/` only.** The 500 Hz copy is not used anywhere, so no sampling-rate mixing occurs.
  Each record becomes 12 leads × 1,000 samples over a 10-second window.

## 4. What is already included

`data/splits/cohort_metadata.csv` — 21,799 rows, 14 columns, no signal data:

| column | meaning |
|---|---|
| `patient_id`, `strat_fold` | official patient-level fold assignment (1–10) |
| `age`, `sex`, `height`, `weight` | demographics; `age = 300` is the "> 89" HIPAA code |
| `device`, `site` | acquisition device and site, used for the shift analysis |
| `baseline_drift`, `static_noise`, `burst_noise`, `electrodes_problems` | native quality annotations |
| `validated_by_human` | human validation flag (confounded with the split — see below) |
| `filename_lr` | relative path to the 100 Hz record |

## 5. Two traps in the metadata

1. **`age = 300`** appears in 293 records and means "older than 89". Treating it as a continuous
   value distorts every age statistic; `src/ptbxl/cohort.py` buckets it as `>89`.
2. **`validated_by_human`** is 100 % true in folds 9 and 10 but only about 67 % in training. It is
   therefore perfectly confounded with the split and cannot be used as a test-time subgroup.

## 6. Licence

PTB-XL is released under CC-BY 4.0. Cite Wagner et al., *Scientific Data* 7, 154 (2020),
<https://doi.org/10.1038/s41597-020-0495-6>. The derived table in `data/splits/` is redistributed
under the same licence.
