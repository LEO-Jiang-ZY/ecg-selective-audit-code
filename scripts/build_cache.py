"""构建 100 Hz 波形 memmap 缓存。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptbxl.cache import build_cache
from ptbxl.labels import load_and_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="code/configs/default.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--index", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg = cfg["data"]
    labeling = cfg.get("labeling", {})
    root = Path(data_cfg["root"])
    frame = load_and_label(
        root / data_cfg["metadata"],
        root / data_cfg["scp_statements"],
        data_cfg["classes"],
        min_scp_probability=labeling.get("min_scp_probability"),
        drop_all_zero=bool(labeling.get("drop_all_zero", False)),
    )
    if args.limit:
        frame = frame.head(args.limit)

    output = args.output or data_cfg.get("cache", "result/cache/ptbxl_100hz.npy")
    index = args.index or data_cfg.get("cache_index", "result/cache/index.csv")
    meta = build_cache(
        frame,
        root,
        output,
        index,
        target_length=int(data_cfg["target_length"]),
        sampling_rate=int(data_cfg["sampling_rate"]),
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
