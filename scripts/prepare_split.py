"""Prepare a provisional development split; does not access FCD or train a model."""
import argparse
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import random

import lmdb
import numpy as np
from PIL import Image


def group_records(records):
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, record in enumerate(records):
        for field in ("rgb_sha256", "dhash"):
            key = (field, record[field])
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[find(i)].append(record["index"])
    return list(groups.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="data/raw/DocTamperV1-TrainingSet")
    parser.add_argument("--output", default="outputs/split_v1")
    parser.add_argument("--candidates", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overfit-config", default="outputs/overfit_16_continue/config.json")
    args = parser.parse_args()
    if args.candidates < 2:
        parser.error("Need at least two candidates")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Output folder is not empty; use a new --output")
    excluded = set(json.loads(Path(args.overfit_config).read_text(encoding="utf-8-sig"))["indices"])
    env = lmdb.open(str(Path(args.database).resolve()), readonly=True,
                    lock=False, create=False, readahead=False, meminit=False)
    records = []
    try:
        with env.begin() as txn:
            count = int(txn.get(b"num-samples"))
            available = [i for i in range(count) if i not in excluded]
            if args.candidates > len(available):
                raise ValueError("Not enough available samples")
            indices = random.Random(args.seed).sample(available, args.candidates)
            for position, index in enumerate(indices, 1):
                image_bytes = txn.get(f"image-{index:09d}".encode())
                label_bytes = txn.get(f"label-{index:09d}".encode())
                if image_bytes is None or label_bytes is None:
                    raise ValueError(f"Missing image/label at {index}")
                with Image.open(BytesIO(image_bytes)) as image:
                    image = image.convert("RGB")
                    rgb = np.array(image)
                    tiny = np.array(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
                with Image.open(BytesIO(label_bytes)) as label:
                    mask = np.array(label)
                if rgb.shape != (512, 512, 3) or mask.shape != (512, 512):
                    raise ValueError(f"Invalid shapes at {index}")
                if not np.isin(mask, [0, 1, 255]).all():
                    raise ValueError(f"Unexpected mask values at {index}")
                records.append({"index": index,
                                "rgb_sha256": sha256(rgb.tobytes()).hexdigest(),
                                "dhash": np.packbits(tiny[:, 1:] > tiny[:, :-1]).tobytes().hex(),
                                "positive_fraction": float((mask > 0).mean())})
                if position % 1000 == 0 or position == len(indices):
                    print(f"Audited {position}/{len(indices)} samples", flush=True)
    finally:
        env.close()
    # Union all exact RGB or exact 64-bit dHash matches before splitting.
    # dHash collisions group conservatively; unequal hashes do NOT imply unrelated sources.
    groups = group_records(records)
    if len(groups) < 2:
        raise ValueError("Not enough independent groups to split")
    random.Random(args.seed + 1).shuffle(groups)
    validation, training = [], []
    target_validation = round(len(records) / 6)
    for group in groups:
        if len(validation) < target_validation and len(validation) + len(group) < len(records):
            validation.extend(group)
        else:
            training.extend(group)
    assert training and validation
    assert not set(training) & set(validation)
    assert not (set(training) | set(validation)) & excluded
    partition = {index: "train" for index in training} | {index: "validation" for index in validation}
    for field in ("rgb_sha256", "dhash"):
        locations = defaultdict(set)
        for record in records:
            locations[record[field]].add(partition[record["index"]])
        assert all(len(value) == 1 for value in locations.values())
    report = {
        "database": str(Path(args.database).resolve()), "database_samples": count,
        "seed": args.seed, "audited_samples": len(records),
        "excluded_overfit_indices": sorted(excluded),
        "train_samples": len(training), "validation_samples": len(validation),
        "groups": len(groups), "multi_sample_groups": sum(len(group) > 1 for group in groups),
        "largest_group": max(map(len, groups)),
        "cross_split_exact_rgb_matches": 0, "cross_split_exact_dhash_matches": 0,
        "limitation": "PROVISIONAL development split: source-document IDs unavailable in image/label keys. Exact RGB and exact dHash grouping cannot exclude related crops, different edits, or all near-duplicates. No source-disjoint generalization claim. Audit covers only selected candidates.",
    }
    for name in ("train", "validation"):
        fractions = [r["positive_fraction"] for r in records if partition[r["index"]] == name]
        report[name + "_mask_fraction"] = {
            "min": min(fractions), "median": float(np.median(fractions)),
            "mean": float(np.mean(fractions)), "max": max(fractions),
            "empty_masks": sum(value == 0 for value in fractions),
        }
    output.mkdir(parents=True, exist_ok=True)
    for filename, value in (("train_indices.json", sorted(training)),
                            ("validation_indices.json", sorted(validation)),
                            ("audit_records.json", records), ("report.json", report)):
        (output / filename).write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved manifests: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
