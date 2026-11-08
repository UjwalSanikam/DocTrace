"""Compare a seeded training subset with saved full-validation checkpoint metrics."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from torch.utils.data import DataLoader, Subset
from doctrace.dataset import DocTamperDataset
from doctrace.model import build_model
from train_baseline import validate, write_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="outputs/baseline_v1")
    parser.add_argument("--output", default="outputs/baseline_gap_v1")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run, output = Path(args.run), Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Output folder is not empty; choose another --output")
    if args.samples < 1 or args.workers < 0:
        parser.error("Invalid sample count/workers")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    train_ids = json.loads((run / "train_indices.json").read_text(encoding="utf-8"))
    val_ids = json.loads((run / "validation_indices.json").read_text(encoding="utf-8"))
    if args.samples > len(train_ids) or set(train_ids) & set(val_ids):
        parser.error("Invalid subset size or overlapping manifests")
    indices = random.Random(args.seed).sample(train_ids, args.samples)
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config.get("model", "small"), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    dataset = DocTamperDataset(config["database"])
    loader = DataLoader(Subset(dataset, indices), batch_size=config["batch_size"],
                        num_workers=args.workers, pin_memory=device.type == "cuda", shuffle=False)
    print(f"Evaluating {len(indices)} fixed training images at checkpoint epoch {checkpoint['epoch']}; no training updates", flush=True)
    try:
        metrics, rows, _ = validate(model, loader, device)
    finally:
        dataset.close()
    report = {"checkpoint_epoch": checkpoint["epoch"],
              "checkpoint_sha256": sha256((run / "best.pt").read_bytes()).hexdigest(),
              "seed": args.seed, "training_subset_count": len(indices),
              "training_subset": metrics,
              "saved_validation_count": len(val_ids), "saved_validation": checkpoint["metrics"],
              "train_minus_validation_dice": metrics["macro_dice"] - checkpoint["metrics"]["macro_dice"],
              "note": "Training metrics are a seeded subset estimate; validation metrics reuse the checkpoint evaluation. Both use eval mode and threshold 0.5. No tuning or FCD access.",
              "split_limitation": config["split_limitation"]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "train_indices.json").write_text(json.dumps(indices), encoding="utf-8")
    write_rows(output / "training_per_image.csv", sorted(rows, key=lambda row: row["dice"]))
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
