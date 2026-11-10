"""Frozen-checkpoint perturbation study; no training or threshold selection."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from doctrace.dataset import DocTamperDataset
from doctrace.degradations import CONDITIONS, degrade, to_tensor
from doctrace.model import build_model
from doctrace.synthetic import make_document
from train_baseline import metric_rows, summarize, write_rows


class PerturbedDataset(Dataset):
    def __init__(self, database, indices, condition, synthetic=False):
        self.base = None if synthetic else DocTamperDataset(database)
        self.indices, self.condition = indices, condition

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        if self.base is None:
            image = make_document(index)
            mask = torch.zeros(1, image.height, image.width)
        else:
            sample = self.base[index]
            image = Image.fromarray((sample["image"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
            mask = sample["mask"]
        return {"image": to_tensor(degrade(image, self.condition)), "mask": mask, "index": index}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="outputs/resnet18_continue_v1")
    parser.add_argument("--checkpoint", choices=["best.pt", "last.pt"], default="best.pt")
    parser.add_argument("--suite", choices=["validation", "fcd", "synthetic"], default="validation")
    parser.add_argument("--database", default="data/raw/DocTamperV1-FCD")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    output, run = Path(args.output), Path(args.run)
    if output.exists() and any(output.iterdir()):
        parser.error("Output is not empty; choose a new directory")
    if args.workers < 0:
        parser.error("workers must be nonnegative")
    checkpoint = torch.load(run / args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if args.suite == "validation":
        indices = json.loads((run / "validation_indices.json").read_text())
        database = config["database"]
    elif args.suite == "fcd":
        database = args.database
        indices = list(range(len(DocTamperDataset(database))))
    else:
        database = None
        indices = list(range(1000, 1064))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config.get("model", "small")).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for condition in dict.fromkeys(args.conditions):
        dataset = PerturbedDataset(database, indices, condition, args.suite == "synthetic")
        loader = DataLoader(dataset, batch_size=config["batch_size"], num_workers=args.workers, shuffle=False)
        rows, fractions = [], []
        with torch.inference_mode():
            for step, batch in enumerate(loader):
                probabilities = model(batch["image"].to(device)).sigmoid()
                if not torch.isfinite(probabilities).all():
                    raise RuntimeError("Nonfinite predictions")
                rows.extend(metric_rows(probabilities, batch["mask"].to(device), batch["index"]))
                fractions.extend((probabilities >= 0.5).float().mean((1, 2, 3)).cpu().tolist())
                if (step + 1) % 100 == 0:
                    print(f"{args.suite} / {condition}: {len(rows)}/{len(indices)}", flush=True)
        if dataset.base is not None:
            dataset.base.close()
        write_rows(output / f"{condition}_per_image.csv", rows)
        if args.suite == "synthetic":
            metrics = {"count": len(rows), "mean_false_positive_pixel_fraction": float(np.mean(fractions)),
                       "median_false_positive_pixel_fraction": float(np.median(fractions)),
                       "p95_false_positive_pixel_fraction": float(np.percentile(fractions, 95)),
                       "documents_with_any_predicted_pixel": sum(f > 0 for f in fractions),
                       "documents_with_at_least_0.1_percent_predicted_area": sum(f >= .001 for f in fractions)}
        else:
            metrics = summarize(rows, None)
            metrics.pop("loss")
            metrics.update(count=len(rows), zero_overlap_images=sum(r["tp"] == 0 for r in rows))
        summaries[condition] = metrics
        print(f"{condition}: {json.dumps(metrics)}", flush=True)
        del loader, dataset
    if "clean" in summaries and args.suite != "synthetic":
        for metrics in summaries.values():
            metrics["dice_change_from_clean"] = metrics["macro_dice"] - summaries["clean"]["macro_dice"]
    report = {"suite": args.suite, "checkpoint_epoch": checkpoint["epoch"],
              "checkpoint_sha256": sha256((run / args.checkpoint).read_bytes()).hexdigest(),
              "index_manifest_sha256": sha256(json.dumps(indices).encode()).hexdigest(),
              "threshold": .5, "batch_size": config["batch_size"], "conditions": summaries,
              "protocol": "Frozen weights, native 512x512 images, no tuning. JPEG uses Pillow quality and 4:2:0 subsampling; resize is bicubic down/up to original grid; blur radius 0.7. Masks unchanged.",
              "limitations": config["split_limitation"],
              "suite_note": {"validation": "Development robustness study on the same provisional validation set used for checkpoint selection.",
                             "fcd": "Separate FCD evaluation after freezing checkpoint and threshold. Some samples were inspected during loader development; not a pristine blind test. No claim of source separation from training.",
                             "synthetic": "64 procedural, unedited documents. Pixel false positives only; not representative of real authentic documents. No document-authenticity accuracy is claimed."}[args.suite]}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
