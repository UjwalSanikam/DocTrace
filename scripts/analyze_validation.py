"""Analyze saved validation scores; infer only selected visualization examples."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from doctrace.dataset import DocTamperDataset
from doctrace.model import build_model


def aggregate(rows):
    if not rows:
        return {"count": 0, "macro_dice": None, "macro_iou": None,
                "micro_precision": None, "micro_recall": None}
    tp, fp, fn = [sum(row[key] for row in rows) for key in ("tp", "fp", "fn")]
    return {"count": len(rows),
            "macro_dice": sum(row["dice"] for row in rows) / len(rows),
            "macro_iou": sum(row["iou"] for row in rows) / len(rows),
            "micro_precision": tp / max(tp + fp, 1),
            "micro_recall": tp / max(tp + fn, 1)}


def area_bin(fraction):
    if fraction < 0.001:
        return "under 0.1%"
    if fraction < 0.01:
        return "0.1% to under 1%"
    if fraction < 0.05:
        return "1% to under 5%"
    return "5% or more"


@torch.no_grad()
def preview(model, dataset, rows, device, path, validation_ids, batch_size, comparisons):
    fig, axes = plt.subplots(len(rows), 5, figsize=(16, 3.2 * len(rows)), squeeze=False)
    positions = {index: pos for pos, index in enumerate(validation_ids)}
    for position, row in enumerate(rows):
        sample = dataset[row["index"]]
        image = sample["image"]
        target = sample["mask"][0].numpy() > 0
        # Match the original validation batch composition and shape.
        original_position = positions[row["index"]]
        start = (original_position // batch_size) * batch_size
        context_ids = validation_ids[start:start + batch_size]
        context = torch.stack([dataset[index]["image"] for index in context_ids])
        probability = model(context.to(device)).sigmoid()[original_position - start, 0].cpu().numpy()
        pred = probability >= 0.5
        a = int((pred & target).sum())
        b = int((pred & ~target).sum())
        c = int((~pred & target).sum())
        current_dice = 2 * a / max(2 * a + b + c, 1)
        saved_counts = [row["tp"], row["fp"], row["fn"]]
        current_counts = [a, b, c]
        comparison = {"index": row["index"], "saved_counts_tp_fp_fn": saved_counts,
                      "preview_counts_tp_fp_fn": current_counts,
                      "saved_dice": row["dice"], "preview_dice": current_dice}
        comparisons.append(comparison)
        if a + c != row["tp"] + row["fn"]:
            raise RuntimeError(f"Ground-truth pixel count changed at {row['index']}: {comparison}")
        if saved_counts != current_counts:
            max_delta = max(abs(old - new) for old, new in zip(saved_counts, current_counts))
            tolerance = max(8, round(0.001 * sum(saved_counts)))
            if max_delta > tolerance or abs(current_dice - row["dice"]) > 0.005:
                raise RuntimeError(f"Material prediction difference after matching validation batch at index {row['index']}: {comparison}. Do not interpret the preview until reevaluation resolves this difference.")
            print(f"Minor numerical difference at index {row['index']}: saved={saved_counts}, preview={current_counts}", flush=True)
        error = np.zeros((*target.shape, 3), dtype=np.uint8)
        error[pred & target] = [0, 200, 0]
        error[pred & ~target] = [255, 0, 0]
        error[~pred & target] = [0, 120, 255]
        pictures = [image.permute(1, 2, 0).numpy(), target, probability, pred, error]
        titles = [f"Index {row['index']} / preview Dice {current_dice:.3f}",
                  f"Target ({row['target_fraction']:.2%})", "Probability", "Prediction >= 0.5",
                  "Green: TP / Red: FP / Blue: FN"]
        for ax, picture, title in zip(axes[position], pictures, titles):
            if picture.ndim == 2:
                ax.imshow(picture, cmap="magma" if title == "Probability" else "gray", vmin=0, vmax=1)
            else:
                ax.imshow(picture)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="outputs/baseline_v1")
    parser.add_argument("--output", default="outputs/diagnostics_v1")
    args = parser.parse_args()
    run, output = Path(args.run), Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Output is not empty; choose a new --output")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if config.get("smoke"):
        parser.error("Use a completed baseline, not a smoke run")
    expected = json.loads((run / "validation_indices.json").read_text(encoding="utf-8"))
    with (run / "best_validation_per_image.csv").open(newline="", encoding="utf-8") as file:
        rows = []
        for source in csv.DictReader(file):
            row = {key: int(source[key]) for key in ("index", "tp", "fp", "fn")}
            row.update({key: float(source[key]) for key in ("dice", "iou")})
            row["target_fraction"] = (row["tp"] + row["fn"]) / (512 * 512)
            row["predicted_fraction"] = (row["tp"] + row["fp"]) / (512 * 512)
            row["recall"] = row["tp"] / max(row["tp"] + row["fn"], 1)
            row["area_bin"] = area_bin(row["target_fraction"])
            rows.append(row)
    if len(rows) != len(expected) or set(row["index"] for row in rows) != set(expected):
        raise ValueError("Saved validation rows do not match the active manifest")
    if any(row["tp"] + row["fn"] == 0 for row in rows):
        raise ValueError("Authentic images require separate metric handling")
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    overall = aggregate(rows)
    if abs(overall["macro_dice"] - checkpoint["metrics"]["macro_dice"]) > 1e-6:
        raise ValueError("Best checkpoint and saved validation CSV do not agree")
    order = sorted(rows, key=lambda row: (row["dice"], row["index"]))
    middle = max(0, len(order) // 2 - 2)
    groups = {"lowest_dice": order[:4], "median_dice": order[middle:middle+4],
              "highest_dice": order[-4:],
              "most_false_positive_pixels": sorted(rows, key=lambda row: (-row["fp"], row["index"]))[:4]}
    bins = ("under 0.1%", "0.1% to under 1%", "1% to under 5%", "5% or more")
    by_area = {name: aggregate([row for row in rows if row["area_bin"] == name]) for name in bins}
    report = {"checkpoint_epoch": checkpoint["epoch"], "checkpoint_sha256": sha256((run / "best.pt").read_bytes()).hexdigest(),
              "threshold": 0.5, "overall": overall,
              "no_predicted_positive_images": sum(row["tp"] + row["fp"] == 0 for row in rows),
              "zero_overlap_images": sum(row["tp"] == 0 for row in rows),
              "by_target_area": by_area,
              "preview_indices": {name: [row["index"] for row in group] for name, group in groups.items()},
              "limitations": config["split_limitation"],
              "note": "Reuses saved validation metrics; re-infers only preview images. Extremes deliberately selected for diagnosis, not representative performance. No authentic-document evaluation or FCD access."}
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config.get("model", "small"), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    comparisons = []
    dataset = DocTamperDataset(config["database"])
    try:
        for name, group in groups.items():
            preview(model, dataset, group, device, output / f"{name}.png", expected, int(config["batch_size"]), comparisons)
            print(f"Saved {name}.png", flush=True)
    finally:
        dataset.close()
    report["preview_consistency_checks"] = comparisons
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(order)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist([row["dice"] for row in rows], bins=np.linspace(0, 1, 21))
    axes[0].set(xlabel="Per-image Dice", ylabel="Validation images", title="Dice distribution")
    axes[1].bar(range(len(bins)), [by_area[name]["macro_dice"] or 0 for name in bins])
    axes[1].set_xticks(range(len(bins)), [f"{name}\n(n={by_area[name]['count']})" for name in bins], fontsize=8)
    axes[1].set(ylim=(0, 1), ylabel="Macro Dice", xlabel="Ground-truth tampered area", title="Localization by region size")
    fig.tight_layout()
    fig.savefig(output / "diagnostic_charts.png", dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Diagnostics saved: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
