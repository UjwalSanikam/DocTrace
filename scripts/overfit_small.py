"""Memorize a fixed training subset. Scores here are NOT validation results."""
import argparse
import csv
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from doctrace.dataset import DocTamperDataset
from doctrace.model import SmallUNet


def loss_fn(logits, masks):
    # Per-image Dice gives small positive regions a voice in the objective.
    probabilities = logits.float().sigmoid()
    numerator = 2 * (probabilities * masks).sum((1, 2, 3))
    denominator = probabilities.sum((1, 2, 3)) + masks.sum((1, 2, 3))
    dice_loss = 1 - ((numerator + 1e-6) / (denominator + 1e-6)).mean()
    return F.binary_cross_entropy_with_logits(logits.float(), masks) + dice_loss


@torch.no_grad()
def evaluate(model, images, masks, batch_size, device):
    model.eval()
    outputs = []
    loss_sum = 0.0
    for start in range(0, len(images), batch_size):
        x = images[start:start + batch_size].to(device)
        y = masks[start:start + batch_size].to(device)
        logits = model(x)
        loss_sum += loss_fn(logits, y).item() * len(x)
        outputs.append(logits.sigmoid().cpu())
    probabilities = torch.cat(outputs)
    pred, truth = probabilities >= 0.5, masks.bool()
    dims = (1, 2, 3)
    tp = (pred & truth).sum(dims).double()
    fp = (pred & ~truth).sum(dims).double()
    fn = (~pred & truth).sum(dims).double()
    dice = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    iou = tp / (tp + fp + fn).clamp_min(1)
    metrics = {
        "loss": loss_sum / len(images),
        "macro_dice": dice.mean().item(),
        "macro_iou": iou.mean().item(),
        "micro_precision": (tp.sum() / (tp.sum() + fp.sum()).clamp_min(1)).item(),
        "micro_recall": (tp.sum() / (tp.sum() + fn.sum()).clamp_min(1)).item(),
        "predicted_positive_fraction": pred.float().mean().item(),
    }
    return metrics, probabilities


def save_preview(images, masks, probabilities, indices, path):
    count = len(images)
    fig, axes = plt.subplots(count, 4, figsize=(12, count * 2.8), squeeze=False)
    for row in range(count):
        axes[row, 0].imshow(images[row].permute(1, 2, 0).numpy())
        axes[row, 1].imshow(masks[row, 0], cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(probabilities[row, 0], cmap="magma", vmin=0, vmax=1)
        axes[row, 3].imshow(probabilities[row, 0] >= 0.5, cmap="gray", vmin=0, vmax=1)
        for ax, title in zip(axes[row], [f"Training image {indices[row]}", "Ground truth", "Probability (0-1)", "Prediction >= 0.5"]):
            ax.set_title(title)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_per_image(masks, probabilities, indices, path):
    pred, truth = probabilities >= 0.5, masks.bool()
    dims = (1, 2, 3)
    tp = (pred & truth).sum(dims).double()
    fp = (pred & ~truth).sum(dims).double()
    fn = (~pred & truth).sum(dims).double()
    rows = []
    for i, index in enumerate(indices):
        rows.append({
            "index": index,
            "dice": (2 * tp[i] / (2 * tp[i] + fp[i] + fn[i]).clamp_min(1)).item(),
            "iou": (tp[i] / (tp[i] + fp[i] + fn[i]).clamp_min(1)).item(),
            "precision": (tp[i] / (tp[i] + fp[i]).clamp_min(1)).item(),
            "recall": (tp[i] / (tp[i] + fn[i]).clamp_min(1)).item(),
            "true_positive_pixels": int(tp[i]),
            "false_positive_pixels": int(fp[i]),
            "false_negative_pixels": int(fn[i]),
            "target_fraction": masks[i].mean().item(),
            "predicted_fraction": pred[i].float().mean().item(),
        })
    rows.sort(key=lambda row: row["dice"])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="data/raw/DocTamperV1-TrainingSet")
    parser.add_argument("--output", default="outputs/overfit_16")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--cpu", action="store_true", help="Only for a small implementation check")
    parser.add_argument("--checkpoint", help="Warm-start model weights and exact sample indices; reset AdamW state")
    args = parser.parse_args()
    if args.lr <= 0:
        parser.error("Learning rate must be positive")
    checkpoint = None
    parent_step = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        parent_step = int(checkpoint["step"])
        source_database = Path(checkpoint["config"]["database"]).resolve()
        if source_database != Path(args.database).resolve():
            parser.error("Checkpoint database differs from --database; use the original dataset")
    if min(args.samples, args.steps, args.batch_size, args.log_every) < 1:
        parser.error("Counts must be positive")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Output folder is not empty; choose a new --output to preserve existing results")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use --cpu only for a tiny check.")
    device = torch.device("cpu" if args.cpu else "cuda")
    dataset = DocTamperDataset(args.database)
    if args.samples > len(dataset):
        raise ValueError("Requested more samples than available")
    indices = (list(checkpoint["config"]["indices"]) if checkpoint is not None
               else random.Random(args.seed).sample(range(len(dataset)), args.samples))
    if len(indices) != args.samples or len(set(indices)) != len(indices):
        dataset.close()
        parser.error("--samples must match the checkpoint subset length (with unique indices)")
    if any(index < 0 or index >= len(dataset) for index in indices):
        dataset.close()
        parser.error("Checkpoint contains invalid sample indices")
    try:
        records = [dataset[index] for index in indices]
    finally:
        dataset.close()
    images = torch.stack([record["image"] for record in records])
    masks = torch.stack([record["mask"] for record in records])
    if images.shape[1:] != (3, 512, 512) or masks.shape[1:] != (1, 512, 512):
        raise ValueError("Expected original 512x512 images and masks")
    if not (masks.flatten(1).sum(1) > 0).all():
        raise ValueError("Selected sample has no positives; inspect empty-mask metric conventions")
    config = vars(args) | {"indices": indices, "torch_version": str(torch.__version__),
                            "device": str(device), "cuda_runtime": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                            "precision": "float32", "parent_step": parent_step,
                            "optimizer_state": "new AdamW; model-weight warm start" if checkpoint else "new AdamW",
                            "purpose": "training-subset memorization only; no generalization claim"}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Device: {device} | samples: {len(images)} | float32 | no augmentation", flush=True)
    print(f"Indices: {indices}", flush=True)
    model = SmallUNet().to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
        print(f"Loaded weights from step {parent_step}; new AdamW optimizer, lr={args.lr}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    history = []
    best = -1.0
    started = time.perf_counter()
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = None
        for step in range(args.steps + 1):
            if step > 0:
                model.train()
                chosen = torch.randperm(len(images))[:args.batch_size]
                x, y = images[chosen].to(device), masks[chosen].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = loss_fn(logits, y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at step {step}; sample indices: {[indices[i] for i in chosen.tolist()]}")
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise RuntimeError(f"Nonfinite gradient at step {step} in {name}; sample indices: {[indices[i] for i in chosen.tolist()]}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
            if step % args.log_every == 0 or step == args.steps:
                metrics, probabilities = evaluate(model, images, masks, args.batch_size, device)
                row = {"step": step, "total_step": parent_step + step, **metrics, "seconds": time.perf_counter() - started}
                history.append(row)
                if writer is None:
                    writer = csv.DictWriter(file, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                file.flush()
                print(f"step={step:4d} loss={metrics['loss']:.4f} Dice={metrics['macro_dice']:.4f} IoU={metrics['macro_iou']:.4f} precision={metrics['micro_precision']:.4f} recall={metrics['micro_recall']:.4f}", flush=True)
                if metrics["macro_dice"] > best:
                    best = metrics["macro_dice"]
                    torch.save({"model": model.state_dict(), "config": config, "step": parent_step + step, "metrics": metrics}, output / "best.pt")
                    save_preview(images, masks, probabilities, indices, output / "best_predictions.png")
                    best_rows = save_per_image(masks, probabilities, indices, output / "best_per_image.csv")
    torch.save({"model": model.state_dict(), "config": config, "step": parent_step + args.steps, "metrics": metrics}, output / "last.pt")
    save_preview(images, masks, probabilities, indices, output / "last_predictions.png")
    save_per_image(masks, probabilities, indices, output / "last_per_image.csv")
    print("Lowest Dice samples at the best checkpoint:", flush=True)
    for row in best_rows[:5]:
        print(f"  index={row['index']} Dice={row['dice']:.4f} recall={row['recall']:.4f}", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    steps = [row["step"] for row in history]
    axes[0].plot(steps, [row["loss"] for row in history])
    axes[0].set_title("Training-subset loss")
    for key in ["macro_dice", "macro_iou"]:
        axes[1].plot(steps, [row[key] for row in history], label=key)
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel("Optimizer steps")
    fig.tight_layout()
    fig.savefig(output / "learning_curve.png", dpi=150)
    plt.close(fig)
    print(f"Finished. Best training-subset Dice: {best:.4f}. Results: {output.resolve()}", flush=True)
    print("These scores measure memorization, not performance on unseen documents.", flush=True)


if __name__ == "__main__":
    main()
