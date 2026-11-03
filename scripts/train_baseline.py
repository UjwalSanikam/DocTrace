"""Fresh SmallUNet baseline on provisional train/validation manifests."""
import argparse
import csv
from hashlib import sha256
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
from torch.utils.data import DataLoader, Subset
from doctrace.dataset import DocTamperDataset
from doctrace.model import SmallUNet
from overfit_small import loss_fn, save_preview


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def metric_rows(probabilities, masks, indices):
    pred, truth = probabilities >= 0.5, masks.bool()
    axes = (1, 2, 3)
    tp = (pred & truth).sum(axes).cpu().tolist()
    fp = (pred & ~truth).sum(axes).cpu().tolist()
    fn = (~pred & truth).sum(axes).cpu().tolist()
    for index, a, b, c in zip(indices.tolist(), tp, fp, fn):
        yield {"index": index, "tp": a, "fp": b, "fn": c,
               "dice": 2 * a / max(2 * a + b + c, 1),
               "iou": a / max(a + b + c, 1)}


def summarize(rows, loss):
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    return {"loss": loss,
            "macro_dice": sum(row["dice"] for row in rows) / len(rows),
            "macro_iou": sum(row["iou"] for row in rows) / len(rows),
            "micro_precision": tp / max(tp + fp, 1),
            "micro_recall": tp / max(tp + fn, 1)}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    rows, previews = [], []
    total_loss, count = 0.0, 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite validation loss")
        total_loss += loss.item() * len(x)
        count += len(x)
        probabilities = logits.sigmoid()
        rows.extend(metric_rows(probabilities, y, batch["index"]))
        remaining = 8 - sum(len(part[0]) for part in previews)
        if remaining > 0:
            n = min(remaining, len(x))
            previews.append((batch["image"][:n], batch["mask"][:n],
                             probabilities[:n].cpu(), batch["index"][:n]))
    return summarize(rows, total_loss / count), rows, previews


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="outputs/split_v1")
    parser.add_argument("--output", default="outputs/baseline_v1")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smoke", action="store_true", help="One epoch: 64 train / 32 validation; not a baseline result")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.lr <= 0:
        parser.error("Invalid training settings")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for this run")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Output folder is not empty. Choose a new --output; existing runs are preserved.")
    split = Path(args.split)
    report = json.loads((split / "report.json").read_text(encoding="utf-8"))
    train_ids = json.loads((split / "train_indices.json").read_text(encoding="utf-8"))
    val_ids = json.loads((split / "validation_indices.json").read_text(encoding="utf-8"))
    if not train_ids or not val_ids or len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids):
        raise ValueError("Empty or repeated split indices")
    if set(train_ids) & set(val_ids):
        raise ValueError("Train/validation overlap")
    if (set(train_ids) | set(val_ids)) & set(report["excluded_overfit_indices"]):
        raise ValueError("Overfit samples entered development split")
    if args.smoke:
        train_ids, val_ids = train_ids[:64], val_ids[:32]
        args.epochs = 1
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    dataset = DocTamperDataset(report["database"])
    if min(train_ids + val_ids) < 0 or max(train_ids + val_ids) >= len(dataset):
        raise ValueError("Split indices out of range")
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"train_samples": len(train_ids), "validation_samples": len(val_ids),
        "database": report["database"], "precision": "float32", "initialization": "random; no overfit weights",
        "architecture": "SmallUNet(base=16)", "loss": "BCE + per-image soft Dice", "augmentation": "none",
        "threshold": 0.5, "torch_version": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
        "split_limitation": report["limitation"], "authentic_document_evaluation": False,
        "manifest_hashes": {name: sha256((split / name).read_bytes()).hexdigest()
                            for name in ("train_indices.json", "validation_indices.json", "report.json")}}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    # Keep the exact active manifests with each experiment, including smoke subsets.
    (output / "train_indices.json").write_text(json.dumps(train_ids), encoding="utf-8")
    (output / "validation_indices.json").write_text(json.dumps(val_ids), encoding="utf-8")
    common = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                  persistent_workers=args.workers > 0, worker_init_fn=seed_worker)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(Subset(dataset, train_ids), shuffle=True, generator=generator, **common)
    val_loader = DataLoader(Subset(dataset, val_ids), shuffle=False, **common)
    model = SmallUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    print(f"{'SMOKE CHECK' if args.smoke else 'BASELINE'} | fresh weights | float32 | train={len(train_ids)} val={len(val_ids)} epochs={args.epochs}", flush=True)
    print("Provisional split; no claim of source-document separation. FCD is not accessed.", flush=True)
    best = -1.0
    history = []
    started = time.perf_counter()
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = None
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss, seen = 0.0, 0
            for step, batch in enumerate(train_loader, 1):
                x = batch["image"].to(device, non_blocking=True)
                y = batch["mask"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(x), y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss: epoch {epoch}, batch {step}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                running_loss += loss.item() * len(x)
                seen += len(x)
                if step % 100 == 0 or step == len(train_loader):
                    print(f"epoch {epoch}/{args.epochs} batch {step}/{len(train_loader)} train_loss={running_loss / seen:.4f}", flush=True)
            metrics, rows, previews = validate(model, val_loader, device)
            row = {"epoch": epoch, "train_loss": running_loss / seen,
                   **{"val_" + key: value for key, value in metrics.items()},
                   "seconds": time.perf_counter() - started}
            history.append(row)
            if writer is None:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            file.flush()
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "epoch": epoch, "config": config, "metrics": metrics}
            torch.save(state, output / "last.pt")
            write_rows(output / "last_validation_per_image.csv", sorted(rows, key=lambda row: row["dice"]))
            if metrics["macro_dice"] > best:
                best = metrics["macro_dice"]
                torch.save(state, output / "best.pt")
                write_rows(output / "best_validation_per_image.csv", sorted(rows, key=lambda row: row["dice"]))
                parts = [torch.cat([part[i] for part in previews]) for i in range(4)]
                save_preview(parts[0], parts[1], parts[2], parts[3].tolist(), output / "best_validation_predictions.png")
            print(f"epoch={epoch} val_loss={metrics['loss']:.4f} val_Dice={metrics['macro_dice']:.4f} val_IoU={metrics['macro_iou']:.4f} precision={metrics['micro_precision']:.4f} recall={metrics['micro_recall']:.4f}", flush=True)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for key in ("train_loss", "val_loss"):
                axes[0].plot([r["epoch"] for r in history], [r[key] for r in history], label=key)
            for key in ("val_macro_dice", "val_macro_iou"):
                axes[1].plot([r["epoch"] for r in history], [r[key] for r in history], label=key)
            axes[1].set_ylim(0, 1)
            for ax in axes:
                ax.legend()
                ax.set_xlabel("Epoch")
            fig.tight_layout()
            fig.savefig(output / "learning_curve.png", dpi=150)
            plt.close(fig)
    dataset.close()
    print(f"Finished. Best provisional validation Dice={best:.4f}. Results: {output.resolve()}", flush=True)
    print("Authentic-document false alarms and held-out cross-domain performance are not evaluated here.", flush=True)


if __name__ == "__main__":
    main()
