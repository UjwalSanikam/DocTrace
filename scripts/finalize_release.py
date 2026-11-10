"""Evaluate fixed candidates, select on development robustness, and freeze a local release."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ["clean", "jpeg90", "jpeg70", "jpeg50", "resize75", "resize50", "blur07"]
PRIMARY = CONDITIONS[:4]
CANDIDATES = [
    ("epoch8", "outputs/resnet18_continue_v1", "best.pt", "outputs/robustness_v1", 8),
    ("clean_control", "outputs/clean_control_v1", "last.pt", "outputs/clean_control_robustness_v1", 10),
    ("jpeg_augmented", "outputs/jpeg_adapt_v1", "last.pt", "outputs/jpeg_robustness_v1", 10),
]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def execute(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def evaluate(run, checkpoint, output, suite="validation", conditions=CONDITIONS):
    path = ROOT / output / "summary.json"
    if not path.exists():
        execute("scripts/evaluate_robustness.py", "--run", run, "--checkpoint", checkpoint, "--output", output, "--suite", suite, "--conditions", *conditions)
    report = read(path)
    assert report["checkpoint_sha256"] == sha256((ROOT / run / checkpoint).read_bytes()).hexdigest()
    assert report["suite"] == suite and set(report["conditions"]) == set(conditions)
    return report


def main():
    reports, candidates = {}, []
    for name, run, checkpoint, output, epoch in CANDIDATES:
        state = torch.load(ROOT / run / checkpoint, map_location="cpu", weights_only=True)
        if state["epoch"] != epoch:
            raise RuntimeError(f"{name}: expected completed epoch {epoch}, got {state['epoch']}")
        del state
        report = evaluate(run, checkpoint, output)
        reports[name] = report
        score = float(np.mean([report["conditions"][condition]["macro_dice"] for condition in PRIMARY]))
        candidates.append({"name":name, "run":run, "checkpoint":checkpoint, "primary_mean_dice":score,
                           "checkpoint_sha256":report["checkpoint_sha256"], "epoch":epoch})
    selected = max(candidates, key=lambda candidate: candidate["primary_mean_dice"])
    selection = {"criterion":"Equal mean macro Dice on clean, JPEG90, JPEG70, JPEG50 development images; fixed threshold 0.5", "candidates":candidates, "selected":selected,
                 "note":"Single-seed development comparison. FCD and synthetic-negative results are not used for selection."}
    selection_dir = ROOT / "outputs/selection_v1"
    selection_dir.mkdir(parents=True, exist_ok=True)
    selection_path = selection_dir / "selection.json"
    if selection_path.exists() and read(selection_path) != selection:
        raise RuntimeError("Existing frozen selection differs; preserve it and investigate.")
    selection_path.write_text(json.dumps(selection, indent=2)+"\n", encoding="utf-8")
    print("FROZEN SELECTION: " + json.dumps(selected), flush=True)
    fcd = evaluate(selected["run"], selected["checkpoint"], "outputs/fcd_release_v1", suite="fcd")
    synthetic = evaluate(selected["run"], selected["checkpoint"], "outputs/synthetic_release_v1", suite="synthetic", conditions=["clean", "jpeg70"])
    artifact = ROOT / "outputs/release_v1/manifest.json"
    if not artifact.exists():
        execute("scripts/export_checkpoint.py", str(ROOT / selected["run"] / selected["checkpoint"]))
    manifest = read(artifact)
    assert manifest["source_checkpoint_sha256"] == selected["checkpoint_sha256"]
    target = ROOT / "experiments/release_v1"
    target.mkdir(parents=True, exist_ok=True)
    for name, data in (("selection", selection), ("validation", reports), ("fcd", fcd), ("synthetic_negatives", synthetic), ("model_manifest", manifest)):
        (target / f"{name}.json").write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
    for name, run, _, _, _ in CANDIDATES[1:]:
        experiment = ROOT / "experiments" / name
        experiment.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / run / "metrics.csv", experiment / "metrics.csv")
        config = read(ROOT / run / "config.json")
        config["database"] = "data/raw/DocTamperV1-TrainingSet"
        config["provenance_note"] = "Database path normalized for publication. Manifest hashes refer to original run files."
        (experiment / "config.json").write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(CONDITIONS))
    labels = ["Clean", "JPEG\n90", "JPEG\n70", "JPEG\n50", "Resize\n0.75x", "Resize\n0.5x", "Blur\n0.7"]
    for offset, (name, label, color) in enumerate((("epoch8", "Epoch 8", "#aab6ac"), ("clean_control", "Clean control, epoch 10", "#36746a"), ("jpeg_augmented", "JPEG trained, epoch 10", "#d66b47"))):
        axes[0].bar(x + (offset-1)*.24, [reports[name]["conditions"][condition]["macro_dice"] for condition in CONDITIONS], .24, label=label, color=color)
    axes[0].set_xticks(x, labels, fontsize=8)
    axes[0].set(title="Provisional validation: matched training budget", ylabel="Macro Dice", ylim=(0, 1))
    axes[0].legend(fontsize=8)
    axes[1].bar(x, [fcd["conditions"][condition]["macro_dice"] for condition in CONDITIONS], color="#36746a")
    axes[1].set_xticks(x, labels, fontsize=8)
    axes[1].set(title="FCD: frozen release checkpoint", ylabel="Macro Dice", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(target / "robustness.png", dpi=160)
    plt.close(fig)
    lines = ["# Release evaluation", "", "Threshold 0.5. Per-image macro Dice; 2,000 images per dataset. The development split is provisional and not guaranteed source-disjoint.", "", "| Condition | Epoch 8 | Clean control | JPEG trained | FCD release |", "|---|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        values = [reports[name]["conditions"][condition]["macro_dice"] for name in ("epoch8", "clean_control", "jpeg_augmented")]
        values.append(fcd["conditions"][condition]["macro_dice"])
        lines.append("| " + condition + " | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines.extend(["", f"Selected checkpoint: **{selected['name']}**, epoch {selected['epoch']}. Selection score: **{selected['primary_mean_dice']:.4f}** (equal mean of clean and three JPEG conditions).", "", "## Synthetic clean-document check", "", "These 64 procedural images contain no tampering; they do not represent real authentic documents.", "", "| Condition | Mean false-positive pixel fraction | Documents with >=0.1% flagged area |", "|---|---:|---:|"])
    for condition, metrics in synthetic["conditions"].items():
        lines.append(f"| {condition} | {metrics['mean_false_positive_pixel_fraction']:.2%} | {metrics['documents_with_at_least_0.1_percent_predicted_area']}/64 |")
    lines.extend(["", "Compression, resizing, and synthetic blur do not simulate all physical capture effects. No real authentic-document accuracy or calibrated fraud probability is claimed.", "", "![Robustness comparison](robustness.png)", ""])
    (target / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Release artifacts and aggregate report saved.", flush=True)


if __name__ == "__main__":
    main()
