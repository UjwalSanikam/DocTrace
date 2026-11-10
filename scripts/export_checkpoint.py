"""Export inference-only weights from a trusted local training checkpoint."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/release_v1"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output is not empty; choose a new directory")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    source_hash = sha256(args.checkpoint.read_bytes()).hexdigest()
    # Strip optimizer/RNG state and machine-specific paths. Preserve model loading contract.
    config = {key: state["config"][key] for key in ("model", "architecture", "normalization", "augmentation", "precision", "threshold", "split_limitation") if key in state["config"]}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "model.pt"
    torch.save({"model":state["model"], "config":config, "epoch":state["epoch"], "source_checkpoint_sha256":source_hash}, path)
    manifest = {"filename":"model.pt", "sha256":sha256(path.read_bytes()).hexdigest(), "source_checkpoint_sha256":source_hash, "epoch":state["epoch"], "bytes":path.stat().st_size, "config":config,
                "note":"Local inference artifact. Redistribution rights are not assumed; the weights are excluded from Git."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
