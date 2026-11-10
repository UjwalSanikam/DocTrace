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
    if not args.checkpoint.is_file():
        parser.error(f"Source checkpoint does not exist: {args.checkpoint}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output is not empty; choose a new directory")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    for required_key in ("model", "config", "epoch"):
        if required_key not in state:
            raise ValueError(f"Checkpoint {args.checkpoint} missing required key: {required_key}")

    source_hash = sha256(args.checkpoint.read_bytes()).hexdigest()
    # Strip optimizer/RNG state and machine-specific paths. Preserve model loading contract.
    config = {
        key: state["config"][key]
        for key in ("model", "architecture", "normalization", "augmentation", "precision", "threshold", "split_limitation")
        if key in state["config"]
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "model.pt"
    torch.save(
        {
            "model": state["model"],
            "config": config,
            "epoch": state["epoch"],
            "source_checkpoint_sha256": source_hash,
        },
        path,
    )

    if not path.is_file():
        raise RuntimeError(f"Exported model file was not written: {path}")

    written_hash = sha256(path.read_bytes()).hexdigest()
    manifest = {
        "filename": "model.pt",
        "sha256": written_hash,
        "source_checkpoint_sha256": source_hash,
        "epoch": state["epoch"],
        "bytes": path.stat().st_size,
        "config": config,
        "note": "Local inference artifact. Redistribution rights are not assumed; the weights are excluded from Git.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
