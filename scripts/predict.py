"""Analyze a local image and export an overlay, mask, probabilities, ELA, and report."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from doctrace.inference import Predictor, decode_image, png_bytes, overlay, MAX_BYTES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", default="outputs/release_v1/model.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--no-probe", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output is not empty. Choose a new directory.")
    if not args.image.is_file() or not Path(args.checkpoint).is_file():
        parser.error("Image or checkpoint is missing. Check the input path and --checkpoint.")
    if args.image.stat().st_size > MAX_BYTES:
        parser.error("Input exceeds 24 MB")
    image = decode_image(args.image.read_bytes())
    probability, alternative, worst_case, ela_map, report = Predictor(args.checkpoint, args.device).analyze(
        image, not args.no_probe, args.threshold
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "probabilities.npy", probability, allow_pickle=False)
    (args.output / "mask.png").write_bytes(png_bytes((probability >= args.threshold).astype(np.uint8) * 255))
    (args.output / "overlay.png").write_bytes(png_bytes(overlay(image, probability, args.threshold)))
    # ELA: amplified JPEG residual (quality 90, amplify 15, clipped to [0, 255]).
    (args.output / "ela_residual.png").write_bytes(png_bytes(ela_map))
    (args.output / "ela.png").write_bytes(png_bytes(ela_map))  # backward compatibility alias
    if alternative is not None:
        (args.output / "sensitivity.png").write_bytes(
            png_bytes((np.abs(probability - alternative) * 255).round().astype(np.uint8))
        )
    if worst_case is not None:
        (args.output / "sensitivity_worst_case.png").write_bytes(
            png_bytes((worst_case * 255).round().astype(np.uint8))
        )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
