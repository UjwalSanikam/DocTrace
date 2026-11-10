# DocTrace v1 model card

## Purpose

Research and education: localize possible text edits in document images and study sensitivity to JPEG recompression. This system produces a spatial model response, not an authenticity verdict, fraud probability, or evidentiary conclusion.

## Model and training

ResNet-18 ImageNet encoder with a U-Net-style decoder and full-resolution skip connection. Encoder weights and BatchNorm affine parameters train; BatchNorm running statistics remain frozen. Decoder blocks use GroupNorm. Inputs are RGB in [0,1], with ImageNet normalization inside the model. Outputs are per-pixel logits converted with sigmoid.

Training uses BCE plus per-image soft Dice, AdamW (learning rate 0.0002, weight decay 0), batch size 4, float32, gradient clipping at norm 5, and finite-value checks. The matched JPEG experiment resumes the same epoch-8 model and optimizer into two runs of 5,000 updates. See [protocol](EXPERIMENT_PROTOCOL.md) and the [release selection](../experiments/release_v1/selection.json).

## Data and evaluation

The training/development data is a selected 12,000-image subset of DocTamper's 120,000-image training database, divided into 10,000 training and 2,000 provisional validation images. Exact RGB and dHash grouping reduces exact duplicate leakage but does not establish source-document separation. Image/mask pairs remain on the native 512×512 grid; label values 0/1 and 0/255 are binarized consistently.

Candidate selection uses a fixed development score averaging macro Dice across clean, JPEG90, JPEG70, and JPEG50 at threshold 0.5. FCD is evaluated after selection; earlier loader inspection means it is not a pristine blind test. Additional resize/blur conditions are diagnostics. Synthetic negative checks use 64 procedural unedited images and report pixel false positives, not document-level authenticity accuracy.

All measured results and hashes are in [release evaluation](../experiments/release_v1/RESULTS.md). The model artifact itself remains local. Its source training-checkpoint hash and exported-file hash are listed in [the manifest](../experiments/release_v1/model_manifest.json).

## Inference and interface

Single PNG/JPEG/WebP images only; PDFs and animations are excluded. EXIF orientation is applied and transparent pixels are composited on white. Larger images use 512-pixel tiles at stride 384 with raised-cosine overlap weighting. Small images use edge padding; predictions are cropped back to the original size. No page-wide resizing is performed. Full-page tiling can produce context and boundary effects not measured by the fixed-patch benchmark.

The optional sensitivity probe recompresses the entire input image at JPEG quality 70 and repeats inference. Its maps display absolute probability changes; the summary reports how many pixels cross the selected threshold. Instability can occur in both correct and incorrect predictions, so it is not a calibrated uncertainty estimator.

The UI quantizes probabilities to 8 bits for display and interactive export (maximum rounding error 0.5/255). Benchmark evaluation and the CLI use float probabilities. The threshold slider is an inspection tool; results at user-chosen thresholds have not been validated.

## Limitations and failure modes

- Very small edits and partial border regions can be missed.
- Ordinary text, high-contrast glyphs, and document style can trigger false positives.
- JPEG processing, blur, and resizing may remove or alter useful cues.
- Source overlap across data partitions cannot be ruled out.
- Training uses tampered documents; real authentic-document false-alarm rates are unknown.
- Synthetic negatives are narrow and cannot support real-world specificity claims.
- Physical recapture, AIGC edits, multilingual subgroup parity, and distribution shifts are not validated.
- One seed and one development split do not establish statistically robust superiority.

## Operation and privacy

The demo server binds only to 127.0.0.1. It accepts local-origin image requests, serializes model inference, and does not write uploads to disk. There are no external assets or analytics in the UI. User-triggered downloads are saved by the browser. CLI outputs are explicitly written to the chosen directory. The standard-library server is intended for a single-user local demo, not an internet-facing service.

Use only trusted local checkpoints. Dataset and model redistribution rights are not assumed. The public repository includes aggregate results and synthetic examples, not DocTamper images or trained weights.
