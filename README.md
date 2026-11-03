# DocTrace

Document-image tampering localization with PyTorch. This project trains a segmentation model to highlight edited text regions and investigates robustness to image degradation.

**Status:** initial baseline complete. Robustness experiments, authentic-document evaluation, and an interactive demo are planned, not implemented. No new method or state-of-the-art performance is claimed.

## Current implementation

- Read-only DocTamper LMDB loader with Windows DataLoader worker support.
- Binary mask handling for both 0/1 and 0/255 labels at native 512 x 512 resolution.
- Small randomly initialized U-Net with GroupNorm.
- Float32 training with BCE plus per-image soft Dice loss and finite-gradient checks.
- Fixed-subset overfitting checks, checkpoint warm starts, and per-image diagnostics.
- Reproducible provisional development split with exact RGB and exact dHash grouping.
- Baseline training, validation metrics, checkpoint selection, and prediction previews.

## Initial results

| Experiment | Training samples | Evaluation | Dice | IoU |
|---|---:|---|---:|---:|
| Overfit check | 16 | Same 16 training images | 0.9165 | 0.8725 |
| Baseline, epoch 5 | 10,000 | 2,000 provisional validation images | 0.2434 | 0.1834 |

Dice and IoU are per-image macro averages at a fixed probability threshold of 0.5. The baseline uses five epochs, batch size 4, AdamW at 0.0002, no augmentation, and fresh weights. Overfit scores measure memorization only. Validation performance remains limited and is the starting point for further experiments.

### Evaluation limitations

- Source-document IDs are unavailable in the inspected LMDB keys. Grouping exact RGB duplicates and exact 64-bit dHash matches does not eliminate all related crops, alternate edits, or near-duplicates. Validation is not claimed to be source-disjoint.
- Split auditing covers 12,000 selected candidates from the training database. Sixteen previously inspected overfit samples are excluded by index, not by source identity.
- All audited masks contain tampered pixels. False alarms on authentic documents have not been evaluated.
- FCD is reserved for later evaluation and was not used to select this baseline checkpoint. Some FCD samples were inspected during loader development.
- Compression robustness, physical recapture robustness, and cross-domain benchmark performance have not yet been measured.
- Predicted regions are model outputs, not proof of authenticity or fraud.

## Setup (Windows PowerShell)

Tested with Python 3.13.14, PyTorch 2.14.1+cu130, and an RTX 5080 Laptop GPU (16 GB).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

`requirements-lock.txt` records the installed package versions for the development environment. Use the same extra index when installing its CUDA-tagged packages.

## Dataset

Obtain DocTamper separately through the [official repository](https://github.com/qcf-568/DocTamper), follow the authors' access process and usage terms, and place the files as follows:

```text
data/raw/
  DocTamperV1-TrainingSet/data.mdb
  DocTamperV1-FCD/data.mdb
```

Dataset files, document images, generated previews, and model weights are not included. Dataset access eligibility clarification was requested from the authors and remains pending. This code repository does not grant rights to the dataset.

## Run

Run commands from the repository root with the virtual environment active.

Check batches and Windows multiprocessing:

```powershell
python scripts/check_batch.py --workers 0
python scripts/check_batch.py --workers 2
```

Small training-subset memorization test:

```powershell
python scripts/overfit_small.py --samples 16 --output outputs/overfit_16
```

Use the committed provisional split to reproduce the baseline sample selection:

```powershell
python scripts/train_baseline.py --split experiments/split_v1 --smoke --output outputs/baseline_smoke
python scripts/train_baseline.py --split experiments/split_v1 --output outputs/baseline_v1
```

Each experiment requires a new empty output directory. Existing runs are deliberately protected from overwriting. Checkpoints and prediction images are saved under `outputs/` and ignored by Git. Seeded execution is used, but bitwise reproducibility across hardware/software is not guaranteed.

The optional `prepare_split.py` script rebuilds a provisional split after an overfit run and reads its sample indices from `--overfit-config`. The committed manifests avoid needing to rerun the audit for baseline reproduction.

## Repository structure

```text
src/doctrace/     Dataset loader and model
scripts/          Inspection, audit, overfit, and baseline commands
experiments/      Split manifests, aggregate metrics, and run metadata
requirements.txt  Direct dependencies
requirements-lock.txt  Installed environment snapshot
```

## Next experiments

1. Inspect validation errors by tampered-region size and document type.
2. Compare a stronger pretrained baseline under the same split and metrics.
3. Evaluate controlled image degradations and a consistency-training hypothesis.
4. Add authentic-document evaluation and a separate physical recapture set.

## Dataset citation

Chenfan Qu et al. *Towards Robust Tampered Text Detection in Document Image: New Dataset and New Solution*. CVPR 2023, pp. 5937-5946. [Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Qu_Towards_Robust_Tampered_Text_Detection_in_Document_Image_New_Dataset_CVPR_2023_paper.html).
