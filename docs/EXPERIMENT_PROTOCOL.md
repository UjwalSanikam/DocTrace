# JPEG robustness experiment

This protocol was specified before inspecting results from the JPEG-trained model.

## Question

The epoch-8 ResNet-18 U-Net scores 0.4724 macro Dice on unmodified development images, but 0.1544, 0.1184, and 0.1031 after JPEG recompression at qualities 90, 70, and 50. Does simple JPEG augmentation improve this sensitivity compared with an equal-length continuation using clean images?

## Matched runs

- Start both runs from `outputs/resnet18_continue_v1/last.pt` (epoch 8), restoring weights, AdamW state, and saved PyTorch/shuffle-generator states.
- Keep the same 10,000 training indices, 2,000 validation indices, batch size 4, learning rate 0.0002, BCE + per-image soft Dice loss, frozen encoder BatchNorm statistics, and float32 arithmetic.
- Run exactly two additional epochs (5,000 optimizer updates per run). Compare **last.pt at epoch 10**, irrespective of the best clean-validation epoch.
- Control: unchanged images.
- JPEG run: independently choose unchanged images with probability 1/2; otherwise JPEG quality 90, 70, or 50 with equal probability. Use Pillow, RGB, 4:2:0 subsampling. Spatial labels do not change.
- Seed 2026. Augmentation worker RNG state is not saved, so restarting an augmented run is not guaranteed to reproduce an uninterrupted run exactly.
- This is a single-seed comparison, not a statistically established effect across training seeds.

## Selection and final evaluation

The release checkpoint is the candidate with the highest equally weighted mean macro Dice over clean, JPEG 90, JPEG 70, and JPEG 50 on the fixed provisional validation set. Candidates are the existing epoch-8 baseline and the two epoch-10 runs. Report clean performance separately, including any tradeoff. Threshold stays at 0.5; no threshold search is performed.

Evaluate each candidate on all seven conditions: clean, JPEG 90/70/50, 0.75x and 0.5x bicubic downsample/upsample, and Gaussian blur radius 0.7. Resizing returns to the original grid; masks remain unchanged. The latter three conditions are diagnostic and are not included in checkpoint selection.

After choosing the release checkpoint, freeze it and evaluate all 2,000 FCD images once under the same conditions. FCD is not used to select weights, hyperparameters, or threshold. Some FCD images were inspected during earlier loader development, so this is not a pristine blind test. No claim of source-document disjointness is made.

Finally, report false-positive pixel fractions on 64 procedurally generated, unedited documents (seeds 1000–1063), at clean and JPEG 70. This is a synthetic sanity check, not evidence of performance on real authentic documents. Image-level authenticity accuracy is not defined or claimed.

## Interpretation limits

The development split groups exact RGB and exact dHash matches, but source IDs are unavailable and related crops or edited variants may cross partitions. Training augmentation and more compute do not establish a novel research method. The project contribution is the reproducible comparison, explicit failure analysis, native-scale inference, and inspectable compression-response demo. Physical recapture and real authentic-document evaluation remain future work.
