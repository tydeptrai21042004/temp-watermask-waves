# Run Arnold-Hess under a WAVES-style benchmark

## Install

```bash
python -m pip install -r requirements_arnold_hess.txt
```

## One command

```bash
./run_arnold_hess_waves.sh \
  --data-source coco-val2017 \
  --dataset mscoco \
  --limit 1000 \
  --attack-preset distortion-single \
  --strengths 0.2,0.4,0.6,0.8,1.0
```

The command automatically downloads COCO val2017, prepares 512x512 hosts, creates a fixed 64x64 binary watermark if needed, embeds Arnold-Hess, attacks the watermarked images, decodes the watermark, computes PSNR/SSIM/NMI, and exports comparison CSV/plots.

## Main output

```text
results/mscoco/arnold_hess_waves_distortion_summary.csv
results/mscoco/arnold_hess_waves_distortion_summary.md
results/mscoco/arnold_hess_waves_psnr_vs_tpr.png
results/mscoco/arnold_hess_waves_nc_by_strength.png
results/mscoco/arnold_hess_waves_bit_accuracy_by_strength.png
results/mscoco/arnold_hess_waves_tpr_by_strength.png
```

The CSV columns most useful for comparison with WAVES-style papers are:

```text
tpr_at_0_1pct_fpr
auc
psnr_mean
ssim_mean
nmi_mean
ber_mean
bit_accuracy_mean
nc_mean
robustness_class
visual_quality_class
```

## Recommended commands

### Fast check

```bash
./run_arnold_hess_waves.sh \
  --data-source coco-val2017 \
  --dataset mscoco \
  --limit 100 \
  --attack-preset lite \
  --strengths 0.2,0.6,1.0
```

### Paper-scale distortion benchmark

```bash
./run_arnold_hess_waves.sh \
  --data-source coco-val2017 \
  --dataset mscoco \
  --limit 5000 \
  --subset-limit 1000 \
  --watermark-size 64 \
  --attack-preset distortion-single \
  --strengths 0.2,0.4,0.6,0.8,1.0
```

### Full distortion set including combo attacks

```bash
./run_arnold_hess_waves.sh \
  --data-source coco-val2017 \
  --dataset mscoco \
  --limit 5000 \
  --subset-limit 1000 \
  --watermark-size 64 \
  --attack-preset distortion-full \
  --strengths 0.2,0.4,0.6,0.8,1.0
```

## Exactness note

The default runner uses:

```text
--repeat-limit 1
--candidate-scoring ultrafast
```

This makes large benchmark runs practical. To use the slower original all-capacity Arnold-Hess schedule:

```bash
./run_arnold_hess_waves.sh \
  --data-source coco-val2017 \
  --dataset mscoco \
  --limit 5000 \
  --watermark-size 64 \
  --repeat-limit full \
  --candidate-scoring fast \
  --attack-preset distortion-single
```

Report the setting used. The CSV `notes` column records the repeat/candidate-scoring configuration.
