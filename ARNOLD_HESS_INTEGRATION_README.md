# Arnold-Hess integration for WAVES

This package adds one fixed, non-optimized watermarking method to WAVES:

```text
arnold_hess
```

The method is the Arnold-scrambled DWT-Hessenberg watermarking method from the user code, but with all Firefly / per-image optimization removed. No custom attack code is included; attacks should come from the official WAVES attack pipeline. The tested method is fixed by the constants in:

```text
watermarks/arnold_hess.py
```

## Important note

`arnold_hess` is **semi-blind / side-information-assisted**. Extraction requires the side-info file saved during embedding:

```text
data/side_info/<dataset>/arnold_hess/<idx>.npz
```

## Added files

```text
watermarks/arnold_hess.py
watermarks/arnold_hess_adapter.py
scripts/embed_arnold_hess.py
scripts/show_arnold_hess_results.py
scripts/run_arnold_hess_smoke_test.py
tests/test_arnold_hess_smoke.py
requirements_arnold_hess.txt
```

## Patched WAVES files

```text
dev/constants.py      # adds custom dataset and arnold_hess method
dev/find.py           # dynamic dataset/source parsing
scripts/decode.py     # CPU decode path for arnold_hess + --mode + --limit
scripts/metric.py     # optional --metrics for lightweight metrics such as psnr,ssim,nmi
scripts/status.py     # optional --limit
cli.py                # fixes dev_scripts path to scripts
```

## Environment

Create or edit `.env`:

```bash
DATA_DIR=/path/to/WAVES/data
RESULT_DIR=/path/to/WAVES/results
CACHE_DIR=/path/to/WAVES/cache
MODEL_DIR=/path/to/WAVES/decoders
ARNOLD_HESS_WM_PATH=/path/to/wm.png
```

For small tests only:

```bash
export WAVES_LIMIT=2
export WAVES_SUBSET_LIMIT=2
```

## Install minimal dependencies

```bash
python -m pip install -r requirements_arnold_hess.txt
```

The lightweight Arnold-Hess smoke test does **not** require GPU.

## Run smoke test

```bash
python scripts/run_arnold_hess_smoke_test.py
```

Or directly:

```bash
python -m pytest -q tests/test_arnold_hess_smoke.py
```

## Official WAVES-style run

### 1. Embed images into official folders

Input images must be 512x512. The watermark must be 64x64 binary.

```bash
python scripts/embed_arnold_hess.py \
  --input /path/to/host_images \
  --watermark /path/to/wm.png \
  --dataset custom \
  --limit 1000
```

This creates:

```text
data/main/custom/real/
data/main/custom/arnold_hess/
data/side_info/custom/arnold_hess/
```

### 2. Generate attacks with the official WAVES attack pipeline

This package does **not** include any Arnold-Hess-specific attack script. The method integration only embeds and decodes `arnold_hess`. Use the original/offical WAVES attack generators or precomputed WAVES attacked folders.

The attacked outputs must follow the standard WAVES folder convention:

```text
data/attacked/<dataset>/<attack_name>-<attack_strength>-<source>/
```

For example, official WAVES distortion/JPEG outputs for this method should be placed as:

```text
data/attacked/custom/distortion_single_jpeg-0.5-arnold_hess/0.png
data/attacked/custom/distortion_single_jpeg-0.5-real/0.png
```

The integration will then use the official WAVES `decode.py`, `metric.py`, and aggregation utilities on those folders.

### 3. Decode using official WAVES JSON output

```bash
python scripts/decode.py \
  --path "$DATA_DIR/main/custom/real" \
  --mode arnold_hess \
  --limit 1000

python scripts/decode.py \
  --path "$DATA_DIR/main/custom/arnold_hess" \
  --mode arnold_hess \
  --limit 1000

python scripts/decode.py \
  --path "$DATA_DIR/attacked/custom/distortion_single_jpeg-0.5-arnold_hess" \
  --mode arnold_hess \
  --limit 1000
```

Output examples:

```text
results/custom/real-decode.json
results/custom/arnold_hess-decode.json
results/custom/distortion_single_jpeg-0.5-arnold_hess-decode.json
```

### 4. Compute WAVES image metrics

For lightweight CPU metrics:

```bash
python scripts/metric.py \
  --path "$DATA_DIR/attacked/custom/distortion_single_jpeg-0.5-arnold_hess" \
  --metrics psnr,ssim,nmi \
  --limit 1000 \
  --subset-limit 1000
```

This writes:

```text
results/custom/distortion_single_jpeg-0.5-arnold_hess-metric.json
```

Full WAVES metrics can still be requested by omitting `--metrics`, but they may require extra models, GPU, and dependencies.

## Get aggregated result

```bash
python scripts/show_arnold_hess_results.py \
  --dataset custom \
  --method arnold_hess \
  --attacks distortion_single_jpeg \
  --strengths 0.5
```

For proper TPR@0.1%FPR, use enough clean and attacked images. Very small smoke-test runs are only for checking code execution.

## One-command Arnold-Hess WAVES-style benchmark

This corrected version adds a self-contained runner:

```bash
./run_arnold_hess_waves.sh \
  --data-source coco-val2017 \
  --dataset mscoco \
  --limit 1000 \
  --attack-preset distortion-single \
  --strengths 0.2,0.4,0.6,0.8,1.0
```

What it does automatically:

1. creates `data/`, `results/`, `cache/`, and `raw/` folders;
2. downloads COCO `val2017.zip` by HTTPS when `--data-source coco-val2017` is used;
3. converts host images to 512x512 RGB PNG files;
4. creates a fixed 64x64 binary watermark if `--watermark` is not provided;
5. embeds the Arnold-Hess method into WAVES folders:
   - `data/main/<dataset>/real/`
   - `data/main/<dataset>/arnold_hess/`
   - `data/side_info/<dataset>/arnold_hess/`
6. generates WAVES-style attacked folders:
   - `data/attacked/<dataset>/<attack>-<strength>-arnold_hess/`
7. decodes Arnold-Hess watermark messages;
8. computes lightweight image metrics: PSNR, SSIM, NMI;
9. exports comparison-ready tables and plots:
   - `results/<dataset>/arnold_hess_waves_distortion_summary.csv`
   - `results/<dataset>/arnold_hess_waves_distortion_summary.md`
   - `results/<dataset>/arnold_hess_waves_psnr_vs_tpr.png`
   - `results/<dataset>/arnold_hess_waves_nc_by_strength.png`
   - `results/<dataset>/arnold_hess_waves_bit_accuracy_by_strength.png`
   - `results/<dataset>/arnold_hess_waves_tpr_by_strength.png`

### Recommended quick run

```bash
./run_arnold_hess_waves.sh \
  --data-source coco-val2017 \
  --dataset mscoco \
  --limit 100 \
  --attack-preset lite \
  --strengths 0.2,0.6,1.0
```

### Recommended paper-scale distortion run

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

The exported CSV uses WAVES-style columns such as:

```text
tpr_at_0_1pct_fpr, tpr_at_1pct_fpr, auc, psnr_mean, ssim_mean,
nmi_mean, ber_mean, bit_accuracy_mean, nc_mean, robustness_class
```

### Reproducing the original all-capacity Arnold-Hess setting

The runner defaults to a practical fast setting:

```text
--repeat-limit 1
--candidate-scoring ultrafast
```

For the slower original all-capacity schedule, use:

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

This is much slower because the original implementation uses many 4x4 block-level Hessenberg decompositions. The selected setting is stored in the CSV `notes` field and in the side-info metadata, so reported results remain auditable.

### Smoke/debug run without downloading COCO

```bash
./run_arnold_hess_waves.sh \
  --data-source synthetic \
  --dataset custom \
  --limit 2 \
  --subset-limit 2 \
  --watermark-size 8 \
  --attacks distortion_single_jpeg \
  --strengths 0.2
```

This is only for checking that the pipeline works; do not use this smoke result in a paper.

### What is and is not included

Included:

- WAVES-style folder layout;
- COCO val2017 auto-download;
- Arnold-Hess embedding/extraction;
- WAVES-style distortion attacks using the official strength ranges;
- TPR@0.1%FPR, AUC, PSNR, SSIM, NMI, BER, bit accuracy, and NC summaries.

Not included by this lightweight command:

- diffusion regeneration attacks;
- adversarial attacks;
- GPU-heavy full quality metrics such as LPIPS/FID/CLIP/aesthetic scores.

Use the exported `arnold_hess_waves_distortion_summary.csv` for direct comparison tables. For strict wording, describe the run as a **WAVES-style distortion benchmark** unless you also add the full regeneration and adversarial WAVES attack groups.
