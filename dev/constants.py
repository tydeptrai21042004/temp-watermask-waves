import os
import warnings
import numpy as np

try:
    import dotenv
    dotenv.load_dotenv(override=False)
except Exception:
    pass

from .io import decode_array_from_string

LIMIT = int(os.environ.get("WAVES_LIMIT", "5000"))
SUBSET_LIMIT = int(os.environ.get("WAVES_SUBSET_LIMIT", "1000"))

DATASET_NAMES = {
    "diffusiondb": "DiffusionDB",
    "mscoco": "MS-COCO",
    "dalle3": "DALL-E 3",
    "custom": "Custom",
}

WATERMARK_METHODS = {
    "tree_ring": "Tree-Ring",
    "stable_sig": "Stable-Signature",
    "stegastamp": "Stega-Stamp",
    "arnold_hess": "Proposed DWT-Hess",
}

PERFORMANCE_METRICS = {
    "acc_1": "Mean Accuracy",
    "auc_1": "AUC",
    "low100_1": "TPR@1%FPR",
    "low1000_1": "TPR@0.1%FPR",
}

QUALITY_METRICS = {
    "legacy_fid": "Legacy FID",
    "clip_fid": "CLIP FID",
    "psnr": "PSNR",
    "ssim": "SSIM",
    "nmi": "Normed Mutual-Info",
    "lpips": "LPIPS",
    "watson": "Watson-DFT",
    "aesthetics": "Delta Aesthetics",
    "artifacts": "Delta Artifacts",
    "clip_score": "Delta CLIP-Score",
}

EVALUATION_SETUPS = {
    "combined": "Combined",
    "removal": "Removal",
    "spoofing": "Spoofing",
}


def _load_arnold_hess_ground_truth():
    """Load 64x64 binary watermark as WAVES bool message.

    The one-command Arnold-Hess runner creates and exports ARNOLD_HESS_WM_PATH
    before evaluation. For general CLI imports/help, avoid crashing when the
    watermark has not been created yet; use a zero placeholder with a warning.
    Any real Arnold-Hess benchmark must still set ARNOLD_HESS_WM_PATH.
    """
    wm_path = os.environ.get("ARNOLD_HESS_WM_PATH")
    if not wm_path:
        # Common defaults created by scripts/run_arnold_hess_waves_benchmark.py.
        candidates = [
            os.path.join(os.getcwd(), "data", "watermark", "arnold_hess_wm64.png"),
            os.path.join(os.getcwd(), "data", "watermark", "wm.png"),
        ]
        wm_path = next((candidate for candidate in candidates if os.path.exists(candidate)), None)

    if not wm_path or not os.path.exists(wm_path):
        warnings.warn(
            "ARNOLD_HESS_WM_PATH is not set and no default watermark was found. "
            "Using a zero placeholder only so non-Arnold-Hess CLI imports can continue. "
            "Run scripts/run_arnold_hess_waves_benchmark.py or set ARNOLD_HESS_WM_PATH "
            "before reporting Arnold-Hess results."
        )
        wm_size = int(os.environ.get("ARNOLD_HESS_WM_SIZE", "64"))
        return np.zeros((wm_size, wm_size), dtype=bool)

    try:
        from watermarks.arnold_hess_adapter import ground_truth_message
        return ground_truth_message(wm_path).astype(bool)
    except Exception as exc:
        raise RuntimeError(f"Could not load ARNOLD_HESS_WM_PATH={wm_path}: {exc}") from exc


GROUND_TRUTH_MESSAGES = {
    "tree_ring": decode_array_from_string(
        "H4sIALRwUmUC/42SvYrCQBSFLW18iam3EZcUFgErkYBFSLcYENZgISgoiMjCVj6FzyFCCoUlTQgrE8TnkcNhcEZIbjhw7x3Ox/zdu1fr+XQ1U/0vr/c5+VDfmx1WKlksp5uup35anX+jLHrVWFHX0FS2cw2h8x+z69KBYs1MxvVjDXkPZpvh/iC8B1SUzKTMWTZTlEV5iBBtzqVAHKJEI4KrphL9OwInU1AzUqbku9W/s+mf1f+93D+5/1Vz8z5j+djIv79qrKj2wFS20x5Al5LZdelApxszGdc/3aBUM9sM9weRasgPmEmZs2zGD/xgmyPanEuB2ObHDBFcNXXMwiE4mYKakTIl363+nU3/rP7v5f7J/a+am/cZewKA1ipNFgUAAA=="
    ),
    "stable_sig": decode_array_from_string(
        "H4sIADtrUmUC/6tWKs5ILEhVsoo2sYjVUUopqQRxlJLy83OUahkYGRkZgBBMMIAAhAvmwUUZIRIgcQBxGJ0kTgAAAA=="
    ),
    "stegastamp": decode_array_from_string(
        "H4sIAGRrUmUC/6tWKs5ILEhVsoo2NDCI1VFKKakE8ZSS8vNzlGoZGBkZGRhAmAFCgxGIC+dBCAaYEEyCEVkOzISrYIToZIAoY2AEAG5jy4ODAAAA"
    ),
    "arnold_hess": _load_arnold_hess_ground_truth(),
}


ATTACK_NAMES = {
    "distortion_single_rotation": "Dist-Rotation",
    "distortion_single_resizedcrop": "Dist-RCrop",
    "distortion_single_erasing": "Dist-Erase",
    "distortion_single_brightness": "Dist-Bright",
    "distortion_single_contrast": "Dist-Contrast",
    "distortion_single_blurring": "Dist-Blur",
    "distortion_single_noise": "Dist-Noise",
    "distortion_single_jpeg": "Dist-JPEG",
    "distortion_combo_geometric": "Dist-Com-Geo",
    "distortion_combo_photometric": "Dist-Com-Photo",
    "distortion_combo_degradation": "Dist-Com-Deg",
    "distortion_combo_all": "Dist-Com-All",
    "regen_diffusion": "Regen-Diffusion",
    "regen_diffusion_prompt": "Regen-Diffusion&P",
    "regen_vae": "Regen-VAE",
    "kl_vae": "Regen-KLVAE",
    "2x_regen": "Regen-2xDiffusion",
    "4x_regen": "Regen-4xDiffusion",
    "4x_regen_bmshj": "Regen-4xVAE",
    "4x_regen_kl_vae": "Regen-4xKLVAE",
    "adv_emb_resnet18_untg": "AdvEmb-RN18",
    "adv_emb_clip_untg_alphaRatio_0.05_step_200": "AdvEmb-CLIP",
    "adv_emb_same_vae_untg": "AdvEmb-KLVAE8",
    "adv_emb_klf16_vae_untg": "AdvEmb-KLVAE16",
    "adv_emb_sdxl_vae_untg": "AdvEmb-SdxlVAE",
    "adv_cls_unwm_wm_0.01_50_warm_train3k": "AdvCls-UnWM-WM",
    "adv_cls_real_wm_0.01_50_warm": "AdvCls-Real-WM",
    "adv_cls_wm1_wm2_0.01_50_warm": "AdvCls-WM1-WM2",
    "adv_cls_wm1_wm2_0.04_200_warm": "abandon",
}
