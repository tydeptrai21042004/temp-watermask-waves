import os
import time
import traceback
import hashlib
import json
import csv
import random
import math
import shutil
from copy import deepcopy

import cv2
import numpy as np
try:
    import pywt
except Exception:
    pywt = None
from scipy import linalg
from skimage.metrics import structural_similarity as ssim


# =========================================================
# Global settings
# =========================================================
BLOCKSIZE = 4
WM_SIZE = int(os.environ.get("ARNOLD_HESS_WM_SIZE", "64"))
PRIVATE_KEY = os.environ.get("ARNOLD_HESS_PRIVATE_KEY", "KB123")
HOST_SIZE = int(os.environ.get("ARNOLD_HESS_HOST_SIZE", "512"))

ARNOLD_ITERATIONS = int(os.environ.get("ARNOLD_HESS_ITERATIONS", "17"))

DWT_WAVELET = "haar"
DWT_LEVEL = 1

DWT_BANDS = tuple(
    x.strip() for x in os.environ.get("ARNOLD_HESS_DWT_BANDS", "LL,HL,HH,LH").split(",") if x.strip()
)
DWT_BAND_CHOICES = [
    ("LH", "HL"),
    ("LL", "HL"),
    ("LL", "LH", "HL"),
    ("LL", "HL", "HH", "LH"),
]

# Embedding is now performed only in the luminance channel.
# Images are loaded/saved as OpenCV BGR, but the embedding domain is YCrCb.
# Channel index 0 in YCrCb is Y.
HOST_EMBED_COLOR_SPACE = "YCrCb"
HOST_CHANNELS = (0,)  # Y only after BGR -> YCrCb conversion
WM_BIN_THRESH = 127

FLAG_Q4 = 0
FLAG_HPOS = 1
FLAG_SKIP = 2

Q4_TAU = 0.50
Q4_MARGIN = 0.08

H01_Q = 7.0
H01_MARGIN = 0.90

# Only two H-domain candidates, matching the simplified paper description.
# Mathematical notation:
#   h21 -> h_{2,1}, code index (1,0)
#   h22 -> h_{2,2}, code index (1,1)
HPOS_CANDIDATES = [
    ("h21", (1, 0)),
    ("h22", (1, 1)),
]
HPOS_NONE = -1

EPS_SCORE = 1e-9
EPS_CONF = 1e-12

USE_STRUCTURED_REPETITION = True

# =========================================================
# Engineering-style candidate selection
# =========================================================
# Bit Survival Score (BSS):
#   BSS = passed_stress_tests / total_stress_tests
#
# H-candidate selection:
#   choose the valid H candidate with the lowest MSE.
#
# Final Q-vs-H selection:
#   choose the candidate with the highest combined engineering score:
#   score = BSS_WEIGHT * BSS - MSE_WEIGHT * normalized_MSE
MIN_SURVIVAL_RATE = 0.75
BSS_WEIGHT = 0.75
MSE_WEIGHT = 0.25

MAX_Q_CAND_MSE = 0.18
MAX_H_CAND_MSE = 0.24

Q4_GIVENS_THETA_MAX = 0.42
Q4_GIVENS_COARSE_STEPS = 13
Q4_GIVENS_FINE_THETA_MAX = 0.08
Q4_GIVENS_FINE_STEPS = 0
Q4_GIVENS_PASSES = 1
Q4_GIVENS_MSE_WEIGHT = 0.10
Q4_GIVENS_EXTRA_MARGIN_WEIGHT = 0.04
Q4_GIVENS_PAIRS = (
    (0, 2),
    (1, 3),
    (0, 1),
    (1, 2),
)

FAST_CANDIDATE_SCORING = True
DEBUG = bool(int(os.environ.get("ARNOLD_HESS_DEBUG", "0")))


# =========================================================
# Attack settings
# =========================================================
PSNR_MIN_THRESHOLD = 55.0

ATTACK_SUITE = [
    ("Blur", 1.0),
    ("Sharpen", (1.0, 1.5)),
    ("Speckle Noise", (0.0, 0.001)),
    ("Salt & Pepper", 0.1),
    ("JPEG", 90),
    ("JPEG2000", 7.0),
    ("Lowpass", (5, 5)),
    ("Scale", 0.5),
    ("Scale", 4.0),
    ("Rotation", 45.0),
    ("Histogram", None),
    ("Occlusion", (50.0, "random", "random", 3)),
]

# Faster attack set for Firefly search. The final saved result can still be
# evaluated with ATTACK_SUITE if you pass attack_suite=ATTACK_SUITE manually.
FAST_ATTACK_SUITE = [
    ("JPEG", 90),
    ("Blur", 1.0),
    ("Salt & Pepper", 0.1),
    ("Scale", 0.5),
    ("Rotation", 45.0),
]


# =========================================================
# Basic helpers
# =========================================================
def _debug(msg):
    if DEBUG:
        print(msg)


def _flag_stats(flags):
    q4_used = sum(1 for f in flags if int(f) == FLAG_Q4)
    hpos_used = sum(1 for f in flags if int(f) == FLAG_HPOS)
    skip_used = sum(1 for f in flags if int(f) == FLAG_SKIP)
    return q4_used, hpos_used, skip_used


def _json_safe_number(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def _to_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _safe_filename(s):
    s = str(s)
    for ch in [" ", "&", "(", ")", ",", ".", "'", '"', "/", "\\", ":", ";", "[", "]"]:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def load_color_image_24bit_safe(path, size=None, bg=255):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image file: {path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        bgr = img[..., :3].astype(np.float32)
        a = img[..., 3:4].astype(np.float32) / 255.0
        bg_bgr = np.full_like(bgr, float(bg), dtype=np.float32)
        comp = bgr * a + bg_bgr * (1.0 - a)
        img = np.clip(np.rint(comp), 0, 255).astype(np.uint8)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = img.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported image format for: {path}")

    if size is not None:
        if img.shape[0] != size or img.shape[1] != size:
            raise ValueError(
                f"Image must be exactly {size}x{size}. Got {img.shape[1]}x{img.shape[0]}."
            )
    return img


def load_binary_watermark_8bit_safe(path, size=None, thresh=WM_BIN_THRESH):
    img = load_color_image_24bit_safe(path, size=size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    wm_bin = np.where(gray >= int(thresh), 255, 0).astype(np.uint8)
    if size is not None and wm_bin.shape != (size, size):
        raise ValueError(f"Binary watermark must be exactly {size}x{size}.")
    return wm_bin


def _force_binary_watermark_exact(img_u8: np.ndarray, size: int, thresh: int = WM_BIN_THRESH) -> np.ndarray:
    if img_u8 is None:
        raise ValueError("Watermark image is None.")
    out = img_u8.astype(np.uint8)
    if out.shape != (size, size):
        out = cv2.resize(out, (size, size), interpolation=cv2.INTER_NEAREST)
    out = np.where(out >= int(thresh), 255, 0).astype(np.uint8)
    return out


def _host_bgr_to_embed_space(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR host image to the embedding color space.

    The requested embedding domain is the Y channel, so this helper converts
    OpenCV BGR to YCrCb. The rest of the algorithm can keep using channel index
    0 through HOST_CHANNELS = (0,).
    """
    img_bgr = _ensure_uint8_bgr_basic(img_bgr)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.uint8)


def _embed_space_to_host_bgr(img_embed: np.ndarray) -> np.ndarray:
    """Convert the embedding color space back to OpenCV BGR for saving/display."""
    img_embed = _ensure_uint8_bgr_basic(img_embed)
    return cv2.cvtColor(img_embed, cv2.COLOR_YCrCb2BGR).astype(np.uint8)


def _ensure_uint8_bgr_basic(img: np.ndarray) -> np.ndarray:
    """Small local uint8/3-channel guard used before attack helpers are defined."""
    img = np.asarray(img)
    if img.ndim == 2:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Image must have 3 channels.")
    return np.clip(img, 0, 255).astype(np.uint8)


# =========================================================
# Haar DWT fallback helpers
# =========================================================
def _haar_dwt2(channel_f64: np.ndarray):
    a = np.asarray(channel_f64, dtype=np.float64)
    if a.shape[0] % 2 != 0 or a.shape[1] % 2 != 0:
        a = a[: (a.shape[0] // 2) * 2, : (a.shape[1] // 2) * 2]
    even_rows = a[0::2, :]
    odd_rows = a[1::2, :]
    low_rows = (even_rows + odd_rows) / np.sqrt(2.0)
    high_rows = (even_rows - odd_rows) / np.sqrt(2.0)
    LL = (low_rows[:, 0::2] + low_rows[:, 1::2]) / np.sqrt(2.0)
    LH = (low_rows[:, 0::2] - low_rows[:, 1::2]) / np.sqrt(2.0)
    HL = (high_rows[:, 0::2] + high_rows[:, 1::2]) / np.sqrt(2.0)
    HH = (high_rows[:, 0::2] - high_rows[:, 1::2]) / np.sqrt(2.0)
    return LL, (LH, HL, HH)


def _haar_idwt2(LL: np.ndarray, LH: np.ndarray, HL: np.ndarray, HH: np.ndarray):
    low_even = (LL + LH) / np.sqrt(2.0)
    low_odd = (LL - LH) / np.sqrt(2.0)
    high_even = (HL + HH) / np.sqrt(2.0)
    high_odd = (HL - HH) / np.sqrt(2.0)
    h, w = LL.shape
    out = np.zeros((h * 2, w * 2), dtype=np.float64)
    out[0::2, 0::2] = (low_even + high_even) / np.sqrt(2.0)
    out[1::2, 0::2] = (low_even - high_even) / np.sqrt(2.0)
    out[0::2, 1::2] = (low_odd + high_odd) / np.sqrt(2.0)
    out[1::2, 1::2] = (low_odd - high_odd) / np.sqrt(2.0)
    return out


def _dwt2(arr: np.ndarray, wavelet: str):
    if pywt is not None:
        return pywt.dwt2(arr, wavelet)
    if str(wavelet).lower() != "haar":
        raise ImportError("PyWavelets is required for non-Haar DWT wavelets.")
    return _haar_dwt2(arr)


def _idwt2(coeffs, wavelet: str):
    if pywt is not None:
        return pywt.idwt2(coeffs, wavelet)
    if str(wavelet).lower() != "haar":
        raise ImportError("PyWavelets is required for non-Haar inverse DWT wavelets.")
    LL, details = coeffs
    LH, HL, HH = details
    return _haar_idwt2(LL, LH, HL, HH)


# =========================================================
# Arnold scrambling helpers
# =========================================================
def _normalize_arnold_iterations(iterations: int) -> int:
    iterations = int(round(iterations))
    if iterations < 0:
        raise ValueError("ARNOLD_ITERATIONS must be non-negative.")
    return iterations


def arnold_scramble_binary(wm_bits_2d: np.ndarray, iterations: int) -> np.ndarray:
    wm_bits_2d = np.asarray(wm_bits_2d, dtype=np.uint8)
    if wm_bits_2d.ndim != 2:
        raise ValueError("Arnold scrambling requires a 2D binary image.")

    h, w = wm_bits_2d.shape
    if h != w:
        raise ValueError("Arnold scrambling requires a square watermark.")

    n = h
    out = wm_bits_2d.copy()
    iterations = _normalize_arnold_iterations(iterations)

    for _ in range(iterations):
        temp = np.zeros_like(out)
        for x in range(n):
            for y in range(n):
                x_new = (x + y) % n
                y_new = (x + 2 * y) % n
                temp[x_new, y_new] = out[x, y]
        out = temp

    return out.astype(np.uint8)


def arnold_unscramble_binary(scrambled_bits_2d: np.ndarray, iterations: int) -> np.ndarray:
    scrambled_bits_2d = np.asarray(scrambled_bits_2d, dtype=np.uint8)
    if scrambled_bits_2d.ndim != 2:
        raise ValueError("Inverse Arnold requires a 2D binary image.")

    h, w = scrambled_bits_2d.shape
    if h != w:
        raise ValueError("Inverse Arnold requires a square watermark.")

    n = h
    out = scrambled_bits_2d.copy()
    iterations = _normalize_arnold_iterations(iterations)

    for _ in range(iterations):
        temp = np.zeros_like(out)
        for x_new in range(n):
            for y_new in range(n):
                x = (2 * x_new - y_new) % n
                y = (-x_new + y_new) % n
                temp[x, y] = out[x_new, y_new]
        out = temp

    return out.astype(np.uint8)


# =========================================================
# DWT helpers
# =========================================================
def _required_multiple_for_dwt(level: int, block_size: int) -> int:
    return (2 ** level) * block_size


def _crop_for_dwt(channel_u8: np.ndarray, level: int, block_size: int):
    req = _required_multiple_for_dwt(level, block_size)
    h, w = channel_u8.shape
    h0 = (h // req) * req
    w0 = (w // req) * req
    if h0 == 0 or w0 == 0:
        raise ValueError("Image too small for DWT + BLOCKSIZE constraint.")
    return channel_u8[:h0, :w0].astype(np.float64), h0, w0


def _dwt_split_4bands(channel_f64: np.ndarray, wavelet: str):
    LL, (LH, HL, HH) = _dwt2(channel_f64, wavelet)
    return {
        "LL": LL.copy(),
        "LH": LH.copy(),
        "HL": HL.copy(),
        "HH": HH.copy(),
    }


def _dwt_merge_4bands(bands: dict, wavelet: str):
    return _idwt2((bands["LL"], (bands["LH"], bands["HL"], bands["HH"])), wavelet)


def generate_color_subband_block_indices(bands_by_ch, channel_ids, band_names, block_size, total_needed, key):
    """
    Key-based block shuffling without logistic map.
    Uses SHA-256 digest as deterministic RNG seed.
    """
    all_positions = []

    for ch in channel_ids:
        for band_name in band_names:
            band = bands_by_ch[ch][band_name]
            h, w = band.shape
            h0 = (h // block_size) * block_size
            w0 = (w // block_size) * block_size
            for i in range(0, h0, block_size):
                for j in range(0, w0, block_size):
                    all_positions.append((ch, band_name, i, j))

    total_blocks = len(all_positions)
    if total_needed > total_blocks:
        raise ValueError(
            f"Capacity insufficient: Need {total_needed} blocks, but only {total_blocks} blocks available."
        )

    key_bytes = hashlib.sha256(str(key).encode("utf-8")).digest()
    seed_int = int.from_bytes(key_bytes, "big")
    rng = random.Random(seed_int)
    indices = list(range(total_blocks))
    rng.shuffle(indices)
    selected_indices = indices[:total_needed]
    return [all_positions[idx] for idx in selected_indices]


def make_structured_repetition_schedule(blocks, payload_len: int):
    payload_len = int(payload_len)
    if payload_len <= 0:
        raise ValueError("payload_len must be positive.")

    total_blocks = len(blocks)
    repeat_factor = total_blocks // payload_len

    if repeat_factor <= 0:
        raise ValueError(
            f"Capacity insufficient: payload_len={payload_len}, total_blocks={total_blocks}."
        )

    usable_blocks = repeat_factor * payload_len
    selected_blocks = list(blocks[:usable_blocks])

    schedule = []
    for rep in range(repeat_factor):
        base = rep * payload_len
        for kpos in range(payload_len):
            schedule.append((kpos, selected_blocks[base + kpos]))

    return schedule, repeat_factor, usable_blocks


def _max_payload_bits_for_host(host_h, host_w, dwt_bands=None):
    if dwt_bands is None:
        dwt_bands = DWT_BANDS
    h2 = (host_h // 2) // BLOCKSIZE
    w2 = (host_w // 2) // BLOCKSIZE
    return len(HOST_CHANNELS) * len(tuple(dwt_bands)) * h2 * w2


def _payload_len_from_meta(meta: dict) -> int:
    return int(meta["payload_len"])


# =========================================================
# Embed info helpers
# =========================================================
def _save_embed_info(embed_info_path, flags, hpos_list, payload_meta):
    dir_name = os.path.dirname(embed_info_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    meta_json = json.dumps(_to_json_safe(payload_meta), ensure_ascii=False)
    np.savez_compressed(
        embed_info_path,
        flags=np.array(flags, dtype=np.int16),
        hpos=np.array(hpos_list, dtype=np.int16),
        meta_json=np.array(meta_json),
    )


def _load_embed_info(embed_info_path):
    if not os.path.exists(embed_info_path):
        raise ValueError(f"Embed info file missing: {embed_info_path}")

    data = np.load(embed_info_path, allow_pickle=False)

    if "flags" not in data or "hpos" not in data:
        raise ValueError(f"Embed info file must contain 'flags' and 'hpos': {embed_info_path}")

    flags = data["flags"]
    hpos = data["hpos"]

    if flags.ndim == 0:
        flags = np.array([int(flags)], dtype=np.int32)
    else:
        flags = flags.astype(np.int32)

    if hpos.ndim == 0:
        hpos = np.array([int(hpos)], dtype=np.int32)
    else:
        hpos = hpos.astype(np.int32)

    if len(hpos) < len(flags):
        pad = np.full((len(flags) - len(hpos),), HPOS_NONE, dtype=np.int32)
        hpos = np.concatenate([hpos, pad], axis=0)
    elif len(hpos) > len(flags):
        hpos = hpos[:len(flags)]

    if "meta_json" in data:
        meta_json = data["meta_json"].item() if getattr(data["meta_json"], "ndim", 0) == 0 else str(data["meta_json"])
        payload_meta = json.loads(meta_json)
    else:
        payload_meta = {
            "payload_len": WM_SIZE * WM_SIZE,
            "wm_bit_len": WM_SIZE * WM_SIZE,
            "wm_size": WM_SIZE,
            "arnold_enabled": True,
            "arnold_iterations": int(ARNOLD_ITERATIONS),
            "dwt_bands": list(DWT_BANDS),
        }

    return [int(x) for x in flags.tolist()], [int(x) for x in hpos.tolist()], payload_meta


# =========================================================
# Misc math helpers
# =========================================================
def _safe_sign(x):
    return 1.0 if x >= 0 else -1.0


def _nearest_nonnegative_mod_value(a: float, target: float, q: float) -> float:
    k0 = int(np.round((a - target) / q))
    cands = []
    for k in range(k0 - 3, k0 + 4):
        v = target + k * q
        if v >= 0:
            cands.append(v)
    if not cands:
        return max(0.0, target)
    return min(cands, key=lambda v: abs(v - a))


def _safe_stats(arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.min(arr)), float(np.mean(arr)), float(np.max(arr))


def _print_embed_support_summary(support_counts, support_sel_sum, label: str, target_support=None):
    support_counts = np.asarray(support_counts, dtype=np.int32)
    support_sel_sum = np.asarray(support_sel_sum, dtype=np.float64)

    mn, av, mx = _safe_stats(support_counts)

    if target_support is None:
        target_support = int(np.max(support_counts)) if support_counts.size > 0 else 0

    weak_bits = int(np.sum(support_counts < int(target_support)))
    zero_bits = int(np.sum(support_counts <= 0))

    sel_avg = np.zeros_like(support_sel_sum, dtype=np.float64)
    mask = support_counts > 0
    sel_avg[mask] = support_sel_sum[mask] / np.maximum(support_counts[mask], 1)

    sel_mn, sel_av, sel_mx = _safe_stats(sel_avg[mask]) if np.any(mask) else (0.0, 0.0, 0.0)

    _debug(
        f"[MAIN-EMBED][{label}] support_per_bit(min/mean/max)=({mn:.2f}, {av:.2f}, {mx:.2f}) | "
        f"target_support={int(target_support)} | "
        f"bits_below_target={weak_bits}/{support_counts.size} | zero_bits={zero_bits} | "
        f"avg_combined_score(min/mean/max)=({sel_mn:.4f}, {sel_av:.4f}, {sel_mx:.4f})"
    )


def _print_mode_band_summary(mode_band_counter: dict, label: str):
    if not mode_band_counter:
        return
    parts = []
    for key in sorted(mode_band_counter.keys()):
        parts.append(f"{key}={mode_band_counter[key]}")
    _debug(f"[MAIN-EMBED][{label}] mode_band_counts: " + ", ".join(parts))


# =========================================================
# Watermark payload using Arnold scrambling
# =========================================================
def prepare_binary_watermark_payload(
    watermark_path,
    wm_size,
    arnold_iterations=ARNOLD_ITERATIONS,
):
    """Prepare watermark payload with watermark-side DWT.

    Requested change:
      1. Load the binary watermark.
      2. Apply one-level Haar DWT to the watermark before embedding.
      3. Use only the LL subband as the payload.
      4. Save the other DWT subbands and reconstruction information in meta.

    Because the embedding engine is binary-bit based, the LL coefficients are
    converted to a binary LL payload by thresholding. During reconstruction, the
    extracted LL bits are mapped back to representative low/high LL coefficient
    values and combined with the saved LH/HL/HH subbands.
    """
    wm_binary = load_binary_watermark_8bit_safe(watermark_path, size=wm_size)
    wm_binary = _force_binary_watermark_exact(wm_binary, wm_size, thresh=WM_BIN_THRESH)

    # Work with a normalized binary watermark: 0.0 or 1.0.
    wm_bits_2d = (wm_binary >= WM_BIN_THRESH).astype(np.float64)

    # Watermark-side DWT. Only LL is embedded; LH/HL/HH are saved in meta.
    LL, (LH, HL, HH) = _dwt2(wm_bits_2d, DWT_WAVELET)

    # Haar LL of a binary image is in [0, 2]. Threshold 1.0 corresponds roughly
    # to local 2x2 average >= 0.5.
    ll_threshold = 1.0
    ll_bits_2d = (LL >= ll_threshold).astype(np.uint8)

    low_vals = LL[LL < ll_threshold]
    high_vals = LL[LL >= ll_threshold]
    ll_low_value = float(np.mean(low_vals)) if low_vals.size > 0 else 0.0
    ll_high_value = float(np.mean(high_vals)) if high_vals.size > 0 else 2.0

    scrambled_bits_2d = arnold_scramble_binary(ll_bits_2d, arnold_iterations)
    payload_bits = scrambled_bits_2d.reshape(-1).astype(np.uint8)

    meta = {
        "wm_shape": tuple(int(x) for x in wm_binary.shape),
        "wm_size": int(wm_size),
        "original_wm_bit_len": int(wm_bits_2d.size),
        "wm_bit_len": int(ll_bits_2d.size),
        "payload_len": int(payload_bits.size),
        "payload_shape": tuple(int(x) for x in ll_bits_2d.shape),
        "arnold_enabled": True,
        "arnold_iterations": int(arnold_iterations),
        "watermark_dwt_enabled": True,
        "watermark_dwt_wavelet": str(DWT_WAVELET),
        "watermark_dwt_level": 1,
        "watermark_payload_band": "LL",
        "watermark_ll_threshold": float(ll_threshold),
        "watermark_ll_low_value": float(ll_low_value),
        "watermark_ll_high_value": float(ll_high_value),
        "watermark_detail_subbands": {
            "LH": LH.astype(float).tolist(),
            "HL": HL.astype(float).tolist(),
            "HH": HH.astype(float).tolist(),
        },
    }

    return wm_binary, payload_bits.astype(np.uint8), meta


def reconstruct_binary_watermark_from_payload_bits(payload_bits, meta: dict):
    arnold_iterations = int(meta.get("arnold_iterations", ARNOLD_ITERATIONS))

    # New path: extracted payload represents only the watermark LL subband.
    if bool(meta.get("watermark_dwt_enabled", False)):
        wm_size = int(meta["wm_size"])
        payload_len = int(meta["payload_len"])
        payload_shape = tuple(int(x) for x in meta.get("payload_shape", (wm_size // 2, wm_size // 2)))

        bits_scrambled = np.array(payload_bits[:payload_len], dtype=np.uint8).reshape(-1)
        if bits_scrambled.size < payload_len:
            pad = np.zeros((payload_len - bits_scrambled.size,), dtype=np.uint8)
            bits_scrambled = np.concatenate([bits_scrambled, pad], axis=0)
        elif bits_scrambled.size > payload_len:
            bits_scrambled = bits_scrambled[:payload_len]

        scrambled_2d = bits_scrambled.reshape(payload_shape).astype(np.uint8)
        recovered_ll_bits = arnold_unscramble_binary(scrambled_2d, arnold_iterations)

        ll_low = float(meta.get("watermark_ll_low_value", 0.0))
        ll_high = float(meta.get("watermark_ll_high_value", 2.0))
        LL_rec = np.where(recovered_ll_bits > 0, ll_high, ll_low).astype(np.float64)

        details = meta.get("watermark_detail_subbands", {})
        LH = np.asarray(details.get("LH"), dtype=np.float64)
        HL = np.asarray(details.get("HL"), dtype=np.float64)
        HH = np.asarray(details.get("HH"), dtype=np.float64)

        if LH.shape != LL_rec.shape or HL.shape != LL_rec.shape or HH.shape != LL_rec.shape:
            raise ValueError("Saved watermark DWT detail subbands do not match extracted LL shape.")

        wavelet = str(meta.get("watermark_dwt_wavelet", DWT_WAVELET))
        wm_float = _idwt2((LL_rec, (LH, HL, HH)), wavelet)
        wm_float = wm_float[:wm_size, :wm_size]
        wm_rec = np.where(wm_float >= 0.5, 255, 0).astype(np.uint8)
        wm_rec = _force_binary_watermark_exact(wm_rec, wm_size, thresh=WM_BIN_THRESH)
        return wm_rec

    # Backward-compatible fallback for older embed_info files.
    wm_size = int(meta["wm_size"])
    wm_bit_len = int(meta["wm_bit_len"])
    payload_len = int(meta["payload_len"])

    bits_scrambled = np.array(payload_bits[:payload_len], dtype=np.uint8).reshape(-1)

    if bits_scrambled.size < wm_bit_len:
        pad = np.zeros((wm_bit_len - bits_scrambled.size,), dtype=np.uint8)
        bits_scrambled = np.concatenate([bits_scrambled, pad], axis=0)
    elif bits_scrambled.size > wm_bit_len:
        bits_scrambled = bits_scrambled[:wm_bit_len]

    scrambled_2d = bits_scrambled.reshape((wm_size, wm_size)).astype(np.uint8)
    recovered_bits_2d = arnold_unscramble_binary(scrambled_2d, arnold_iterations)

    wm_rec = (recovered_bits_2d * 255).astype(np.uint8)
    wm_rec = _force_binary_watermark_exact(wm_rec, wm_size, thresh=WM_BIN_THRESH)
    return wm_rec


# =========================================================
# Q candidate
# =========================================================
def _q4_stat(Qm: np.ndarray) -> float:
    return float(Qm[1, 1] ** 2 + Qm[2, 1] ** 2)


def _llr_q4_from_Q(Qm: np.ndarray) -> float:
    return _q4_stat(Qm) - float(Q4_TAU)


def _norm_llr_q4(llr_q4: float) -> float:
    return float(llr_q4) / max(float(Q4_MARGIN), EPS_CONF)


def _givens4(a: int, b: int, theta: float) -> np.ndarray:
    G = np.eye(4, dtype=np.float64)
    c = np.cos(theta)
    s = np.sin(theta)
    G[a, a] = c
    G[a, b] = s
    G[b, a] = -s
    G[b, b] = c
    return G


def _q4_signed_margin_from_Q(Qm: np.ndarray, bit: int) -> float:
    d = _q4_stat(Qm) - float(Q4_TAU)
    return float(d) if int(bit) == 1 else -float(d)


def _q4_givens_loss(block0: np.ndarray, Hh: np.ndarray, Qtrial: np.ndarray, bit: int) -> tuple:
    signed_margin = _q4_signed_margin_from_Q(Qtrial, bit)
    target = float(Q4_MARGIN)

    if signed_margin >= target:
        margin_loss = 0.0
        extra_margin = signed_margin - target
    else:
        margin_loss = target - signed_margin
        extra_margin = 0.0

    block_trial = Qtrial @ Hh @ Qtrial.T
    mse = float(np.mean((block0 - block_trial) ** 2))

    loss = (
        margin_loss
        + float(Q4_GIVENS_MSE_WEIGHT) * math.sqrt(max(mse, 0.0))
        + float(Q4_GIVENS_EXTRA_MARGIN_WEIGHT) * max(extra_margin, 0.0)
    )

    return float(loss), float(mse), float(signed_margin)


def _build_q4_candidate(block: np.ndarray, bit: int):
    Hh, Qm = linalg.hessenberg(block, calc_q=True)

    best_Q = Qm.copy()
    best_loss, best_mse, best_margin = _q4_givens_loss(block, Hh, best_Q, bit)

    if best_margin >= float(Q4_MARGIN):
        return best_Q @ Hh @ best_Q.T

    grids = [
        np.linspace(-float(Q4_GIVENS_THETA_MAX), float(Q4_GIVENS_THETA_MAX), int(Q4_GIVENS_COARSE_STEPS)),
    ]
    if int(Q4_GIVENS_FINE_STEPS) > 1:
        grids.append(
            np.linspace(
                -float(Q4_GIVENS_FINE_THETA_MAX),
                float(Q4_GIVENS_FINE_THETA_MAX),
                int(Q4_GIVENS_FINE_STEPS),
            )
        )

    num_passes = max(1, int(Q4_GIVENS_PASSES))

    for pass_idx in range(num_passes):
        theta_grid = grids[min(pass_idx, len(grids) - 1)]
        improved = False

        for a, b in Q4_GIVENS_PAIRS:
            local_best_Q = best_Q
            local_best_key = (best_loss, best_mse, -best_margin)

            for th in theta_grid:
                if abs(float(th)) < 1e-15:
                    continue

                G = _givens4(a, b, float(th))
                Qtrial = G @ best_Q

                loss, mse, signed_margin = _q4_givens_loss(block, Hh, Qtrial, bit)
                key = (loss, mse, -signed_margin)

                if key < local_best_key:
                    local_best_key = key
                    local_best_Q = Qtrial

            if local_best_Q is not best_Q:
                best_Q = local_best_Q
                best_loss, best_mse, best_margin = _q4_givens_loss(block, Hh, best_Q, bit)
                improved = True

        if not improved:
            break

        if best_margin >= float(Q4_MARGIN) and best_loss <= float(Q4_GIVENS_MSE_WEIGHT) * math.sqrt(max(best_mse, 0.0)) + 1e-12:
            break

    return best_Q @ Hh @ best_Q.T


# =========================================================
# H candidate
# =========================================================
def _llr_hpos_from_H(Hh: np.ndarray, q_h01: float, pos) -> float:
    rr, cc = pos
    z = abs(float(Hh[rr, cc])) % float(q_h01)
    return z - 0.5 * float(q_h01)


def _norm_llr_hpos(llr_h01: float, q_h01: float) -> float:
    return float(llr_h01) / max(0.5 * float(q_h01), EPS_CONF)


def _build_hpos_candidate(block: np.ndarray, bit: int, q_h01: float, h01_margin: float, pos):
    Hh, Qm = linalg.hessenberg(block, calc_q=True)
    H2 = Hh.copy()

    rr, cc = pos
    v = float(H2[rr, cc])
    sgn = _safe_sign(v)
    a = abs(v)
    z = a % q_h01

    safe = False
    if bit == 1:
        if z >= (0.5 * q_h01 + h01_margin) and z <= (q_h01 - h01_margin):
            safe = True
    else:
        if z >= h01_margin and z <= (0.5 * q_h01 - h01_margin):
            safe = True

    if not safe:
        target = 0.75 * q_h01 if bit == 1 else 0.25 * q_h01
        a_new = _nearest_nonnegative_mod_value(a, target, q_h01)
        H2[rr, cc] = sgn * a_new

    return Qm @ H2 @ Qm.T


# =========================================================
# Engineering-style Bit Survival Score and candidate selection
# =========================================================
def _extract_bit_from_candidate_block(block: np.ndarray, mode: int, q_h01: float, pos=None) -> int:
    """
    Extract one bit from a candidate block using the same threshold rule used
    during final watermark extraction.
    """
    Hh, Qm = linalg.hessenberg(block, calc_q=True)

    if mode == FLAG_Q4:
        return 1 if _q4_stat(Qm) >= float(Q4_TAU) else 0

    if mode == FLAG_HPOS:
        if pos is None:
            raise ValueError("H-domain candidate requires a coefficient position.")
        rr, cc = pos
        z = abs(float(Hh[rr, cc])) % float(q_h01)
        return 1 if z >= 0.5 * float(q_h01) else 0

    raise ValueError("Unknown candidate mode.")


def _candidate_attack_score(block0: np.ndarray, block_cand: np.ndarray, mode: int, bit: int, q_h01: float, pos=None):
    """
    Engineering-style Bit Survival Score (BSS).

    The candidate block is tested under small implementation-level disturbances:
        1. exact candidate block A'
        2. rounded candidate block round(A')
        3. positive coefficient drift A' + 0.25
        4. negative coefficient drift A' - 0.25

    BSS = number of tests that still extract the correct bit / total tests.

    MSE is computed separately and is used as the distortion term.
    """
    mse = float(np.mean((block0 - block_cand) ** 2))
    base = block_cand.astype(np.float64)

    stress_tests = [
        ("exact", base),
        ("rounded", np.rint(base)),
        ("positive_drift", base + 0.25),
        ("negative_drift", base - 0.25),
    ]

    if not FAST_CANDIDATE_SCORING:
        stress_tests.extend([
            ("scale_up", 1.01 * base),
            ("scale_down", 0.99 * base),
            ("strong_positive_drift", base + 0.50),
            ("strong_negative_drift", base - 0.50),
        ])

    pass_count = 0
    total_count = 0

    for _, test_block in stress_tests:
        total_count += 1
        try:
            extracted_bit = _extract_bit_from_candidate_block(
                test_block,
                mode,
                q_h01,
                pos=pos,
            )
            if int(extracted_bit) == int(bit):
                pass_count += 1
        except Exception:
            pass

    survival_rate = float(pass_count / max(total_count, 1))
    ok = survival_rate >= float(MIN_SURVIVAL_RATE)

    # Keep these compatibility fields. In the new engineering score,
    # raw/avg/worst all represent the same direct bit survival rate.
    raw_score = survival_rate
    avg_score = survival_rate
    worst_score = survival_rate

    return survival_rate, mse, raw_score, avg_score, worst_score, ok


def _candidate_mse_limit(cand: dict) -> float:
    if int(cand["flag"]) == FLAG_Q4:
        return max(float(MAX_Q_CAND_MSE), EPS_CONF)
    if int(cand["flag"]) == FLAG_HPOS:
        return max(float(MAX_H_CAND_MSE), EPS_CONF)
    return 1.0


def _combined_selection_score(cand: dict) -> float:
    """
    Final Q-vs-H engineering score.

    The bit survival score rewards readability. The normalized MSE penalizes
    distortion. This gives a direct engineering trade-off:

        score = BSS_WEIGHT * BSS - MSE_WEIGHT * normalized_MSE
    """
    bss = float(cand["score"])
    mse = float(cand["mse"])
    mse_limit = _candidate_mse_limit(cand)
    normalized_mse = min(mse / mse_limit, 2.0)
    return float(BSS_WEIGHT) * bss - float(MSE_WEIGHT) * normalized_mse


def _select_value(c):
    return _combined_selection_score(c)


def _candidate_rank_key(c):
    return (
        _combined_selection_score(c),
        float(c["score"]),
        -float(c["mse"]),
    )


def _candidate_is_strong_enough(cand: dict):
    if not bool(cand["ok"]):
        return False

    if float(cand["score"]) < float(MIN_SURVIVAL_RATE):
        return False

    if int(cand["flag"]) == FLAG_Q4:
        if float(cand["mse"]) > float(MAX_Q_CAND_MSE):
            return False

    elif int(cand["flag"]) == FLAG_HPOS:
        if float(cand["mse"]) > float(MAX_H_CAND_MSE):
            return False

    return True


def _best_hpos_candidate(block0: np.ndarray, bit: int, q_h01: float, h01_margin: float):
    """
    H-candidate selection rule.

    First, build both H candidates h21 and h22.
    Then choose the valid H candidate with the lowest MSE.
    If no H candidate passes the survival/MSE constraints, return the lowest-MSE
    H candidate anyway; it will be rejected later by _candidate_is_strong_enough().
    """
    all_h = []
    valid_h = []

    for hpos_idx, (hname, pos) in enumerate(HPOS_CANDIDATES):
        block_h = _build_hpos_candidate(block0, bit, q_h01, h01_margin, pos)
        score_h, mse_h, raw_h, avg_h, worst_h, ok_h = _candidate_attack_score(
            block0, block_h, FLAG_HPOS, bit, q_h01, pos=pos
        )
        cand = {
            "flag": FLAG_HPOS,
            "block": block_h,
            "score": score_h,
            "mse": mse_h,
            "raw": raw_h,
            "avg": avg_h,
            "worst": worst_h,
            "ok": ok_h,
            "hpos_idx": hpos_idx,
            "hpos_name": hname,
            "pos": pos,
        }

        all_h.append(cand)

        if _candidate_is_strong_enough(cand):
            valid_h.append(cand)

    if valid_h:
        return min(valid_h, key=lambda c: float(c["mse"]))

    return min(all_h, key=lambda c: float(c["mse"]))


# =========================================================
# Parameter snapshot/application
# =========================================================
def get_current_params():
    return {
        "ARNOLD_ITERATIONS": int(ARNOLD_ITERATIONS),
        "DWT_BANDS": tuple(DWT_BANDS),
        "Q4_TAU": float(Q4_TAU),
        "Q4_MARGIN": float(Q4_MARGIN),
        "H01_Q": float(H01_Q),
        "H01_MARGIN": float(H01_MARGIN),
        "MIN_SURVIVAL_RATE": float(MIN_SURVIVAL_RATE),
        "BSS_WEIGHT": float(BSS_WEIGHT),
        "MSE_WEIGHT": float(MSE_WEIGHT),
        "MAX_Q_CAND_MSE": float(MAX_Q_CAND_MSE),
        "MAX_H_CAND_MSE": float(MAX_H_CAND_MSE),
        "Q4_GIVENS_THETA_MAX": float(Q4_GIVENS_THETA_MAX),
        "Q4_GIVENS_MSE_WEIGHT": float(Q4_GIVENS_MSE_WEIGHT),
        "Q4_GIVENS_EXTRA_MARGIN_WEIGHT": float(Q4_GIVENS_EXTRA_MARGIN_WEIGHT),
    }


def apply_params(params: dict):
    global ARNOLD_ITERATIONS, DWT_BANDS
    global Q4_TAU, Q4_MARGIN, H01_Q, H01_MARGIN
    global MIN_SURVIVAL_RATE, BSS_WEIGHT, MSE_WEIGHT, MAX_Q_CAND_MSE, MAX_H_CAND_MSE
    global Q4_GIVENS_THETA_MAX, Q4_GIVENS_MSE_WEIGHT, Q4_GIVENS_EXTRA_MARGIN_WEIGHT

    if "ARNOLD_ITERATIONS" in params:
        ARNOLD_ITERATIONS = int(round(params["ARNOLD_ITERATIONS"]))
    if "DWT_BANDS" in params:
        DWT_BANDS = tuple(params["DWT_BANDS"])
    if "Q4_TAU" in params:
        Q4_TAU = float(params["Q4_TAU"])
    if "Q4_MARGIN" in params:
        Q4_MARGIN = float(params["Q4_MARGIN"])
    if "H01_Q" in params:
        H01_Q = float(params["H01_Q"])
    if "H01_MARGIN" in params:
        H01_MARGIN = float(params["H01_MARGIN"])
    if "MIN_SURVIVAL_RATE" in params:
        MIN_SURVIVAL_RATE = float(params["MIN_SURVIVAL_RATE"])
    if "BSS_WEIGHT" in params:
        BSS_WEIGHT = float(params["BSS_WEIGHT"])
    if "MSE_WEIGHT" in params:
        MSE_WEIGHT = float(params["MSE_WEIGHT"])
    if "MAX_Q_CAND_MSE" in params:
        MAX_Q_CAND_MSE = float(params["MAX_Q_CAND_MSE"])
    if "MAX_H_CAND_MSE" in params:
        MAX_H_CAND_MSE = float(params["MAX_H_CAND_MSE"])
    if "Q4_GIVENS_THETA_MAX" in params:
        Q4_GIVENS_THETA_MAX = float(params["Q4_GIVENS_THETA_MAX"])
    if "Q4_GIVENS_MSE_WEIGHT" in params:
        Q4_GIVENS_MSE_WEIGHT = float(params["Q4_GIVENS_MSE_WEIGHT"])
    if "Q4_GIVENS_EXTRA_MARGIN_WEIGHT" in params:
        Q4_GIVENS_EXTRA_MARGIN_WEIGHT = float(params["Q4_GIVENS_EXTRA_MARGIN_WEIGHT"])


def params_to_jsonable(params):
    return _to_json_safe(params)


# =========================================================
# Main embedding
# =========================================================
def _embed_main_in_image(
    host_img_orig,
    watermark_path,
    embed_info_path,
    arnold_iterations=None,
):
    if arnold_iterations is None:
        arnold_iterations = ARNOLD_ITERATIONS

    if host_img_orig.ndim != 3 or host_img_orig.shape[2] != 3:
        raise ValueError("Host image must be a 24-bit color image (3 channels).")

    H_img, W_img = host_img_orig.shape[:2]

    # Requested change: embed only in the Y channel.
    # Convert BGR host image to YCrCb, process channel 0 (Y), then convert back.
    host_embed_orig = _host_bgr_to_embed_space(host_img_orig)

    bands_by_ch = {}
    H0, W0 = None, None

    for ch in HOST_CHANNELS:
        work, h0, w0 = _crop_for_dwt(host_embed_orig[:, :, ch], DWT_LEVEL, BLOCKSIZE)
        if H0 is None:
            H0, W0 = h0, w0
        bands_by_ch[ch] = _dwt_split_4bands(work, DWT_WAVELET)

    wm_binary, wm_bits, wm_meta = prepare_binary_watermark_payload(
        watermark_path,
        WM_SIZE,
        arnold_iterations=arnold_iterations,
    )

    L = len(wm_bits)

    total_blocks = 0
    for ch in HOST_CHANNELS:
        total_blocks += sum(
            ((bands_by_ch[ch][b].shape[0] // BLOCKSIZE) * (bands_by_ch[ch][b].shape[1] // BLOCKSIZE))
            for b in DWT_BANDS
        )

    if total_blocks < L:
        raise ValueError(
            f"Capacity insufficient: need {L} watermark payload bits but only {total_blocks} host blocks are available."
        )

    blocks = generate_color_subband_block_indices(
        bands_by_ch,
        HOST_CHANNELS,
        DWT_BANDS,
        BLOCKSIZE,
        total_blocks,
        PRIVATE_KEY,
    )

    if USE_STRUCTURED_REPETITION:
        schedule, repeat_factor, usable_blocks = make_structured_repetition_schedule(blocks, L)
    else:
        schedule = [(kpos, blocks[kpos]) for kpos in range(L)]
        repeat_factor = 1
        usable_blocks = L

    q_h01 = max(float(H01_Q), 1e-6)
    h01_margin = min(max(0.0, float(H01_MARGIN)), 0.49 * q_h01)

    _debug(f"[MAIN-EMBED] original_size={W_img}x{H_img}, working_size={W0}x{H0}")
    _debug(f"[MAIN-EMBED] color_space={HOST_EMBED_COLOR_SPACE}, channels={HOST_CHANNELS}, DWT={DWT_WAVELET}, bands={DWT_BANDS}")
    _debug(f"[MAIN-EMBED] watermark size={wm_binary.shape[1]}x{wm_binary.shape[0]}, payload_bits={L}")
    _debug(f"[MAIN-EMBED] Arnold iterations={arnold_iterations}")
    _debug(f"[MAIN-EMBED] total_blocks={total_blocks}, usable={usable_blocks}, repeat={repeat_factor}x")

    flag_list = []
    hpos_list = []

    support_counts = np.zeros(L, dtype=np.int32)
    support_sel_sum = np.zeros(L, dtype=np.float64)
    mode_band_counter = {}

    for idx, (kpos, block_pos) in enumerate(schedule):
        ch, band_name, i, j = block_pos
        bit = int(wm_bits[kpos])
        band = bands_by_ch[ch][band_name]
        block0 = band[i:i + BLOCKSIZE, j:j + BLOCKSIZE].copy()

        try:
            block_q4 = _build_q4_candidate(block0, bit)
            score_q4, mse_q4, raw_q4, avg_q4, worst_q4, ok_q4 = _candidate_attack_score(
                block0,
                block_q4,
                FLAG_Q4,
                bit,
                q_h01,
            )

            q_cand = {
                "flag": FLAG_Q4,
                "block": block_q4,
                "score": score_q4,
                "mse": mse_q4,
                "raw": raw_q4,
                "avg": avg_q4,
                "worst": worst_q4,
                "ok": ok_q4,
                "hpos_idx": HPOS_NONE,
                "hpos_name": "none",
            }

            best_h = _best_hpos_candidate(block0, bit, q_h01, h01_margin)

            cands = [q_cand, best_h]
            strong_cands = [c for c in cands if _candidate_is_strong_enough(c)]

            if strong_cands:
                strong_cands = sorted(strong_cands, key=_candidate_rank_key, reverse=True)
                best = strong_cands[0]

                bands_by_ch[ch][band_name][i:i + BLOCKSIZE, j:j + BLOCKSIZE] = best["block"]

                flag_list.append(int(best["flag"]))
                hpos_list.append(int(best.get("hpos_idx", HPOS_NONE)))

                support_counts[kpos] += 1
                support_sel_sum[kpos] += float(_select_value(best))

                mode_name = "Q" if int(best["flag"]) == FLAG_Q4 else f"H:{best['hpos_name']}"
                mode_band_counter[(ch, band_name, mode_name)] = (
                    mode_band_counter.get((ch, band_name, mode_name), 0) + 1
                )

            else:
                flag_list.append(FLAG_SKIP)
                hpos_list.append(HPOS_NONE)

        except Exception:
            flag_list.append(FLAG_SKIP)
            hpos_list.append(HPOS_NONE)

    _print_embed_support_summary(
        support_counts,
        support_sel_sum,
        "structured",
        target_support=repeat_factor,
    )
    _print_mode_band_summary(mode_band_counter, "structured")

    watermarked_embed = host_embed_orig.copy()

    for ch in HOST_CHANNELS:
        rec_channel = _dwt_merge_4bands(bands_by_ch[ch], DWT_WAVELET)
        rec_channel = rec_channel[:H0, :W0]

        out_channel = host_embed_orig[:, :, ch].astype(np.float64).copy()
        out_channel[:H0, :W0] = np.clip(np.rint(rec_channel), 0, 255)
        watermarked_embed[:, :, ch] = out_channel.astype(np.uint8)

    watermarked_img = _embed_space_to_host_bgr(watermarked_embed)

    q4_used, hpos_used, skip_used = _flag_stats(flag_list)

    wm_meta["structured_repetition_enabled"] = bool(USE_STRUCTURED_REPETITION)
    wm_meta["structured_repeat_factor"] = int(repeat_factor)
    wm_meta["structured_usable_blocks"] = int(usable_blocks)
    wm_meta["dwt_bands"] = list(DWT_BANDS)
    wm_meta["host_embed_color_space"] = str(HOST_EMBED_COLOR_SPACE)
    wm_meta["host_embed_channels"] = list(HOST_CHANNELS)
    wm_meta["host_embed_channel_description"] = "Y channel only after BGR_to_YCrCb"
    wm_meta["q4_used"] = int(q4_used)
    wm_meta["hpos_used"] = int(hpos_used)
    wm_meta["skip_used"] = int(skip_used)
    wm_meta["total_flags"] = int(len(flag_list))
    wm_meta["parameter_snapshot"] = params_to_jsonable(get_current_params())
    wm_meta["block_hash"] = "SHA-256"
    wm_meta["h_candidates"] = [name for name, _ in HPOS_CANDIDATES]
    wm_meta["weighted_voting_removed"] = True
    wm_meta["confidence_scoring_removed"] = True
    wm_meta["bit_survival_score"] = "BSS = passed_stress_tests / total_stress_tests"
    wm_meta["logistic_removed"] = True
    wm_meta["rsa_removed"] = True

    _save_embed_info(embed_info_path, flag_list, hpos_list, wm_meta)

    _debug(
        f"[MAIN-EMBED] Q={q4_used}, H={hpos_used}, SKIP={skip_used}, total={len(flag_list)}"
    )

    return watermarked_img


def embed_full(
    host_path,
    watermark_path,
    output_path,
    embed_info_path,
    arnold_iterations=None,
):
    if arnold_iterations is None:
        arnold_iterations = ARNOLD_ITERATIONS

    host_img_orig = load_color_image_24bit_safe(host_path, size=None)
    if host_img_orig.shape[0] != HOST_SIZE or host_img_orig.shape[1] != HOST_SIZE:
        raise ValueError(
            f"Host image must be exactly {HOST_SIZE}x{HOST_SIZE}. "
            f"Got {host_img_orig.shape[1]}x{host_img_orig.shape[0]}."
        )

    _debug(f"[FULL-EMBED] host='{os.path.basename(host_path)}'")

    img_final = _embed_main_in_image(
        host_img_orig,
        watermark_path,
        embed_info_path,
        arnold_iterations=arnold_iterations,
    )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(output_path, img_final)
    _debug(f"[FULL-EMBED] output='{output_path}'")
    return img_final


# =========================================================
# Simple threshold extraction
# =========================================================
def _extract_main_from_work_color(work_img_u8: np.ndarray, flags, hpos_list, schedule, q_h01, payload_len):
    """
    Simple threshold extraction.

    Q branch:
        bit = 1 if q22^2 + q32^2 >= tau_Q else 0

    H branch:
        bit = 1 if |h_ij| mod q >= q/2 else 0

    If structured repetition is enabled, several votes can exist for one
    watermark bit. The final bit is decided by ordinary majority voting.
    """
    bands_by_ch = {}

    for ch in HOST_CHANNELS:
        bands_by_ch[ch] = _dwt_split_4bands(
            work_img_u8[:, :, ch].astype(np.float64),
            DWT_WAVELET,
        )

    L = int(payload_len)
    n = min(len(schedule), len(flags), len(hpos_list))

    q4_used, hpos_used, skip_used = _flag_stats(flags[:n])
    _debug(
        f"[MAIN-EXTRACT-INTERNAL] schedule={n}, Q={q4_used}, H={hpos_used}, "
        f"SKIP={skip_used}, payload_bits={L}"
    )

    bit_votes = [[] for _ in range(L)]

    for idx in range(n):
        kpos, block_pos = schedule[idx]
        ch, band_name, i, j = block_pos
        flag = int(flags[idx])

        if flag == FLAG_SKIP:
            continue

        block = bands_by_ch[ch][band_name][i:i + BLOCKSIZE, j:j + BLOCKSIZE]

        try:
            Hh, Qm = linalg.hessenberg(block, calc_q=True)

            if flag == FLAG_Q4:
                extracted_bit = 1 if _q4_stat(Qm) >= float(Q4_TAU) else 0

            elif flag == FLAG_HPOS:
                hpos_idx = int(hpos_list[idx]) if idx < len(hpos_list) else HPOS_NONE
                if not (0 <= hpos_idx < len(HPOS_CANDIDATES)):
                    continue

                _, pos = HPOS_CANDIDATES[hpos_idx]
                rr, cc = pos
                z = abs(float(Hh[rr, cc])) % float(q_h01)
                extracted_bit = 1 if z >= 0.5 * float(q_h01) else 0

            else:
                continue

            bit_votes[kpos].append(int(extracted_bit))

        except Exception:
            continue

    out_bits = np.zeros(L, dtype=np.uint8)
    valid_count = 0
    total_votes = 0

    for kpos in range(L):
        votes = bit_votes[kpos]

        if len(votes) == 0:
            out_bits[kpos] = 0
            continue

        valid_count += 1
        total_votes += len(votes)

        ones = int(np.sum(votes))
        zeros = len(votes) - ones
        out_bits[kpos] = 1 if ones >= zeros else 0

    coverage = float(valid_count / L) if L > 0 else 0.0
    avg_votes_per_valid_bit = float(total_votes / valid_count) if valid_count > 0 else 0.0

    _debug(
        f"[MAIN-EXTRACT-INTERNAL] coverage={coverage:.6f}, "
        f"avg_votes_per_valid_bit={avg_votes_per_valid_bit:.3f}"
    )

    return out_bits, coverage


def _extract_main_from_image_array(wm_img: np.ndarray, embed_info_path: str):
    flags, hpos_list, payload_meta = _load_embed_info(embed_info_path)

    if len(flags) == 0:
        raise ValueError(f"Embed info is empty: {embed_info_path}")

    payload_len = _payload_len_from_meta(payload_meta)
    dwt_bands_for_meta = tuple(payload_meta.get("dwt_bands", DWT_BANDS))

    # Requested change: extraction must read the same Y channel that was used
    # for embedding. The watermarked/attacked image arrives as BGR, so convert
    # to YCrCb before DWT extraction.
    embed_img = _host_bgr_to_embed_space(wm_img)

    _, H0, W0 = _crop_for_dwt(embed_img[:, :, HOST_CHANNELS[0]], DWT_LEVEL, BLOCKSIZE)
    work_raw_u8 = embed_img[:H0, :W0, :].copy()

    bands_by_ch = {}
    for ch in HOST_CHANNELS:
        bands_by_ch[ch] = _dwt_split_4bands(work_raw_u8[:, :, ch].astype(np.float64), DWT_WAVELET)

    total_blocks = 0
    for ch in HOST_CHANNELS:
        total_blocks += sum(
            ((bands_by_ch[ch][b].shape[0] // BLOCKSIZE) * (bands_by_ch[ch][b].shape[1] // BLOCKSIZE))
            for b in dwt_bands_for_meta
        )

    blocks = generate_color_subband_block_indices(
        bands_by_ch,
        HOST_CHANNELS,
        dwt_bands_for_meta,
        BLOCKSIZE,
        total_blocks,
        PRIVATE_KEY,
    )

    if payload_meta.get("structured_repetition_enabled", True):
        schedule, repeat_factor, usable_blocks = make_structured_repetition_schedule(blocks, payload_len)
    else:
        schedule = [(kpos, blocks[kpos]) for kpos in range(payload_len)]
        repeat_factor = 1
        usable_blocks = payload_len

    _debug(
        f"[MAIN-EXTRACT] repeat_factor={repeat_factor}, usable_blocks={usable_blocks}, payload_len={payload_len}"
    )

    q_h01 = max(float(H01_Q), 1e-6)

    best_bits, coverage = _extract_main_from_work_color(
        work_raw_u8,
        flags,
        hpos_list,
        schedule,
        q_h01,
        payload_len,
    )

    wm_rec = reconstruct_binary_watermark_from_payload_bits(best_bits, payload_meta)
    wm_rec = _force_binary_watermark_exact(wm_rec, int(payload_meta["wm_size"]), thresh=WM_BIN_THRESH)

    return wm_rec, coverage, payload_meta


def extract_main(
    watermarked_path,
    embed_info_path,
    output_path=None,
):
    wm_img = load_color_image_24bit_safe(watermarked_path, size=None)
    wm_rec, coverage, payload_meta = _extract_main_from_image_array(wm_img, embed_info_path)

    if output_path is not None:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(output_path, wm_rec)

    return wm_rec, coverage, payload_meta



# =========================================================
# WAVES adapter compatibility helpers
# =========================================================
def force_binary_watermark_exact(img_u8: np.ndarray, size: int = None, thresh: int = WM_BIN_THRESH) -> np.ndarray:
    """Public alias used by older scripts/tests."""
    if size is None:
        size = WM_SIZE
    return _force_binary_watermark_exact(img_u8, int(size), thresh=thresh)


def watermark_bits_from_file(path: str) -> np.ndarray:
    """Return the original 64x64 binary watermark as a flat bool vector.

    The new proposal embeds only the watermark-DWT LL payload internally, but
    WAVES metrics should still compare the reconstructed full watermark.
    """
    wm = load_binary_watermark_8bit_safe(path, size=WM_SIZE)
    wm = _force_binary_watermark_exact(wm, WM_SIZE, thresh=WM_BIN_THRESH)
    return (wm >= WM_BIN_THRESH).astype(bool).reshape(-1)


def extract_message_bits(image_path: str, embed_info_path: str) -> np.ndarray:
    """Decode an image and return the reconstructed full watermark as bool bits."""
    wm_rec, _, _ = extract_main(image_path, embed_info_path, output_path=None)
    wm_rec = _force_binary_watermark_exact(wm_rec, WM_SIZE, thresh=WM_BIN_THRESH)
    return (wm_rec >= WM_BIN_THRESH).astype(bool).reshape(-1)


# =========================================================
# Metrics
# =========================================================
def _watermark_metrics(extracted_watermark_bin, original_watermark_bin):
    ori = _force_binary_watermark_exact(original_watermark_bin, WM_SIZE, thresh=WM_BIN_THRESH)
    ext = _force_binary_watermark_exact(extracted_watermark_bin, WM_SIZE, thresh=WM_BIN_THRESH)

    if ori.shape != (WM_SIZE, WM_SIZE) or ext.shape != (WM_SIZE, WM_SIZE):
        raise ValueError(f"Watermark metrics require exact {WM_SIZE}x{WM_SIZE} binary watermarks.")

    ori_bits = (ori >= WM_BIN_THRESH).astype(np.uint8).reshape(-1)
    ext_bits = (ext >= WM_BIN_THRESH).astype(np.uint8).reshape(-1)

    ori_vec = ori_bits.astype(np.float64)
    ext_vec = ext_bits.astype(np.float64)

    sum_ori_sq = np.sum(ori_vec ** 2)
    sum_ext_sq = np.sum(ext_vec ** 2)
    if sum_ori_sq == 0 or sum_ext_sq == 0:
        nc_value = 1.0 if sum_ori_sq == sum_ext_sq else 0.0
    else:
        numerator = np.sum(ori_vec * ext_vec)
        denominator = np.sqrt(sum_ori_sq * sum_ext_sq)
        nc_value = numerator / denominator if denominator != 0 else 0.0

    ori_center = ori_vec - np.mean(ori_vec)
    ext_center = ext_vec - np.mean(ext_vec)
    denom_ncc = np.sqrt(np.sum(ori_center ** 2) * np.sum(ext_center ** 2))
    if denom_ncc == 0:
        ncc_value = 1.0 if np.array_equal(ori_bits, ext_bits) else 0.0
    else:
        ncc_value = float(np.sum(ori_center * ext_center) / denom_ncc)

    ber_value = float(np.mean(ori_bits != ext_bits))

    return float(nc_value), float(ncc_value), float(ber_value)


def calculate_metrics(original, watermarked, extracted_watermark_bin, original_watermark_bin):
    psnr_value, ssim_value = float("nan"), float("nan")
    nc_value, ncc_value, ber_value = float("nan"), float("nan"), float("nan")

    try:
        if original is not None and watermarked is not None and original.shape == watermarked.shape:
            ori_f = original.astype(np.float64)
            wm_f = watermarked.astype(np.float64)
            mse = np.mean((ori_f - wm_f) ** 2)
            psnr_value = float("inf") if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
            ssim_value = ssim(
                original.astype(np.uint8),
                watermarked.astype(np.uint8),
                data_range=255,
                channel_axis=-1,
            )
    except Exception as e:
        print(f"Metrics Error (PSNR/SSIM): {e}")
        traceback.print_exc()

    try:
        if original_watermark_bin is not None and extracted_watermark_bin is not None:
            nc_value, ncc_value, ber_value = _watermark_metrics(
                extracted_watermark_bin,
                original_watermark_bin,
            )
    except Exception as e:
        print(f"Metrics Error (NC/NCC/BER): {e}")
        traceback.print_exc()

    return psnr_value, ssim_value, nc_value, ncc_value, ber_value


# =========================================================
# Attack implementations
# =========================================================
def _ensure_uint8_bgr(img):
    img = np.asarray(img)
    if img.ndim == 2:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Image must be BGR with 3 channels.")
    return np.clip(img, 0, 255).astype(np.uint8)


def attack_blur(img, sigma):
    img = _ensure_uint8_bgr(img)
    sigma = float(sigma)
    if sigma <= 0:
        return img.copy()
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)


def attack_sharpen(img, params):
    img = _ensure_uint8_bgr(img)
    sigma, amount = params
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    sharp = cv2.addWeighted(img, 1.0 + float(amount), blurred, -float(amount), 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def attack_speckle_noise(img, params, rng=None):
    img = _ensure_uint8_bgr(img)
    mean, var = params
    if rng is None:
        rng = np.random.default_rng(123)
    img_f = img.astype(np.float64) / 255.0
    noise = rng.normal(float(mean), math.sqrt(max(float(var), 0.0)), img_f.shape)
    out = img_f + img_f * noise
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def attack_salt_pepper(img, amount, rng=None):
    img = _ensure_uint8_bgr(img)
    amount = float(amount)
    if rng is None:
        rng = np.random.default_rng(123)
    out = img.copy()
    h, w = out.shape[:2]
    total = h * w
    num = int(total * amount)
    if num <= 0:
        return out
    coords = rng.choice(total, size=min(num, total), replace=False)
    ys = coords // w
    xs = coords % w
    half = len(coords) // 2
    out[ys[:half], xs[:half], :] = 0
    out[ys[half:], xs[half:], :] = 255
    return out


def attack_jpeg(img, quality):
    img = _ensure_uint8_bgr(img)
    quality = int(quality)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed.")
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if dec is None:
        raise RuntimeError("JPEG decoding failed.")
    return dec.astype(np.uint8)


def attack_jpeg2000(img, compression_value):
    img = _ensure_uint8_bgr(img)
    compression_value = float(compression_value)
    try:
        params = []
        if hasattr(cv2, "IMWRITE_JPEG2000_COMPRESSION_X1000"):
            params = [int(cv2.IMWRITE_JPEG2000_COMPRESSION_X1000), int(max(1, round(compression_value * 100)))]
        ok, enc = cv2.imencode(".jp2", img, params)
        if not ok:
            raise RuntimeError("JPEG2000 encoding failed.")
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if dec is None:
            raise RuntimeError("JPEG2000 decoding failed.")
        return dec.astype(np.uint8)
    except Exception as e:
        print(f"[WARN] JPEG2000 failed, using unchanged image fallback: {e}")
        return img.copy()


def attack_lowpass(img, ksize):
    img = _ensure_uint8_bgr(img)
    if isinstance(ksize, tuple):
        kx, ky = int(ksize[0]), int(ksize[1])
    else:
        kx = ky = int(ksize)
    kx = max(1, kx)
    ky = max(1, ky)
    return cv2.blur(img, (kx, ky))


def attack_scale(img, scale_factor):
    img = _ensure_uint8_bgr(img)
    scale_factor = float(scale_factor)
    if scale_factor <= 0:
        raise ValueError("Scale factor must be positive.")
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale_factor)))
    new_h = max(1, int(round(h * scale_factor)))
    interp1 = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_CUBIC
    interp2 = cv2.INTER_LINEAR if scale_factor < 1.0 else cv2.INTER_AREA
    scaled = cv2.resize(img, (new_w, new_h), interpolation=interp1)
    restored = cv2.resize(scaled, (w, h), interpolation=interp2)
    return restored.astype(np.uint8)


def attack_rotation(img, angle):
    img = _ensure_uint8_bgr(img)
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return rotated.astype(np.uint8)


def attack_histogram(img, _unused=None):
    img = _ensure_uint8_bgr(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return out.astype(np.uint8)


def attack_occlusion(img, params, rng=None):
    img = _ensure_uint8_bgr(img)
    if rng is None:
        rng = np.random.default_rng(123)
    size, x_spec, y_spec, num_blocks = params
    size = int(round(float(size)))
    num_blocks = int(num_blocks)
    h, w = img.shape[:2]
    size = max(1, min(size, h, w))
    out = img.copy()

    for _ in range(num_blocks):
        if x_spec == "random":
            x = int(rng.integers(0, max(1, w - size + 1)))
        else:
            x = int(x_spec)

        if y_spec == "random":
            y = int(rng.integers(0, max(1, h - size + 1)))
        else:
            y = int(y_spec)

        x = max(0, min(x, w - size))
        y = max(0, min(y, h - size))
        out[y:y + size, x:x + size, :] = 0

    return out.astype(np.uint8)


def apply_attack(img, attack_name, attack_param, seed=123):
    img = _ensure_uint8_bgr(img)
    rng = np.random.default_rng(seed)
    name = str(attack_name).lower().strip()

    if name == "blur":
        return attack_blur(img, attack_param)
    if name == "sharpen":
        return attack_sharpen(img, attack_param)
    if name == "speckle noise":
        return attack_speckle_noise(img, attack_param, rng=rng)
    if name == "salt & pepper":
        return attack_salt_pepper(img, attack_param, rng=rng)
    if name == "jpeg":
        return attack_jpeg(img, attack_param)
    if name == "jpeg2000":
        return attack_jpeg2000(img, attack_param)
    if name == "lowpass":
        return attack_lowpass(img, attack_param)
    if name == "scale":
        return attack_scale(img, attack_param)
    if name == "rotation":
        return attack_rotation(img, attack_param)
    if name == "histogram":
        return attack_histogram(img, attack_param)
    if name == "occlusion":
        return attack_occlusion(img, attack_param, rng=rng)

    raise ValueError(f"Unknown attack: {attack_name}")


def attack_label(attack_name, attack_param):
    return _safe_filename(f"{attack_name}_{attack_param}")


def constrained_psnr_nc_objective(
    clean_psnr,
    mean_attack_nc,
    psnr_threshold=PSNR_MIN_THRESHOLD,
):
    try:
        clean_psnr = float(clean_psnr)
        if not np.isfinite(clean_psnr):
            clean_psnr = 0.0
    except Exception:
        clean_psnr = 0.0

    try:
        mean_attack_nc = float(mean_attack_nc)
        if not np.isfinite(mean_attack_nc):
            mean_attack_nc = 0.0
    except Exception:
        mean_attack_nc = 0.0

    psnr_threshold = float(psnr_threshold)
    psnr_feasible = clean_psnr > psnr_threshold

    if psnr_feasible:
        objective = 1.0 + mean_attack_nc
        psnr_gap = 0.0
    else:
        psnr_gap = max(0.0, psnr_threshold - clean_psnr)
        objective = -(psnr_gap / max(psnr_threshold, EPS_CONF)) + 0.001 * mean_attack_nc

    return float(objective), bool(psnr_feasible), float(psnr_gap)


# =========================================================
# Firefly parameter decoding
# =========================================================
PARAM_SPECS = [
    # Requested change: Firefly optimization tunes only these four parameters.
    # All other algorithm parameters stay at their global default values.
    {"name": "Q4_TAU", "type": "float", "min": 0.35, "max": 0.65},
    {"name": "Q4_MARGIN", "type": "float", "min": 0.04, "max": 0.14},
    {"name": "H01_Q", "type": "float", "min": 5.0, "max": 10.0},
    {"name": "H01_MARGIN", "type": "float", "min": 0.50, "max": 1.20},
]


def decode_firefly_position(position):
    position = np.clip(np.asarray(position, dtype=np.float64), 0.0, 1.0)
    params = {}

    for idx, spec in enumerate(PARAM_SPECS):
        z = float(position[idx])
        name = spec["name"]
        typ = spec["type"]

        if typ == "float":
            value = float(spec["min"] + z * (spec["max"] - spec["min"]))
            params[name] = value
        elif typ == "int":
            value = int(round(spec["min"] + z * (spec["max"] - spec["min"])))
            value = max(int(spec["min"]), min(int(spec["max"]), value))
            params[name] = value
        elif typ == "choice":
            choices = list(spec["choices"])
            cidx = int(round(z * (len(choices) - 1)))
            cidx = max(0, min(len(choices) - 1, cidx))
            params[name] = choices[cidx]
        else:
            raise ValueError(f"Unknown parameter type: {typ}")

    # H01_MARGIN must remain inside the usable interval for the H-domain rule.
    params["H01_MARGIN"] = min(float(params["H01_MARGIN"]), 0.49 * float(params["H01_Q"]))

    return params


def firefly_cache_key(position, digits=4):
    arr = np.round(np.clip(np.asarray(position, dtype=np.float64), 0.0, 1.0), digits)
    return tuple(float(x) for x in arr.tolist())


# =========================================================
# Evaluation for one parameter candidate and one image
# =========================================================
def evaluate_candidate_for_image(
    params,
    host_path,
    watermark_path,
    trial_dir,
    attack_suite=FAST_ATTACK_SUITE,
    seed=123,
    save_images=False,
    verbose=False,
):
    os.makedirs(trial_dir, exist_ok=True)
    apply_params(params)

    base_name = os.path.splitext(os.path.basename(host_path))[0]
    original_img = load_color_image_24bit_safe(host_path, size=None)
    original_wm_binary = load_binary_watermark_8bit_safe(watermark_path, WM_SIZE)
    original_wm_binary = _force_binary_watermark_exact(original_wm_binary, WM_SIZE, thresh=WM_BIN_THRESH)

    watermarked_path = os.path.join(trial_dir, f"watermarked_{base_name}.png")
    embed_info_path = os.path.join(trial_dir, f"embed_info_{base_name}.npz")
    clean_extract_path = os.path.join(trial_dir, f"extract_clean_{base_name}.png")

    result = {
        "host_image": os.path.basename(host_path),
        "params": params_to_jsonable(params),
        "objective": -1e9,

        "clean_psnr": 0.0,
        "clean_ssim": 0.0,
        "clean_nc": 0.0,
        "clean_ncc": 0.0,
        "clean_ber": 1.0,
        "clean_coverage": None,

        "mean_attack_nc": 0.0,
        "min_attack_nc": 0.0,

        "psnr_threshold": float(PSNR_MIN_THRESHOLD),
        "psnr_feasible": False,
        "psnr_gap": None,

        "attack_results": [],

        "q4_used": None,
        "hpos_used": None,
        "skip_used": None,
        "total_flags": None,
        "repeat_factor": None,

        "embed_time_sec": None,
        "clean_extract_time_sec": None,

        "error": None,
    }

    try:
        t0 = time.time()
        watermarked_img = embed_full(
            host_path,
            watermark_path,
            watermarked_path,
            embed_info_path,
            arnold_iterations=int(params.get("ARNOLD_ITERATIONS", ARNOLD_ITERATIONS)),
        )
        embed_time = time.time() - t0

        t1 = time.time()
        extracted_clean, clean_coverage, clean_meta = extract_main(
            watermarked_path,
            embed_info_path,
            clean_extract_path,
        )
        clean_extract_time = time.time() - t1

        psnr_val, ssim_val, clean_nc, clean_ncc, clean_ber = calculate_metrics(
            original_img,
            watermarked_img,
            extracted_clean,
            original_wm_binary,
        )

        flags, hpos, meta = _load_embed_info(embed_info_path)
        q4_used, hpos_used, skip_used = _flag_stats(flags)

        result["clean_psnr"] = _json_safe_number(psnr_val) or 0.0
        result["clean_ssim"] = _json_safe_number(ssim_val) or 0.0
        result["clean_nc"] = _json_safe_number(clean_nc) or 0.0
        result["clean_ncc"] = _json_safe_number(clean_ncc) or 0.0
        result["clean_ber"] = _json_safe_number(clean_ber) if _json_safe_number(clean_ber) is not None else 1.0
        result["clean_coverage"] = _json_safe_number(clean_coverage)
        result["embed_time_sec"] = _json_safe_number(embed_time)
        result["clean_extract_time_sec"] = _json_safe_number(clean_extract_time)
        result["q4_used"] = int(q4_used)
        result["hpos_used"] = int(hpos_used)
        result["skip_used"] = int(skip_used)
        result["total_flags"] = int(len(flags))
        result["repeat_factor"] = int(meta.get("structured_repeat_factor", -1))

        attack_ncs = []

        for attack_idx, (attack_name, attack_param) in enumerate(attack_suite):
            label = attack_label(attack_name, attack_param)
            attacked_img_path = os.path.join(trial_dir, f"attacked_{base_name}_{label}.png")
            extracted_attack_path = os.path.join(trial_dir, f"extract_{base_name}_{label}.png")

            attack_record = {
                "attack_name": attack_name,
                "attack_param": _to_json_safe(attack_param),
                "label": label,
                "nc": 0.0,
                "ncc": 0.0,
                "ber": 1.0,
                "coverage": None,
                "error": None,
            }

            try:
                attack_seed = int(seed + 1000 * attack_idx)
                attacked_img = apply_attack(
                    watermarked_img,
                    attack_name,
                    attack_param,
                    seed=attack_seed,
                )

                if attacked_img.shape[:2] != watermarked_img.shape[:2]:
                    attacked_img = cv2.resize(
                        attacked_img,
                        (watermarked_img.shape[1], watermarked_img.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )

                cv2.imwrite(attacked_img_path, attacked_img)

                extracted_attack, attack_coverage, _ = extract_main(
                    attacked_img_path,
                    embed_info_path,
                    extracted_attack_path,
                )

                _, _, attack_nc, attack_ncc, attack_ber = calculate_metrics(
                    original_img,
                    attacked_img,
                    extracted_attack,
                    original_wm_binary,
                )

                attack_nc = float(attack_nc) if np.isfinite(float(attack_nc)) else 0.0
                attack_ncc = float(attack_ncc) if np.isfinite(float(attack_ncc)) else 0.0
                attack_ber = float(attack_ber) if np.isfinite(float(attack_ber)) else 1.0

                attack_record["nc"] = attack_nc
                attack_record["ncc"] = attack_ncc
                attack_record["ber"] = attack_ber
                attack_record["coverage"] = _json_safe_number(attack_coverage)
                attack_ncs.append(attack_nc)

            except Exception as e:
                attack_record["error"] = str(e)
                attack_ncs.append(0.0)
                if verbose:
                    print(f"[WARN] Attack failed: {label}: {e}")

            result["attack_results"].append(attack_record)

        mean_attack_nc = float(np.mean(attack_ncs)) if attack_ncs else 0.0
        min_attack_nc = float(np.min(attack_ncs)) if attack_ncs else 0.0
        clean_psnr = float(result["clean_psnr"]) if result["clean_psnr"] is not None else 0.0

        objective, psnr_feasible, psnr_gap = constrained_psnr_nc_objective(
            clean_psnr=clean_psnr,
            mean_attack_nc=mean_attack_nc,
            psnr_threshold=PSNR_MIN_THRESHOLD,
        )

        result["mean_attack_nc"] = mean_attack_nc
        result["min_attack_nc"] = min_attack_nc
        result["psnr_threshold"] = float(PSNR_MIN_THRESHOLD)
        result["psnr_feasible"] = bool(psnr_feasible)
        result["psnr_gap"] = float(psnr_gap)
        result["objective"] = float(objective)

    except Exception as e:
        result["error"] = str(e)
        result["objective"] = -1e9
        result["psnr_threshold"] = float(PSNR_MIN_THRESHOLD)
        result["psnr_feasible"] = False
        result["psnr_gap"] = None
        if verbose:
            print(f"[WARN] Candidate failed on {base_name}: {e}")
            traceback.print_exc()

    if not save_images:
        try:
            for fn in os.listdir(trial_dir):
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".jp2", ".npz")):
                    os.remove(os.path.join(trial_dir, fn))
        except Exception:
            pass

    return result


# =========================================================
# Firefly optimizer per image
# =========================================================
def firefly_optimize_for_image(
    host_path,
    watermark_path="/content/wm.png",
    output_root="/content/arnold_firefly_output",
    n_fireflies=4,
    n_generations=2,
    alpha=0.18,
    beta0=1.0,
    gamma=1.0,
    alpha_decay=0.80,
    seed=123,
    attack_suite=FAST_ATTACK_SUITE,
    final_attack_suite=ATTACK_SUITE,
    cleanup_candidate_files=True,
    verbose=True,
):
    """
    Faster Firefly algorithm for one host image.

    Speed changes:
        - n_fireflies default reduced from 6/8 to 4
        - n_generations default reduced from 4/5 to 2
        - optimization uses FAST_ATTACK_SUITE
        - final saved result uses final_attack_suite
    """
    os.makedirs(output_root, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(host_path))[0]
    image_dir = os.path.join(output_root, f"image_{_safe_filename(base_name)}")
    trial_root = os.path.join(image_dir, "candidate_trials")
    best_dir = os.path.join(image_dir, "best_final")
    os.makedirs(trial_root, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    dim = len(PARAM_SPECS)

    fireflies = rng.uniform(0.0, 1.0, size=(int(n_fireflies), dim))
    scores = np.full((int(n_fireflies),), -np.inf, dtype=np.float64)
    results = [None for _ in range(int(n_fireflies))]
    cache = {}
    history = []

    original_params_backup = get_current_params()

    best_score = -np.inf
    best_position = None
    best_result = None

    candidate_counter = 0
    current_alpha = float(alpha)

    def evaluate_position(pos, generation, firefly_id):
        nonlocal candidate_counter, best_score, best_position, best_result
        key = firefly_cache_key(pos)
        if key in cache:
            return cache[key]

        params = decode_firefly_position(pos)
        trial_dir = os.path.join(trial_root, f"g{generation:03d}_f{firefly_id:03d}_c{candidate_counter:05d}")
        candidate_counter += 1

        result = evaluate_candidate_for_image(
            params,
            host_path,
            watermark_path,
            trial_dir,
            attack_suite=attack_suite,
            seed=seed + candidate_counter,
            save_images=not cleanup_candidate_files,
            verbose=False,
        )

        result["generation"] = int(generation)
        result["firefly_id"] = int(firefly_id)
        result["candidate_index"] = int(candidate_counter - 1)
        result["position"] = [float(x) for x in np.clip(pos, 0.0, 1.0).tolist()]

        cache[key] = result

        history.append({
            "generation": int(generation),
            "firefly_id": int(firefly_id),
            "candidate_index": int(candidate_counter - 1),
            "objective": result["objective"],
            "clean_psnr": result["clean_psnr"],
            "psnr_threshold": result.get("psnr_threshold"),
            "psnr_feasible": result.get("psnr_feasible"),
            "psnr_gap": result.get("psnr_gap"),
            "mean_attack_nc": result["mean_attack_nc"],
            "min_attack_nc": result["min_attack_nc"],
            "clean_nc": result["clean_nc"],
            "clean_ber": result["clean_ber"],
            "q4_used": result.get("q4_used"),
            "hpos_used": result.get("hpos_used"),
            "skip_used": result.get("skip_used"),
            "repeat_factor": result.get("repeat_factor"),
            "params": params_to_jsonable(params),
            "error": result.get("error"),
        })

        if float(result["objective"]) > best_score:
            best_score = float(result["objective"])
            best_position = np.clip(pos.copy(), 0.0, 1.0)
            best_result = deepcopy(result)
            if verbose:
                print(
                    f"[NEW BEST][{base_name}] gen={generation}, firefly={firefly_id}, "
                    f"score={best_score:.6f}, "
                    f"PSNR={result['clean_psnr']:.4f}, "
                    f"PSNR_OK={result.get('psnr_feasible')}, "
                    f"gap={result.get('psnr_gap')}, "
                    f"meanNC={result['mean_attack_nc']:.4f}, "
                    f"minNC={result['min_attack_nc']:.4f}"
                )

        return result

    if verbose:
        print("=========================================================")
        print(f"FAST FIREFLY OPTIMIZATION FOR IMAGE: {os.path.basename(host_path)}")
        print(f"Constraint: clean_PSNR > {PSNR_MIN_THRESHOLD}")
        print("Optimization objective: maximize mean_NC_after_FAST_ATTACK_SUITE")
        print(f"Fireflies={n_fireflies}, generations={n_generations}, opt_attacks={len(attack_suite)}")
        print(f"Final evaluation attacks={len(final_attack_suite)}")
        print("Output:", image_dir)
        print("=========================================================")

    try:
        for i in range(int(n_fireflies)):
            res = evaluate_position(fireflies[i], generation=0, firefly_id=i)
            scores[i] = float(res["objective"])
            results[i] = res

        for gen in range(1, int(n_generations) + 1):
            if verbose:
                print(f"\n[GENERATION {gen}/{n_generations}] current_best={best_score:.6f}, alpha={current_alpha:.4f}")

            for i in range(int(n_fireflies)):
                for j in range(int(n_fireflies)):
                    if scores[j] > scores[i]:
                        rij = np.linalg.norm(fireflies[i] - fireflies[j])
                        beta = float(beta0) * math.exp(-float(gamma) * (rij ** 2))
                        random_step = current_alpha * (rng.random(dim) - 0.5)
                        new_pos = fireflies[i] + beta * (fireflies[j] - fireflies[i]) + random_step
                        new_pos = np.clip(new_pos, 0.0, 1.0)

                        new_res = evaluate_position(new_pos, generation=gen, firefly_id=i)
                        new_score = float(new_res["objective"])

                        if new_score > scores[i]:
                            fireflies[i] = new_pos
                            scores[i] = new_score
                            results[i] = new_res

            current_alpha *= float(alpha_decay)

        if best_position is None:
            best_idx = int(np.argmax(scores))
            best_position = fireflies[best_idx].copy()

        best_params = decode_firefly_position(best_position)
        apply_params(best_params)

        final_result = evaluate_candidate_for_image(
            best_params,
            host_path,
            watermark_path,
            best_dir,
            attack_suite=final_attack_suite,
            seed=seed + 999999,
            save_images=True,
            verbose=verbose,
        )
        final_result["best_position"] = [float(x) for x in best_position.tolist()]
        final_result["best_params"] = params_to_jsonable(best_params)
        final_result["optimization_attack_suite"] = _to_json_safe(attack_suite)
        final_result["final_attack_suite"] = _to_json_safe(final_attack_suite)

        best_params_path = os.path.join(image_dir, "best_params.json")
        best_result_path = os.path.join(image_dir, "best_result.json")
        history_json_path = os.path.join(image_dir, "firefly_history.json")
        history_csv_path = os.path.join(image_dir, "firefly_history.csv")
        attack_csv_path = os.path.join(image_dir, "attack_results_best.csv")

        with open(best_params_path, "w", encoding="utf-8") as f:
            json.dump(_to_json_safe(best_params), f, indent=2, ensure_ascii=False)

        with open(best_result_path, "w", encoding="utf-8") as f:
            json.dump(_to_json_safe(final_result), f, indent=2, ensure_ascii=False)

        with open(history_json_path, "w", encoding="utf-8") as f:
            json.dump(_to_json_safe(history), f, indent=2, ensure_ascii=False)

        if history:
            flat_rows = []
            for h in history:
                row = {
                    "generation": h["generation"],
                    "firefly_id": h["firefly_id"],
                    "candidate_index": h["candidate_index"],
                    "objective": h["objective"],
                    "clean_psnr": h["clean_psnr"],
                    "psnr_threshold": h.get("psnr_threshold"),
                    "psnr_feasible": h.get("psnr_feasible"),
                    "psnr_gap": h.get("psnr_gap"),
                    "mean_attack_nc": h["mean_attack_nc"],
                    "min_attack_nc": h["min_attack_nc"],
                    "clean_nc": h["clean_nc"],
                    "clean_ber": h["clean_ber"],
                    "q4_used": h.get("q4_used"),
                    "hpos_used": h.get("hpos_used"),
                    "skip_used": h.get("skip_used"),
                    "repeat_factor": h.get("repeat_factor"),
                    "error": h.get("error"),
                }
                for pk, pv in h["params"].items():
                    row[f"param_{pk}"] = str(pv)
                flat_rows.append(row)

            with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
                writer.writeheader()
                writer.writerows(flat_rows)

        attack_rows = []
        for ar in final_result.get("attack_results", []):
            attack_rows.append({
                "host_image": os.path.basename(host_path),
                "attack_name": ar.get("attack_name"),
                "attack_param": str(ar.get("attack_param")),
                "label": ar.get("label"),
                "nc": ar.get("nc"),
                "ncc": ar.get("ncc"),
                "ber": ar.get("ber"),
                "coverage": ar.get("coverage"),
                "error": ar.get("error"),
            })

        if attack_rows:
            with open(attack_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(attack_rows[0].keys()))
                writer.writeheader()
                writer.writerows(attack_rows)

        if cleanup_candidate_files:
            try:
                shutil.rmtree(trial_root)
            except Exception:
                pass

        if verbose:
            print("\n[FIREFLY DONE]")
            print("Best objective:", final_result["objective"])
            print("Best clean PSNR:", final_result["clean_psnr"])
            print("PSNR threshold:", final_result.get("psnr_threshold"))
            print("PSNR feasible:", final_result.get("psnr_feasible"))
            print("PSNR gap:", final_result.get("psnr_gap"))
            print("Best mean attack NC:", final_result["mean_attack_nc"])
            print("Best min attack NC:", final_result["min_attack_nc"])
            print("Best params saved:", best_params_path)
            print("Best result saved:", best_result_path)
            print("History saved:", history_csv_path)
            print("Final outputs saved:", best_dir)

        return final_result

    finally:
        apply_params(original_params_backup)


# =========================================================
# Find host images
# =========================================================
def find_host_images(input_folder="/content", watermark_path="/content/wm.png", exts=None):
    if exts is None:
        exts = [".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]

    image_paths = []
    watermark_abs = os.path.abspath(watermark_path)

    for fn in os.listdir(input_folder):
        ext = os.path.splitext(fn)[1].lower()
        if ext not in exts:
            continue
        full_path = os.path.abspath(os.path.join(input_folder, fn))
        if full_path == watermark_abs:
            continue
        try:
            img = load_color_image_24bit_safe(full_path, size=None)
            if img.shape[0] == HOST_SIZE and img.shape[1] == HOST_SIZE:
                image_paths.append(full_path)
        except Exception:
            pass

    return sorted(image_paths)


# =========================================================
# Optimize every image and save optimal parameter choice per image
# =========================================================
def optimize_all_images_with_firefly(
    input_folder="/content",
    watermark_path="/content/wm.png",
    output_root="/content/arnold_firefly_output",
    max_images=None,
    n_fireflies=4,
    n_generations=2,
    alpha=0.18,
    beta0=1.0,
    gamma=1.0,
    alpha_decay=0.80,
    seed=123,
    cleanup_candidate_files=True,
    verbose=True,
    attack_suite=FAST_ATTACK_SUITE,
    final_attack_suite=ATTACK_SUITE,
):
    os.makedirs(output_root, exist_ok=True)

    host_paths = find_host_images(input_folder=input_folder, watermark_path=watermark_path)
    if max_images is not None:
        host_paths = host_paths[:int(max_images)]

    if len(host_paths) == 0:
        raise ValueError(f"No valid {HOST_SIZE}x{HOST_SIZE} host images found in {input_folder}.")

    all_image_results = []

    print("=========================================================")
    print("SHA-256 + ARNOLD + FAST FIREFLY WATERMARK OPTIMIZATION")
    print("Weighted voting removed: True")
    print("Candidate scoring: Bit Survival Score + normalized MSE")
    print("H candidates: h21 and h22 only")
    print("Logistic map removed: True")
    print("RSA removed: True")
    print(f"Constraint: PSNR_before_attack > {PSNR_MIN_THRESHOLD}")
    print("Optimization objective: maximize mean_NC_after_FAST_ATTACK_SUITE")
    print("Images:", len(host_paths))
    print("Output root:", output_root)
    print("=========================================================")

    for idx, host_path in enumerate(host_paths):
        image_seed = int(seed + idx * 100000)
        result = firefly_optimize_for_image(
            host_path=host_path,
            watermark_path=watermark_path,
            output_root=output_root,
            n_fireflies=n_fireflies,
            n_generations=n_generations,
            alpha=alpha,
            beta0=beta0,
            gamma=gamma,
            alpha_decay=alpha_decay,
            seed=image_seed,
            attack_suite=attack_suite,
            final_attack_suite=final_attack_suite,
            cleanup_candidate_files=cleanup_candidate_files,
            verbose=verbose,
        )
        all_image_results.append(result)

    combined_json_path = os.path.join(output_root, "all_images_best_results.json")
    combined_csv_path = os.path.join(output_root, "all_images_optimal_params.csv")
    combined_attack_csv_path = os.path.join(output_root, "all_images_attack_results_best.csv")

    with open(combined_json_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(all_image_results), f, indent=2, ensure_ascii=False)

    csv_rows = []
    attack_rows = []

    for res in all_image_results:
        host_image = res.get("host_image")
        row = {
            "host_image": host_image,
            "objective": res.get("objective"),
            "clean_psnr": res.get("clean_psnr"),
            "psnr_threshold": res.get("psnr_threshold"),
            "psnr_feasible": res.get("psnr_feasible"),
            "psnr_gap": res.get("psnr_gap"),
            "clean_ssim": res.get("clean_ssim"),
            "clean_nc": res.get("clean_nc"),
            "clean_ncc": res.get("clean_ncc"),
            "clean_ber": res.get("clean_ber"),
            "clean_coverage": res.get("clean_coverage"),
            "mean_attack_nc": res.get("mean_attack_nc"),
            "min_attack_nc": res.get("min_attack_nc"),
            "q4_used": res.get("q4_used"),
            "hpos_used": res.get("hpos_used"),
            "skip_used": res.get("skip_used"),
            "total_flags": res.get("total_flags"),
            "repeat_factor": res.get("repeat_factor"),
            "error": res.get("error"),
        }

        best_params = res.get("best_params", res.get("params", {}))
        for k, v in best_params.items():
            row[f"param_{k}"] = str(v)

        csv_rows.append(row)

        for ar in res.get("attack_results", []):
            attack_rows.append({
                "host_image": host_image,
                "attack_name": ar.get("attack_name"),
                "attack_param": str(ar.get("attack_param")),
                "label": ar.get("label"),
                "nc": ar.get("nc"),
                "ncc": ar.get("ncc"),
                "ber": ar.get("ber"),
                "coverage": ar.get("coverage"),
                "error": ar.get("error"),
            })

    if csv_rows:
        with open(combined_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    if attack_rows:
        with open(combined_attack_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(attack_rows[0].keys()))
            writer.writeheader()
            writer.writerows(attack_rows)

    print("\n=========================================================")
    print("ALL IMAGE OPTIMIZATION FINISHED")
    print("Saved combined best results:", combined_json_path)
    print("Saved per-image optimal parameters:", combined_csv_path)
    print("Saved combined attack results:", combined_attack_csv_path)
    print("=========================================================")

    return all_image_results


# =========================================================
# Non-optimized processing with current parameters
# =========================================================
def process_images(
    input_folder="/content",
    watermark_path="/content/wm.png",
    output_root="/content/output_sha256_simple_threshold",
    arnold_iterations=None,
):
    if arnold_iterations is None:
        arnold_iterations = ARNOLD_ITERATIONS

    original_wm_binary = load_binary_watermark_8bit_safe(watermark_path, WM_SIZE)
    original_wm_binary = _force_binary_watermark_exact(original_wm_binary, WM_SIZE, thresh=WM_BIN_THRESH)

    print("Watermark path:", watermark_path)
    print("Watermark shape:", original_wm_binary.shape)
    print("Expected host size:", (HOST_SIZE, HOST_SIZE, 3))
    print("BLOCKSIZE:", BLOCKSIZE)
    print("Host DWT wavelet:", DWT_WAVELET, "bands:", DWT_BANDS)
    print("Host embedding color space:", HOST_EMBED_COLOR_SPACE)
    print("Host channels used:", HOST_CHANNELS, "(Y channel only)")
    print("Block selection hash: SHA-256")
    print("H candidates: h21 and h22 only")
    print("Extraction: simple threshold + majority voting for repeated bits")
    print("Candidate selection: H by lowest MSE, then Q-vs-H by Bit Survival Score + normalized MSE")
    print("Watermark preprocessing: binary -> watermark DWT -> LL payload -> Arnold scrambling -> payload bits")
    print("Arnold iterations:", int(arnold_iterations))
    print("RSA removed: True")
    print("Logistic map removed: True")

    out_img_dir = os.path.join(output_root, "watermarked_img")
    out_wm_dir = os.path.join(output_root, "extracted_wm")
    out_info_dir = os.path.join(output_root, "embed_info")
    out_summary_dir = os.path.join(output_root, "summary")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_wm_dir, exist_ok=True)
    os.makedirs(out_info_dir, exist_ok=True)
    os.makedirs(out_summary_dir, exist_ok=True)

    image_paths = find_host_images(input_folder=input_folder, watermark_path=watermark_path)

    print("Found image files:", [os.path.basename(p) for p in image_paths])
    print("Output root:", output_root)

    all_hosts_summary = []

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        original_img = load_color_image_24bit_safe(image_path, size=None)

        total_bits = _max_payload_bits_for_host(original_img.shape[0], original_img.shape[1], DWT_BANDS)

        base_name = os.path.splitext(filename)[0]
        watermarked_path = os.path.join(out_img_dir, f"watermarked_{base_name}.png")
        embed_info_path = os.path.join(out_info_dir, f"embed_info_{base_name}.npz")
        extracted_wm_path = os.path.join(out_wm_dir, f"extracted_{base_name}.png")

        print(f"\n--- Processing: {filename} ---")
        print(f"Available host blocks/payload capacity: {total_bits}")

        watermarked_img, extracted_wm = None, None

        print("Embedding.")
        t0 = time.time()
        try:
            watermarked_img = embed_full(
                image_path,
                watermark_path,
                watermarked_path,
                embed_info_path,
                arnold_iterations=arnold_iterations,
            )
        except Exception as e:
            print(f"Embedding failed for {filename}: {e}")
            traceback.print_exc()
        embed_time = time.time() - t0

        q4_used, hpos_used, skip_used = None, None, None
        dbg_meta = None
        try:
            if os.path.exists(embed_info_path):
                dbg_flags, dbg_hpos, dbg_meta = _load_embed_info(embed_info_path)
                q4_used, hpos_used, skip_used = _flag_stats(dbg_flags)
                print(
                    f"Embed info summary for {filename}: "
                    f"Q={q4_used}, H={hpos_used}, SKIP={skip_used}, TOTAL={len(dbg_flags)}, "
                    f"payload_bits={dbg_meta['payload_len']}, "
                    f"wm_bits={dbg_meta['wm_bit_len']}, "
                    f"repeat_factor={dbg_meta.get('structured_repeat_factor')}, "
                    f"arnold_iterations={dbg_meta.get('arnold_iterations')}"
                )
        except Exception as e:
            print(f"Could not read embed info for {filename}: {e}")

        print("Extracting watermark.")
        t0 = time.time()
        try:
            if watermarked_img is not None and os.path.exists(embed_info_path):
                extracted_wm, main_coverage, extract_meta = extract_main(
                    watermarked_path,
                    embed_info_path,
                    extracted_wm_path,
                )
            else:
                main_coverage = float("nan")
                extract_meta = None
        except Exception as e:
            print(f"Main extraction failed for {filename}: {e}")
            traceback.print_exc()
            main_coverage = float("nan")
            extract_meta = None
        main_extract_time = time.time() - t0

        print("Calculating metrics.")
        psnr_val, ssim_val, nc_val, ncc_val, ber_val = calculate_metrics(
            original_img,
            watermarked_img,
            extracted_wm,
            original_wm_binary,
        )

        print(
            f"Metrics for {filename}: "
            f"PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}, NC: {nc_val:.4f}, "
            f"NCC: {ncc_val:.4f}, BER: {ber_val:.6f}, "
            f"COVERAGE: {main_coverage:.4f}, Embed: {embed_time:.4f}s, Extract: {main_extract_time:.4f}s"
        )

        summary_entry = {
            "host_image": filename,
            "base_name": base_name,
            "arnold_enabled": True,
            "arnold_iterations": int(extract_meta.get("arnold_iterations", arnold_iterations)) if extract_meta is not None else int(arnold_iterations),
            "payload_len": int(extract_meta["payload_len"]) if extract_meta is not None else None,
            "wm_bit_len": int(extract_meta["wm_bit_len"]) if extract_meta is not None else None,
            "structured_repeat_factor": int(extract_meta.get("structured_repeat_factor", -1)) if extract_meta is not None else None,
            "q4_used": int(q4_used) if q4_used is not None else None,
            "hpos_used": int(hpos_used) if hpos_used is not None else None,
            "skip_used": int(skip_used) if skip_used is not None else None,
            "final_main_psnr": _json_safe_number(psnr_val),
            "final_main_ssim": _json_safe_number(ssim_val),
            "final_main_nc": _json_safe_number(nc_val),
            "final_main_ncc": _json_safe_number(ncc_val),
            "final_main_ber": _json_safe_number(ber_val),
            "final_main_coverage": _json_safe_number(main_coverage),
            "embed_time_sec": _json_safe_number(embed_time),
            "extract_time_sec": _json_safe_number(main_extract_time),
            "params": params_to_jsonable(get_current_params()),
        }
        all_hosts_summary.append(summary_entry)

    if len(all_hosts_summary) > 0:
        all_json_path = os.path.join(out_summary_dir, "all_hosts_summary.json")
        all_csv_path = os.path.join(out_summary_dir, "all_hosts_summary.csv")

        with open(all_json_path, "w", encoding="utf-8") as f:
            json.dump(_to_json_safe(all_hosts_summary), f, indent=2, ensure_ascii=False)

        csv_rows = []
        for item in all_hosts_summary:
            row = {
                "host_image": item.get("host_image"),
                "base_name": item.get("base_name"),
                "arnold_enabled": item.get("arnold_enabled"),
                "arnold_iterations": item.get("arnold_iterations"),
                "payload_len": item.get("payload_len"),
                "wm_bit_len": item.get("wm_bit_len"),
                "structured_repeat_factor": item.get("structured_repeat_factor"),
                "q4_used": item.get("q4_used"),
                "hpos_used": item.get("hpos_used"),
                "skip_used": item.get("skip_used"),
                "final_main_psnr": item.get("final_main_psnr"),
                "final_main_ssim": item.get("final_main_ssim"),
                "final_main_nc": item.get("final_main_nc"),
                "final_main_ncc": item.get("final_main_ncc"),
                "final_main_ber": item.get("final_main_ber"),
                "final_main_coverage": item.get("final_main_coverage"),
                "embed_time_sec": item.get("embed_time_sec"),
                "extract_time_sec": item.get("extract_time_sec"),
            }
            for pk, pv in item.get("params", {}).items():
                row[f"param_{pk}"] = str(pv)
            csv_rows.append(row)

        with open(all_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

        print("\nSaved combined summary files:")
        print(" -", all_json_path)
        print(" -", all_csv_path)

    return all_hosts_summary


# =========================================================
# Main run examples
# =========================================================
if __name__ == "__main__":
    # Fast default: Firefly optimization with fewer candidates and fewer generations.
    optimize_all_images_with_firefly(
        input_folder="/content",
        watermark_path="/content/wm.png",
        output_root="/content/arnold_firefly_output",
        max_images=None,
        n_fireflies=4,
        n_generations=2,
        alpha=0.18,
        beta0=1.0,
        gamma=1.0,
        alpha_decay=0.80,
        seed=123,
        cleanup_candidate_files=True,
        verbose=True,
        attack_suite=FAST_ATTACK_SUITE,
        final_attack_suite=ATTACK_SUITE,
    )

    # If you do not want optimization, comment the block above and uncomment:
    # process_images(
    #     input_folder="/content",
    #     watermark_path="/content/wm.png",
    #     output_root="/content/output_sha256_simple_threshold",
    #     arnold_iterations=ARNOLD_ITERATIONS,
    # )
