"""
Arnold-scrambled DWT-Hessenberg watermarking method for WAVES.

This module contains ONLY the fixed watermarking method. It deliberately removes
all Firefly/optimization code so that WAVES evaluates one frozen method.

Important: extraction uses the per-image side-information file produced during
embedding (`embed_info_path`). Therefore this implementation is semi-blind / side-
information-assisted.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
try:
    import pywt
except Exception:  # fallback keeps smoke tests runnable without PyWavelets
    pywt = None
from scipy import linalg
from skimage.metrics import structural_similarity as ssim

# -----------------------------------------------------------------------------
# Fixed global parameters. These are not optimized inside WAVES.
# -----------------------------------------------------------------------------
BLOCKSIZE = 4
WM_SIZE = int(os.environ.get("ARNOLD_HESS_WM_SIZE", "64"))
PRIVATE_KEY = os.environ.get("ARNOLD_HESS_PRIVATE_KEY", "KB123")
HOST_SIZE = int(os.environ.get("ARNOLD_HESS_HOST_SIZE", "512"))
ARNOLD_ITERATIONS = int(os.environ.get("ARNOLD_HESS_ITERATIONS", "17"))

DWT_WAVELET = "haar"
DWT_LEVEL = 1
DWT_BANDS = tuple(os.environ.get("ARNOLD_HESS_DWT_BANDS", "LL,HL,HH,LH").split(","))
HOST_CHANNELS = (0, 1, 2)  # B, G, R for OpenCV images
WM_BIN_THRESH = 127

FLAG_Q4 = 0
FLAG_HPOS = 1
FLAG_SKIP = 2
HPOS_NONE = -1

Q4_TAU = 0.50
Q4_MARGIN = 0.08
H01_Q = 7.0
H01_MARGIN = 0.90
HPOS_CANDIDATES = (("h21", (1, 0)), ("h12", (0, 1)), ("h22", (1, 1)))

EPS_CONF = 1e-12
HARD_MAJORITY_ON_AMBIGUOUS = True
HARD_MAJORITY_MIN_COUNT = 3
MIN_LLR_WEIGHT_Q = 0.25
MIN_LLR_WEIGHT_H = 0.12
Q_LLR_SCALE = 2.50
H_LLR_SCALE = 0.75
Q_VOTE_WEIGHT = 0.90
H_VOTE_WEIGHT = 0.75
TOPK_VOTES_PER_BIT = None
EXTRACT_AMBIGUITY_THRESH = 0.12

MIN_WORST_MARGIN_TO_EMBED = 0.04
MIN_AVG_MARGIN_TO_EMBED = 0.12
MAX_Q_CAND_MSE = 0.18
MAX_H_CAND_MSE = 0.24
PREFER_HPOS_WHEN_CLOSE = False
HPOS_CLOSE_SEL_RATIO = 0.88
HPOS_CLOSE_SCORE_GAP = 0.08

Q4_GIVENS_THETA_MAX = 0.42
Q4_GIVENS_COARSE_STEPS = 13
Q4_GIVENS_FINE_THETA_MAX = 0.08
Q4_GIVENS_FINE_STEPS = 0
Q4_GIVENS_PASSES = 1
Q4_GIVENS_MSE_WEIGHT = 0.10
Q4_GIVENS_EXTRA_MARGIN_WEIGHT = 0.04
Q4_GIVENS_PAIRS = ((0, 2), (1, 3), (0, 1), (1, 2))

Q_FAST_ACCEPT_ENABLE = True
Q_FAST_ACCEPT_SELECT_VALUE = 0.75
Q_FAST_ACCEPT_WORST_MARGIN = 0.20
Q_FAST_ACCEPT_MAX_MSE = 0.08
FAST_CANDIDATE_SCORING = True

DEBUG = bool(int(os.environ.get("ARNOLD_HESS_DEBUG", "0")))
MAX_REPEAT_FACTOR = int(os.environ.get("ARNOLD_HESS_MAX_REPEAT_FACTOR", "1"))
FAST_EMBED_MODE = bool(int(os.environ.get("ARNOLD_HESS_FAST_EMBED", "1")))


def _debug(msg: str) -> None:
    if DEBUG:
        print(msg)


def _to_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _safe_filename(s: str) -> str:
    for ch in [" ", "&", "(", ")", ",", ".", "'", '"', "/", "\\", ":", ";", "[", "]"]:
        s = str(s).replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def _flag_stats(flags: Sequence[int]) -> Tuple[int, int, int]:
    q4_used = sum(1 for f in flags if int(f) == FLAG_Q4)
    hpos_used = sum(1 for f in flags if int(f) == FLAG_HPOS)
    skip_used = sum(1 for f in flags if int(f) == FLAG_SKIP)
    return q4_used, hpos_used, skip_used


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def load_color_image_24bit_safe(path: str, size: int | None = None, bg: int = 255) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image file: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        bgr = img[..., :3].astype(np.float32)
        a = img[..., 3:4].astype(np.float32) / 255.0
        comp = bgr * a + np.full_like(bgr, float(bg)) * (1.0 - a)
        img = np.clip(np.rint(comp), 0, 255).astype(np.uint8)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = img.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported image format for: {path}")

    if size is not None and (img.shape[0] != size or img.shape[1] != size):
        raise ValueError(f"Image must be exactly {size}x{size}. Got {img.shape[1]}x{img.shape[0]}.")
    return img


def load_binary_watermark_8bit_safe(path: str, size: int | None = None, thresh: int = WM_BIN_THRESH) -> np.ndarray:
    img = load_color_image_24bit_safe(path, size=size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    wm_bin = np.where(gray >= int(thresh), 255, 0).astype(np.uint8)
    if size is not None and wm_bin.shape != (size, size):
        raise ValueError(f"Binary watermark must be exactly {size}x{size}.")
    return wm_bin


def force_binary_watermark_exact(img_u8: np.ndarray, size: int = WM_SIZE, thresh: int = WM_BIN_THRESH) -> np.ndarray:
    if img_u8 is None:
        raise ValueError("Watermark image is None.")
    out = np.asarray(img_u8).astype(np.uint8)
    if out.ndim == 3:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    if out.shape != (size, size):
        out = cv2.resize(out, (size, size), interpolation=cv2.INTER_NEAREST)
    return np.where(out >= int(thresh), 255, 0).astype(np.uint8)


# Backward-compatible private alias used by older scripts.
_force_binary_watermark_exact = force_binary_watermark_exact


def watermark_bits_from_file(path: str) -> np.ndarray:
    wm = load_binary_watermark_8bit_safe(path, size=WM_SIZE)
    return (wm >= WM_BIN_THRESH).astype(bool).reshape(-1)


# -----------------------------------------------------------------------------
# Arnold scrambling
# -----------------------------------------------------------------------------
def _normalize_arnold_iterations(iterations: int) -> int:
    iterations = int(round(iterations))
    if iterations < 0:
        raise ValueError("ARNOLD_ITERATIONS must be non-negative.")
    return iterations


def arnold_scramble_binary(wm_bits_2d: np.ndarray, iterations: int) -> np.ndarray:
    bits = np.asarray(wm_bits_2d, dtype=np.uint8)
    if bits.ndim != 2 or bits.shape[0] != bits.shape[1]:
        raise ValueError("Arnold scrambling requires a square 2D image.")
    n = bits.shape[0]
    out = bits.copy()
    for _ in range(_normalize_arnold_iterations(iterations)):
        temp = np.zeros_like(out)
        for x in range(n):
            for y in range(n):
                temp[(x + y) % n, (x + 2 * y) % n] = out[x, y]
        out = temp
    return out.astype(np.uint8)


def arnold_unscramble_binary(scrambled_bits_2d: np.ndarray, iterations: int) -> np.ndarray:
    bits = np.asarray(scrambled_bits_2d, dtype=np.uint8)
    if bits.ndim != 2 or bits.shape[0] != bits.shape[1]:
        raise ValueError("Inverse Arnold requires a square 2D image.")
    n = bits.shape[0]
    out = bits.copy()
    for _ in range(_normalize_arnold_iterations(iterations)):
        temp = np.zeros_like(out)
        for x_new in range(n):
            for y_new in range(n):
                temp[(2 * x_new - y_new) % n, (-x_new + y_new) % n] = out[x_new, y_new]
        out = temp
    return out.astype(np.uint8)


# -----------------------------------------------------------------------------
# DWT and capacity helpers
# -----------------------------------------------------------------------------
def _required_multiple_for_dwt(level: int, block_size: int) -> int:
    return (2 ** int(level)) * int(block_size)


def _crop_for_dwt(channel_u8: np.ndarray, level: int, block_size: int):
    req = _required_multiple_for_dwt(level, block_size)
    h, w = channel_u8.shape
    h0 = (h // req) * req
    w0 = (w // req) * req
    if h0 == 0 or w0 == 0:
        raise ValueError("Image too small for DWT + BLOCKSIZE constraint.")
    return channel_u8[:h0, :w0].astype(np.float64), h0, w0


def _haar_dwt2(channel_f64: np.ndarray):
    """One-level orthonormal Haar DWT fallback matching PyWavelets band naming."""
    a = channel_f64.astype(np.float64)
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

    h2, w2 = LL.shape
    low_rows = np.empty((h2, w2 * 2), dtype=np.float64)
    high_rows = np.empty((h2, w2 * 2), dtype=np.float64)
    low_rows[:, 0::2], low_rows[:, 1::2] = low_even, low_odd
    high_rows[:, 0::2], high_rows[:, 1::2] = high_even, high_odd

    even_rows = (low_rows + high_rows) / np.sqrt(2.0)
    odd_rows = (low_rows - high_rows) / np.sqrt(2.0)
    out = np.empty((h2 * 2, w2 * 2), dtype=np.float64)
    out[0::2, :], out[1::2, :] = even_rows, odd_rows
    return out


def _dwt_split_4bands(channel_f64: np.ndarray, wavelet: str) -> Dict[str, np.ndarray]:
    if pywt is not None:
        LL, (LH, HL, HH) = pywt.dwt2(channel_f64, wavelet)
    else:
        if wavelet != "haar":
            raise ImportError("PyWavelets is required for non-haar wavelets")
        LL, (LH, HL, HH) = _haar_dwt2(channel_f64)
    return {"LL": LL.copy(), "LH": LH.copy(), "HL": HL.copy(), "HH": HH.copy()}


def _dwt_merge_4bands(bands: Dict[str, np.ndarray], wavelet: str) -> np.ndarray:
    if pywt is not None:
        return pywt.idwt2((bands["LL"], (bands["LH"], bands["HL"], bands["HH"])), wavelet)
    if wavelet != "haar":
        raise ImportError("PyWavelets is required for non-haar wavelets")
    return _haar_idwt2(bands["LL"], bands["LH"], bands["HL"], bands["HH"])


def generate_color_subband_block_indices(bands_by_ch, channel_ids, band_names, block_size, total_needed, key):
    all_positions = []
    for ch in channel_ids:
        for band_name in band_names:
            band = bands_by_ch[ch][band_name]
            h, w = band.shape
            h0 = (h // block_size) * block_size
            w0 = (w // block_size) * block_size
            for i in range(0, h0, block_size):
                for j in range(0, w0, block_size):
                    all_positions.append((int(ch), str(band_name), int(i), int(j)))
    total_blocks = len(all_positions)
    if total_needed > total_blocks:
        raise ValueError(f"Capacity insufficient: need {total_needed}, available {total_blocks}.")
    seed_int = int.from_bytes(hashlib.sha512(str(key).encode("utf-8")).digest(), "big")
    rng = random.Random(seed_int)
    indices = list(range(total_blocks))
    rng.shuffle(indices)
    return [all_positions[idx] for idx in indices[:total_needed]]


def make_structured_repetition_schedule(blocks, payload_len: int):
    payload_len = int(payload_len)
    if payload_len <= 0:
        raise ValueError("payload_len must be positive.")
    total_blocks = len(blocks)
    repeat_factor = total_blocks // payload_len
    if MAX_REPEAT_FACTOR > 0:
        repeat_factor = min(repeat_factor, int(MAX_REPEAT_FACTOR))
    if repeat_factor <= 0:
        raise ValueError(f"Capacity insufficient: payload_len={payload_len}, total_blocks={total_blocks}.")
    usable_blocks = repeat_factor * payload_len
    selected = list(blocks[:usable_blocks])
    schedule = []
    for rep in range(repeat_factor):
        base = rep * payload_len
        for kpos in range(payload_len):
            schedule.append((kpos, selected[base + kpos]))
    return schedule, repeat_factor, usable_blocks


def max_payload_bits_for_host(host_h: int, host_w: int, dwt_bands=None) -> int:
    dwt_bands = DWT_BANDS if dwt_bands is None else tuple(dwt_bands)
    h2 = (host_h // 2) // BLOCKSIZE
    w2 = (host_w // 2) // BLOCKSIZE
    return len(HOST_CHANNELS) * len(dwt_bands) * h2 * w2


def _payload_len_from_meta(meta: dict) -> int:
    return int(meta["payload_len"])


# -----------------------------------------------------------------------------
# Side information
# -----------------------------------------------------------------------------
def _save_embed_info(embed_info_path: str, flags, hpos_list, payload_meta: dict) -> None:
    dirname = os.path.dirname(embed_info_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    meta_json = json.dumps(_to_json_safe(payload_meta), ensure_ascii=False)
    np.savez_compressed(
        embed_info_path,
        flags=np.array(flags, dtype=np.int16),
        hpos=np.array(hpos_list, dtype=np.int16),
        meta_json=np.array(meta_json),
    )


def _load_embed_info(embed_info_path: str):
    if not os.path.exists(embed_info_path):
        raise ValueError(f"Embed info file missing: {embed_info_path}")
    data = np.load(embed_info_path, allow_pickle=False)
    if "flags" not in data or "hpos" not in data:
        raise ValueError(f"Embed info file must contain flags and hpos: {embed_info_path}")
    flags = np.atleast_1d(data["flags"]).astype(np.int32)
    hpos = np.atleast_1d(data["hpos"]).astype(np.int32)
    if len(hpos) < len(flags):
        hpos = np.concatenate([hpos, np.full(len(flags) - len(hpos), HPOS_NONE, dtype=np.int32)])
    elif len(hpos) > len(flags):
        hpos = hpos[: len(flags)]
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


# -----------------------------------------------------------------------------
# Payload helpers
# -----------------------------------------------------------------------------
def prepare_binary_watermark_payload(watermark_path: str, wm_size: int = WM_SIZE, arnold_iterations: int = ARNOLD_ITERATIONS):
    wm_binary = load_binary_watermark_8bit_safe(watermark_path, size=wm_size)
    wm_binary = force_binary_watermark_exact(wm_binary, wm_size, WM_BIN_THRESH)
    wm_bits_2d = (wm_binary >= WM_BIN_THRESH).astype(np.uint8)
    scrambled_bits_2d = arnold_scramble_binary(wm_bits_2d, arnold_iterations)
    payload_bits = scrambled_bits_2d.reshape(-1).astype(np.uint8)
    meta = {
        "wm_shape": tuple(int(x) for x in wm_binary.shape),
        "wm_size": int(wm_size),
        "wm_bit_len": int(wm_bits_2d.size),
        "payload_len": int(payload_bits.size),
        "arnold_enabled": True,
        "arnold_iterations": int(arnold_iterations),
    }
    return wm_binary, payload_bits, meta


def reconstruct_binary_watermark_from_payload_bits(payload_bits, meta: dict) -> np.ndarray:
    wm_size = int(meta["wm_size"])
    wm_bit_len = int(meta["wm_bit_len"])
    payload_len = int(meta["payload_len"])
    arnold_iterations = int(meta.get("arnold_iterations", ARNOLD_ITERATIONS))
    bits = np.array(payload_bits[:payload_len], dtype=np.uint8).reshape(-1)
    if bits.size < wm_bit_len:
        bits = np.concatenate([bits, np.zeros(wm_bit_len - bits.size, dtype=np.uint8)])
    elif bits.size > wm_bit_len:
        bits = bits[:wm_bit_len]
    scrambled_2d = bits.reshape((wm_size, wm_size)).astype(np.uint8)
    recovered = arnold_unscramble_binary(scrambled_2d, arnold_iterations)
    return force_binary_watermark_exact((recovered * 255).astype(np.uint8), wm_size, WM_BIN_THRESH)


# -----------------------------------------------------------------------------
# Hessenberg/Q-candidate math
# -----------------------------------------------------------------------------
def _safe_sign(x: float) -> float:
    return 1.0 if x >= 0 else -1.0


def _nearest_nonnegative_mod_value(a: float, target: float, q: float) -> float:
    k0 = int(np.round((a - target) / q))
    cands = [target + k * q for k in range(k0 - 3, k0 + 4) if target + k * q >= 0]
    return min(cands, key=lambda v: abs(v - a)) if cands else max(0.0, target)


def _q4_stat(Qm: np.ndarray) -> float:
    return float(Qm[1, 1] ** 2 + Qm[2, 1] ** 2)


def _llr_q4_from_Q(Qm: np.ndarray) -> float:
    return _q4_stat(Qm) - float(Q4_TAU)


def _norm_llr_q4(llr_q4: float) -> float:
    return float(llr_q4) / max(float(Q4_MARGIN), EPS_CONF)


def _givens4(a: int, b: int, theta: float) -> np.ndarray:
    G = np.eye(4, dtype=np.float64)
    c, s = np.cos(theta), np.sin(theta)
    G[a, a], G[a, b], G[b, a], G[b, b] = c, s, -s, c
    return G


def _q4_signed_margin_from_Q(Qm: np.ndarray, bit: int) -> float:
    d = _q4_stat(Qm) - float(Q4_TAU)
    return float(d) if int(bit) == 1 else -float(d)


def _q4_givens_loss(block0: np.ndarray, Hh: np.ndarray, Qtrial: np.ndarray, bit: int):
    signed_margin = _q4_signed_margin_from_Q(Qtrial, bit)
    target = float(Q4_MARGIN)
    margin_loss = 0.0 if signed_margin >= target else target - signed_margin
    extra_margin = max(0.0, signed_margin - target)
    block_trial = Qtrial @ Hh @ Qtrial.T
    mse = float(np.mean((block0 - block_trial) ** 2))
    loss = margin_loss + Q4_GIVENS_MSE_WEIGHT * math.sqrt(max(mse, 0.0)) + Q4_GIVENS_EXTRA_MARGIN_WEIGHT * extra_margin
    return float(loss), float(mse), float(signed_margin)


def _build_q4_candidate(block: np.ndarray, bit: int) -> np.ndarray:
    Hh, Qm = linalg.hessenberg(block, calc_q=True)
    best_Q = Qm.copy()
    best_loss, best_mse, best_margin = _q4_givens_loss(block, Hh, best_Q, bit)
    if best_margin >= float(Q4_MARGIN):
        return best_Q @ Hh @ best_Q.T

    grids = [np.linspace(-Q4_GIVENS_THETA_MAX, Q4_GIVENS_THETA_MAX, int(Q4_GIVENS_COARSE_STEPS))]
    if int(Q4_GIVENS_FINE_STEPS) > 1:
        grids.append(np.linspace(-Q4_GIVENS_FINE_THETA_MAX, Q4_GIVENS_FINE_THETA_MAX, int(Q4_GIVENS_FINE_STEPS)))

    for pass_idx in range(max(1, int(Q4_GIVENS_PASSES))):
        theta_grid = grids[min(pass_idx, len(grids) - 1)]
        improved = False
        for a, b in Q4_GIVENS_PAIRS:
            local_best_Q = best_Q
            local_best_key = (best_loss, best_mse, -best_margin)
            for th in theta_grid:
                if abs(float(th)) < 1e-15:
                    continue
                Qtrial = _givens4(a, b, float(th)) @ best_Q
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
    return best_Q @ Hh @ best_Q.T


def _llr_hpos_from_H(Hh: np.ndarray, q_h01: float, pos) -> float:
    rr, cc = pos
    z = abs(float(Hh[rr, cc])) % float(q_h01)
    return z - 0.5 * float(q_h01)


def _norm_llr_hpos(llr_h01: float, q_h01: float) -> float:
    return float(llr_h01) / max(0.5 * float(q_h01), EPS_CONF)


def _build_hpos_candidate(block: np.ndarray, bit: int, q_h01: float, h01_margin: float, pos) -> np.ndarray:
    Hh, Qm = linalg.hessenberg(block, calc_q=True)
    H2 = Hh.copy()
    rr, cc = pos
    v = float(H2[rr, cc])
    sgn = _safe_sign(v)
    a = abs(v)
    z = a % q_h01
    safe = False
    if bit == 1:
        safe = (z >= (0.5 * q_h01 + h01_margin)) and (z <= (q_h01 - h01_margin))
    else:
        safe = (z >= h01_margin) and (z <= (0.5 * q_h01 - h01_margin))
    if not safe:
        target = 0.75 * q_h01 if bit == 1 else 0.25 * q_h01
        H2[rr, cc] = sgn * _nearest_nonnegative_mod_value(a, target, q_h01)
    return Qm @ H2 @ Qm.T


def _signed_margin_from_block(block: np.ndarray, mode: int, bit: int, q_h01: float, pos=None) -> float:
    Hh, Qm = linalg.hessenberg(block, calc_q=True)
    if mode == FLAG_Q4:
        llr = _norm_llr_q4(_llr_q4_from_Q(Qm))
    elif mode == FLAG_HPOS and pos is not None:
        llr = _norm_llr_hpos(_llr_hpos_from_H(Hh, q_h01, pos), q_h01)
    else:
        llr = -1.0
    return llr if bit == 1 else -llr


def _candidate_attack_score(block0: np.ndarray, block_cand: np.ndarray, mode: int, bit: int, q_h01: float, pos=None):
    mse = float(np.mean((block0 - block_cand) ** 2))
    base = block_cand.astype(np.float64)
    variants = [base, np.rint(base), base + 0.25, base - 0.25]
    if not FAST_CANDIDATE_SCORING:
        variants += [1.01 * base, 0.99 * base]
    margins = []
    for v in variants:
        try:
            margins.append(_signed_margin_from_block(v, mode, bit, q_h01, pos=pos))
        except Exception:
            margins.append(-1.0)
    raw_margin = float(margins[0])
    avg_margin = float(np.mean(margins))
    worst_margin = float(np.min(margins))
    robust_score = 0.5 * raw_margin + 0.3 * avg_margin + 0.2 * worst_margin
    final_score = robust_score / (1.0 + np.sqrt(mse + 1e-6))
    ok = raw_margin >= 0.0 and avg_margin >= 0.0 and worst_margin >= 0.0
    return float(final_score), float(mse), raw_margin, avg_margin, worst_margin, bool(ok)


def _cap01(x, t=1.0) -> float:
    return min(float(x), float(t))


def _select_value(c: dict) -> float:
    mse_penalty = 0.35 * math.sqrt(max(float(c["mse"]), 0.0))
    return 0.50 * _cap01(c["worst"]) + 0.30 * _cap01(c["avg"]) + 0.20 * _cap01(c["raw"]) - mse_penalty


def _candidate_rank_key(c: dict):
    return (_select_value(c), _cap01(c["worst"]), _cap01(c["avg"]), _cap01(c["raw"]), -float(c["mse"]))


def _candidate_is_strong_enough(cand: dict) -> bool:
    if not bool(cand["ok"]):
        return False
    if float(_select_value(cand)) <= 0.0:
        return False
    if float(cand["worst"]) < float(MIN_WORST_MARGIN_TO_EMBED):
        return False
    if float(cand["avg"]) < float(MIN_AVG_MARGIN_TO_EMBED):
        return False
    if int(cand["flag"]) == FLAG_Q4 and float(cand["mse"]) > MAX_Q_CAND_MSE:
        return False
    if int(cand["flag"]) == FLAG_HPOS and float(cand["mse"]) > MAX_H_CAND_MSE:
        return False
    return True


def _best_hpos_candidate(block0: np.ndarray, bit: int, q_h01: float, h01_margin: float):
    best = None
    for hpos_idx, (hname, pos) in enumerate(HPOS_CANDIDATES):
        block_h = _build_hpos_candidate(block0, bit, q_h01, h01_margin, pos)
        score_h, mse_h, raw_h, avg_h, worst_h, ok_h = _candidate_attack_score(block0, block_h, FLAG_HPOS, bit, q_h01, pos=pos)
        cand = {"flag": FLAG_HPOS, "block": block_h, "score": score_h, "mse": mse_h, "raw": raw_h, "avg": avg_h, "worst": worst_h, "ok": ok_h, "hpos_idx": hpos_idx, "hpos_name": hname, "pos": pos}
        if best is None or _candidate_rank_key(cand) > _candidate_rank_key(best):
            best = cand
    return best


# -----------------------------------------------------------------------------
# Public embedding/extraction API
# -----------------------------------------------------------------------------
def get_current_params() -> dict:
    return {
        "ARNOLD_ITERATIONS": int(ARNOLD_ITERATIONS),
        "DWT_BANDS": tuple(DWT_BANDS),
        "Q4_MARGIN": float(Q4_MARGIN),
        "H01_Q": float(H01_Q),
        "H01_MARGIN": float(H01_MARGIN),
        "MIN_LLR_WEIGHT_Q": float(MIN_LLR_WEIGHT_Q),
        "MIN_LLR_WEIGHT_H": float(MIN_LLR_WEIGHT_H),
        "Q_LLR_SCALE": float(Q_LLR_SCALE),
        "H_LLR_SCALE": float(H_LLR_SCALE),
        "Q_VOTE_WEIGHT": float(Q_VOTE_WEIGHT),
        "H_VOTE_WEIGHT": float(H_VOTE_WEIGHT),
        "MIN_WORST_MARGIN_TO_EMBED": float(MIN_WORST_MARGIN_TO_EMBED),
        "MIN_AVG_MARGIN_TO_EMBED": float(MIN_AVG_MARGIN_TO_EMBED),
        "MAX_Q_CAND_MSE": float(MAX_Q_CAND_MSE),
        "MAX_H_CAND_MSE": float(MAX_H_CAND_MSE),
        "Q4_GIVENS_THETA_MAX": float(Q4_GIVENS_THETA_MAX),
        "Q4_GIVENS_MSE_WEIGHT": float(Q4_GIVENS_MSE_WEIGHT),
        "Q4_GIVENS_EXTRA_MARGIN_WEIGHT": float(Q4_GIVENS_EXTRA_MARGIN_WEIGHT),
    }


def _embed_main_in_image(host_img_orig: np.ndarray, watermark_path: str, embed_info_path: str, arnold_iterations: int | None = None) -> np.ndarray:
    arnold_iterations = ARNOLD_ITERATIONS if arnold_iterations is None else int(arnold_iterations)
    if host_img_orig.ndim != 3 or host_img_orig.shape[2] != 3:
        raise ValueError("Host image must be a 24-bit color image with 3 channels.")

    bands_by_ch = {}
    H0 = W0 = None
    for ch in HOST_CHANNELS:
        work, h0, w0 = _crop_for_dwt(host_img_orig[:, :, ch], DWT_LEVEL, BLOCKSIZE)
        if H0 is None:
            H0, W0 = h0, w0
        bands_by_ch[ch] = _dwt_split_4bands(work, DWT_WAVELET)

    _, wm_bits, wm_meta = prepare_binary_watermark_payload(watermark_path, WM_SIZE, arnold_iterations)
    L = len(wm_bits)
    total_blocks = sum(
        (bands_by_ch[ch][b].shape[0] // BLOCKSIZE) * (bands_by_ch[ch][b].shape[1] // BLOCKSIZE)
        for ch in HOST_CHANNELS
        for b in DWT_BANDS
    )
    if total_blocks < L:
        raise ValueError(f"Capacity insufficient: need {L} watermark bits but only {total_blocks} blocks available.")

    blocks = generate_color_subband_block_indices(bands_by_ch, HOST_CHANNELS, DWT_BANDS, BLOCKSIZE, total_blocks, PRIVATE_KEY)
    schedule, repeat_factor, usable_blocks = make_structured_repetition_schedule(blocks, L)
    q_h01 = max(float(H01_Q), 1e-6)
    h01_margin = min(max(0.0, float(H01_MARGIN)), 0.49 * q_h01)

    flag_list: List[int] = []
    hpos_list: List[int] = []

    for kpos, block_pos in schedule:
        ch, band_name, i, j = block_pos
        bit = int(wm_bits[kpos])
        band = bands_by_ch[ch][band_name]
        block0 = band[i : i + BLOCKSIZE, j : j + BLOCKSIZE].copy()
        try:
            if FAST_EMBED_MODE:
                # Fast fixed method for benchmark-scale runs: use the first Hessenberg
                # off-diagonal carrier. This preserves the Arnold + DWT + Hessenberg
                # embedding logic but avoids the expensive candidate search / Firefly path.
                hname, pos = HPOS_CANDIDATES[0]
                block_h = _build_hpos_candidate(block0, bit, q_h01, h01_margin, pos)
                band[i : i + BLOCKSIZE, j : j + BLOCKSIZE] = block_h
                flag_list.append(FLAG_HPOS)
                hpos_list.append(0)
                continue

            block_q4 = _build_q4_candidate(block0, bit)
            score_q4, mse_q4, raw_q4, avg_q4, worst_q4, ok_q4 = _candidate_attack_score(block0, block_q4, FLAG_Q4, bit, q_h01)
            q_cand = {"flag": FLAG_Q4, "block": block_q4, "score": score_q4, "mse": mse_q4, "raw": raw_q4, "avg": avg_q4, "worst": worst_q4, "ok": ok_q4, "hpos_idx": HPOS_NONE, "hpos_name": "none"}
            q_sel = _select_value(q_cand)

            if Q_FAST_ACCEPT_ENABLE and _candidate_is_strong_enough(q_cand) and q_sel >= Q_FAST_ACCEPT_SELECT_VALUE and q_cand["worst"] >= Q_FAST_ACCEPT_WORST_MARGIN and q_cand["mse"] <= Q_FAST_ACCEPT_MAX_MSE:
                strong_cands = [q_cand]
            else:
                best_h = _best_hpos_candidate(block0, bit, q_h01, h01_margin)
                strong_cands = [c for c in [q_cand, best_h] if c is not None and _candidate_is_strong_enough(c)]

            if strong_cands:
                strong_cands = sorted(strong_cands, key=_candidate_rank_key, reverse=True)
                best = strong_cands[0]
                if len(strong_cands) >= 2:
                    alt = strong_cands[1]
                    if abs(_select_value(best) - _select_value(alt)) <= 0.03 and float(alt["mse"]) < float(best["mse"]):
                        best = alt
                if PREFER_HPOS_WHEN_CLOSE:
                    h_close = max([c for c in strong_cands if c["flag"] == FLAG_HPOS], key=_candidate_rank_key, default=None)
                    if h_close is not None and best["flag"] == FLAG_Q4:
                        best_sel = max(float(_select_value(best)), EPS_CONF)
                        if _select_value(h_close) >= HPOS_CLOSE_SEL_RATIO * best_sel or float(best["score"]) - float(h_close["score"]) <= HPOS_CLOSE_SCORE_GAP:
                            best = h_close
                band[i : i + BLOCKSIZE, j : j + BLOCKSIZE] = best["block"]
                flag_list.append(int(best["flag"]))
                hpos_list.append(int(best.get("hpos_idx", HPOS_NONE)))
            else:
                flag_list.append(FLAG_SKIP)
                hpos_list.append(HPOS_NONE)
        except Exception:
            flag_list.append(FLAG_SKIP)
            hpos_list.append(HPOS_NONE)

    watermarked_img = host_img_orig.copy()
    for ch in HOST_CHANNELS:
        rec = _dwt_merge_4bands(bands_by_ch[ch], DWT_WAVELET)[:H0, :W0]
        out_channel = host_img_orig[:, :, ch].astype(np.float64).copy()
        out_channel[:H0, :W0] = np.clip(np.rint(rec), 0, 255)
        watermarked_img[:, :, ch] = out_channel.astype(np.uint8)

    q4_used, hpos_used, skip_used = _flag_stats(flag_list)
    wm_meta.update(
        {
            "structured_repetition_enabled": True,
            "structured_repeat_factor": int(repeat_factor),
            "structured_usable_blocks": int(usable_blocks),
            "dwt_bands": list(DWT_BANDS),
            "q4_used": int(q4_used),
            "hpos_used": int(hpos_used),
            "skip_used": int(skip_used),
            "total_flags": int(len(flag_list)),
            "parameter_snapshot": _to_json_safe(get_current_params()),
            "logistic_removed": True,
            "rsa_removed": True,
            "semi_blind_side_info": True,
        }
    )
    _save_embed_info(embed_info_path, flag_list, hpos_list, wm_meta)
    return watermarked_img


def embed_full(host_path: str, watermark_path: str, output_path: str, embed_info_path: str, arnold_iterations: int | None = None) -> np.ndarray:
    host_img = load_color_image_24bit_safe(host_path, size=None)
    if host_img.shape[:2] != (HOST_SIZE, HOST_SIZE):
        raise ValueError(f"Host image must be exactly {HOST_SIZE}x{HOST_SIZE}. Got {host_img.shape[1]}x{host_img.shape[0]}.")
    img_final = _embed_main_in_image(host_img, watermark_path, embed_info_path, arnold_iterations)
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not cv2.imwrite(output_path, img_final):
        raise RuntimeError(f"Could not write watermarked image: {output_path}")
    return img_final


def _extract_main_from_work_color(work_img_u8: np.ndarray, flags, hpos_list, schedule, q_h01: float, payload_len: int):
    bands_by_ch = {ch: _dwt_split_4bands(work_img_u8[:, :, ch].astype(np.float64), DWT_WAVELET) for ch in HOST_CHANNELS}
    L = int(payload_len)
    n = min(len(schedule), len(flags), len(hpos_list))
    vote_lists = [[] for _ in range(L)]
    llr_lists = [[] for _ in range(L)]
    mode_lists = [[] for _ in range(L)]
    hard_vote_lists = [[] for _ in range(L)]

    for idx in range(n):
        kpos, block_pos = schedule[idx]
        ch, band_name, i, j = block_pos
        flag = int(flags[idx])
        if flag == FLAG_SKIP:
            continue
        block = bands_by_ch[ch][band_name][i : i + BLOCKSIZE, j : j + BLOCKSIZE]
        try:
            Hh, Qm = linalg.hessenberg(block, calc_q=True)
            if flag == FLAG_Q4:
                llr = _norm_llr_q4(_llr_q4_from_Q(Qm))
                mode_weight, llr_min, llr_scale = Q_VOTE_WEIGHT, MIN_LLR_WEIGHT_Q, Q_LLR_SCALE
            elif flag == FLAG_HPOS:
                hpos_idx = int(hpos_list[idx]) if idx < len(hpos_list) else HPOS_NONE
                if not (0 <= hpos_idx < len(HPOS_CANDIDATES)):
                    continue
                _, pos = HPOS_CANDIDATES[hpos_idx]
                llr = _norm_llr_hpos(_llr_hpos_from_H(Hh, q_h01, pos), q_h01)
                mode_weight, llr_min, llr_scale = H_VOTE_WEIGHT, MIN_LLR_WEIGHT_H, H_LLR_SCALE
            else:
                continue
        except Exception:
            continue
        a = abs(llr)
        if a < llr_min:
            continue
        w = float(mode_weight) * math.tanh(a / max(float(llr_scale), EPS_CONF))
        vote_lists[kpos].append(w if llr >= 0.0 else -w)
        llr_lists[kpos].append(llr)
        mode_lists[kpos].append(flag)
        hard_vote_lists[kpos].append(1 if llr >= 0.0 else -1)

    sum_vote = np.zeros(L, dtype=np.float64)
    sum_absw = np.zeros(L, dtype=np.float64)
    sum_llr = np.zeros(L, dtype=np.float64)
    sum_count = np.zeros(L, dtype=np.float64)

    for kpos in range(L):
        pairs = list(zip(vote_lists[kpos], llr_lists[kpos], mode_lists[kpos], hard_vote_lists[kpos]))
        if not pairs:
            continue
        if TOPK_VOTES_PER_BIT is not None and TOPK_VOTES_PER_BIT > 0 and len(pairs) > int(TOPK_VOTES_PER_BIT):
            pairs = sorted(pairs, key=lambda p: abs(p[0]), reverse=True)[: int(TOPK_VOTES_PER_BIT)]
        q_votes = [p[0] for p in pairs if int(p[2]) == FLAG_Q4]
        h_votes = [p[0] for p in pairs if int(p[2]) == FLAG_HPOS]
        q_sum = float(np.sum(q_votes)) if q_votes else 0.0
        h_sum = float(np.sum(h_votes)) if h_votes else 0.0
        if q_votes and h_votes and q_sum * h_sum < 0.0:
            if abs(q_sum) >= 1.10 * abs(h_sum):
                fused_vote = q_sum + 0.15 * h_sum
            elif abs(h_sum) >= 1.60 * abs(q_sum):
                fused_vote = 0.25 * q_sum + h_sum
            else:
                fused_vote = 0.65 * q_sum + 0.30 * h_sum
        else:
            fused_vote = q_sum + h_sum
        hard_sum = float(np.sum([p[3] for p in pairs]))
        if HARD_MAJORITY_ON_AMBIGUOUS and len(pairs) >= HARD_MAJORITY_MIN_COUNT and abs(fused_vote) < EXTRACT_AMBIGUITY_THRESH and hard_sum != 0.0:
            fused_vote = hard_sum
        sum_vote[kpos] = fused_vote
        sum_absw[kpos] = float(np.sum([abs(p[0]) for p in pairs]))
        sum_llr[kpos] = float(np.sum([p[1] for p in pairs]))
        sum_count[kpos] = float(len(pairs))

    valid = sum_count > 0
    out_bits = np.zeros(L, dtype=np.uint8)
    out_bits[valid & (sum_vote > 0.0)] = 1
    ties = valid & (sum_vote == 0.0)
    out_bits[ties] = (sum_llr[ties] >= 0.0).astype(np.uint8)
    strength = np.zeros(L, dtype=np.float64)
    strength[valid] = np.abs(sum_vote[valid]) / (sum_absw[valid] + EPS_CONF)
    coverage = float(np.mean(valid.astype(np.float64))) if L > 0 else 0.0
    conf = (float(np.mean(strength[valid])) if np.any(valid) else 0.0) * coverage
    return out_bits, conf


def _extract_main_from_image_array(wm_img: np.ndarray, embed_info_path: str):
    flags, hpos_list, payload_meta = _load_embed_info(embed_info_path)
    if len(flags) == 0:
        raise ValueError(f"Embed info is empty: {embed_info_path}")
    payload_len = _payload_len_from_meta(payload_meta)
    dwt_bands = tuple(payload_meta.get("dwt_bands", DWT_BANDS))
    _, H0, W0 = _crop_for_dwt(wm_img[:, :, 0], DWT_LEVEL, BLOCKSIZE)
    work_raw_u8 = wm_img[:H0, :W0, :].copy()
    bands_by_ch = {ch: _dwt_split_4bands(work_raw_u8[:, :, ch].astype(np.float64), DWT_WAVELET) for ch in HOST_CHANNELS}
    total_blocks = sum(
        (bands_by_ch[ch][b].shape[0] // BLOCKSIZE) * (bands_by_ch[ch][b].shape[1] // BLOCKSIZE)
        for ch in HOST_CHANNELS
        for b in dwt_bands
    )
    blocks = generate_color_subband_block_indices(bands_by_ch, HOST_CHANNELS, dwt_bands, BLOCKSIZE, total_blocks, PRIVATE_KEY)
    schedule, _, _ = make_structured_repetition_schedule(blocks, payload_len)
    best_bits, best_conf = _extract_main_from_work_color(work_raw_u8, flags, hpos_list, schedule, max(float(H01_Q), 1e-6), payload_len)
    wm_rec = reconstruct_binary_watermark_from_payload_bits(best_bits, payload_meta)
    return wm_rec, best_conf, payload_meta


def extract_main(watermarked_path: str, embed_info_path: str, output_path: str | None = None):
    wm_img = load_color_image_24bit_safe(watermarked_path, size=None)
    wm_rec, best_conf, payload_meta = _extract_main_from_image_array(wm_img, embed_info_path)
    if output_path is not None:
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not cv2.imwrite(output_path, wm_rec):
            raise RuntimeError(f"Could not write extracted watermark: {output_path}")
    return wm_rec, best_conf, payload_meta


def extract_message_bits(image_path: str, embed_info_path: str) -> np.ndarray:
    wm_rec, _, _ = extract_main(image_path, embed_info_path, output_path=None)
    return (wm_rec >= WM_BIN_THRESH).astype(bool).reshape(-1)


# -----------------------------------------------------------------------------
# Metrics and utility helpers
# -----------------------------------------------------------------------------
def watermark_metrics(extracted_watermark_bin, original_watermark_bin):
    ori = force_binary_watermark_exact(original_watermark_bin, WM_SIZE, WM_BIN_THRESH)
    ext = force_binary_watermark_exact(extracted_watermark_bin, WM_SIZE, WM_BIN_THRESH)
    ori_bits = (ori >= WM_BIN_THRESH).astype(np.uint8).reshape(-1)
    ext_bits = (ext >= WM_BIN_THRESH).astype(np.uint8).reshape(-1)
    ori_vec, ext_vec = ori_bits.astype(np.float64), ext_bits.astype(np.float64)
    denom = np.sqrt(np.sum(ori_vec ** 2) * np.sum(ext_vec ** 2))
    nc = float(np.sum(ori_vec * ext_vec) / denom) if denom != 0 else float(np.array_equal(ori_bits, ext_bits))
    oc, ec = ori_vec - np.mean(ori_vec), ext_vec - np.mean(ext_vec)
    denom_ncc = np.sqrt(np.sum(oc ** 2) * np.sum(ec ** 2))
    ncc = float(np.sum(oc * ec) / denom_ncc) if denom_ncc != 0 else float(np.array_equal(ori_bits, ext_bits))
    ber = float(np.mean(ori_bits != ext_bits))
    return nc, ncc, ber


_watermark_metrics = watermark_metrics


def calculate_metrics(original, watermarked, extracted_watermark_bin, original_watermark_bin):
    psnr_value = float("nan")
    ssim_value = float("nan")
    nc_value = ncc_value = ber_value = float("nan")
    if original is not None and watermarked is not None and original.shape == watermarked.shape:
        mse = np.mean((original.astype(np.float64) - watermarked.astype(np.float64)) ** 2)
        psnr_value = float("inf") if mse == 0 else float(20 * np.log10(255.0 / np.sqrt(mse)))
        ssim_value = float(ssim(original.astype(np.uint8), watermarked.astype(np.uint8), data_range=255, channel_axis=-1))
    if original_watermark_bin is not None and extracted_watermark_bin is not None:
        nc_value, ncc_value, ber_value = watermark_metrics(extracted_watermark_bin, original_watermark_bin)
    return psnr_value, ssim_value, nc_value, ncc_value, ber_value


def find_host_images(input_folder: str, watermark_path: str | None = None, exts: Iterable[str] | None = None) -> List[str]:
    exts = [".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"] if exts is None else [e.lower() for e in exts]
    watermark_abs = os.path.abspath(watermark_path) if watermark_path else None
    image_paths = []
    for fn in os.listdir(input_folder):
        if os.path.splitext(fn)[1].lower() not in exts:
            continue
        full_path = os.path.abspath(os.path.join(input_folder, fn))
        if watermark_abs and full_path == watermark_abs:
            continue
        try:
            img = load_color_image_24bit_safe(full_path, size=None)
            if img.shape[:2] == (HOST_SIZE, HOST_SIZE):
                image_paths.append(full_path)
        except Exception:
            pass
    return sorted(image_paths)
