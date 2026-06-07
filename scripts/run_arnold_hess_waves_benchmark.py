#!/usr/bin/env python3
"""One-command WAVES-style benchmark runner for the Arnold-Hess proposal.

This script makes the Arnold-Hess integration self-contained for reproducible
WAVES-style distortion benchmarking:
  1) download/open-source host data, default COCO val2017;
  2) prepare 512x512 hosts;
  3) create/use a fixed 64x64 binary watermark;
  4) embed Arnold-Hess into WAVES main/ folders;
  5) generate WAVES-style attacked folders for real and watermarked images;
  6) decode messages;
  7) compute lightweight image metrics;
  8) export comparison-ready CSV summaries and plots.

The default run is CPU friendly and focuses on the distortion portion of WAVES.
Regeneration/adversarial WAVES attacks need heavy external models and are not
run by this lightweight command.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

COCO_VAL2017_URL = "https://images.cocodataset.org/zips/val2017.zip"
METHOD = "arnold_hess"
DEFAULT_DATASET = "mscoco"
DEFAULT_STRENGTHS = ["0.2", "0.4", "0.6", "0.8", "1.0"]

# WAVES distortion-single names and the corresponding primitive distortion.
ATTACK_TO_PRIMITIVES: Dict[str, Tuple[str, ...]] = {
    "distortion_single_rotation": ("rotation",),
    "distortion_single_resizedcrop": ("resizedcrop",),
    "distortion_single_erasing": ("erasing",),
    "distortion_single_brightness": ("brightness",),
    "distortion_single_contrast": ("contrast",),
    "distortion_single_blurring": ("blurring",),
    "distortion_single_noise": ("noise",),
    "distortion_single_jpeg": ("compression",),
    "distortion_combo_geometric": ("rotation", "resizedcrop", "erasing"),
    "distortion_combo_photometric": ("brightness", "contrast"),
    "distortion_combo_degradation": ("blurring", "noise", "compression"),
    "distortion_combo_all": (
        "rotation",
        "resizedcrop",
        "erasing",
        "brightness",
        "contrast",
        "blurring",
        "noise",
        "compression",
    ),
}

DISTORTION_SINGLE_ATTACKS = [
    "distortion_single_rotation",
    "distortion_single_resizedcrop",
    "distortion_single_erasing",
    "distortion_single_brightness",
    "distortion_single_contrast",
    "distortion_single_blurring",
    "distortion_single_noise",
    "distortion_single_jpeg",
]

DISTORTION_COMBO_ATTACKS = [
    "distortion_combo_geometric",
    "distortion_combo_photometric",
    "distortion_combo_degradation",
    "distortion_combo_all",
]

QUALITY_KEYS = [
    "legacy_fid",
    "clip_fid",
    "psnr",
    "ssim",
    "nmi",
    "lpips",
    "watson",
    "aesthetics",
    "artifacts",
    "clip_score",
]

WATERMARK_METHOD_KEYS = ["tree_ring", "stable_sig", "stegastamp", METHOD]


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_csv(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def strength_label(value: str | float) -> str:
    # Keep stable, parseable folder names such as 0.2 and 1.0.
    return f"{float(value):.3f}".rstrip("0").rstrip(".") if float(value) % 1 else f"{float(value):.1f}"


def run_cmd(cmd: Sequence[str], env: Dict[str, str], dry_run: bool = False) -> None:
    text = " ".join(str(c) for c in cmd)
    log(f"[CMD] {text}")
    if dry_run:
        return
    subprocess.run(list(cmd), cwd=str(PROJECT_ROOT), env=env, check=True)


def write_env_file(env_path: Path, env: Dict[str, str]) -> None:
    lines = [
        f"DATA_DIR={env['DATA_DIR']}",
        f"RESULT_DIR={env['RESULT_DIR']}",
        f"CACHE_DIR={env['CACHE_DIR']}",
        f"MODEL_DIR={env['MODEL_DIR']}",
        f"ARNOLD_HESS_WM_PATH={env['ARNOLD_HESS_WM_PATH']}",
        f"ARNOLD_HESS_WM_SIZE={env.get('ARNOLD_HESS_WM_SIZE', '64')}",
        f"ARNOLD_HESS_REPEAT_LIMIT={env.get('ARNOLD_HESS_REPEAT_LIMIT', '1')}",
        f"ARNOLD_HESS_ULTRA_FAST_CANDIDATE_SCORING={env.get('ARNOLD_HESS_ULTRA_FAST_CANDIDATE_SCORING', '1')}",
        f"ARNOLD_HESS_FAST_CANDIDATE_SCORING={env.get('ARNOLD_HESS_FAST_CANDIDATE_SCORING', '1')}",
        f"WAVES_LIMIT={env['WAVES_LIMIT']}",
        f"WAVES_SUBSET_LIMIT={env['WAVES_SUBSET_LIMIT']}",
    ]
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def download_with_progress(url: str, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    if output_path.exists() and output_path.stat().st_size > 0:
        log(f"[OK] Reusing existing download: {output_path}")
        return

    log(f"[DOWNLOAD] {url}")
    last = [time.time()]

    def hook(block_num: int, block_size: int, total_size: int) -> None:
        now = time.time()
        if now - last[0] < 3:
            return
        last[0] = now
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            log(f"  {downloaded / (1024**2):.1f} / {total_size / (1024**2):.1f} MiB ({pct:.1f}%)")
        else:
            log(f"  {downloaded / (1024**2):.1f} MiB")

    urllib.request.urlretrieve(url, output_path, reporthook=hook)
    log(f"[DONE] Downloaded to {output_path}")


def download_and_extract_coco(raw_dir: Path, skip_download: bool = False) -> Path:
    coco_dir = ensure_dir(raw_dir / "coco")
    zip_path = coco_dir / "val2017.zip"
    val_dir = coco_dir / "val2017"

    if not val_dir.is_dir() or len(list(val_dir.glob("*.jpg"))) == 0:
        if skip_download:
            raise RuntimeError(f"COCO val2017 is missing at {val_dir}, and --skip-download was used.")
        download_with_progress(COCO_VAL2017_URL, zip_path)
        log(f"[EXTRACT] {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(coco_dir)
    else:
        log(f"[OK] Reusing extracted COCO val2017: {val_dir}")

    return val_dir


def create_synthetic_hosts(raw_dir: Path, limit: int, seed: int = 123) -> Path:
    out_dir = ensure_dir(raw_dir / "synthetic512")
    rng = np.random.default_rng(seed)
    for idx in range(limit):
        path = out_dir / f"{idx}.png"
        if path.exists():
            continue
        # Smooth-ish deterministic RGB pattern, not just random noise.
        y, x = np.mgrid[0:512, 0:512]
        base = np.zeros((512, 512, 3), dtype=np.uint8)
        base[..., 0] = (x / 2 + idx * 11) % 256
        base[..., 1] = (y / 2 + idx * 17) % 256
        base[..., 2] = ((x + y) / 4 + idx * 23) % 256
        noise = rng.normal(0, 8, size=base.shape)
        arr = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        Image.fromarray(arr, mode="RGB").save(path)
    return out_dir


def prepare_512_hosts(src_dir: Path, out_dir: Path, limit: int, force: bool = False) -> Path:
    ensure_dir(out_dir)
    existing = sorted(out_dir.glob("*.png"))
    if len(existing) >= limit and not force:
        log(f"[OK] Reusing {len(existing)} prepared 512x512 hosts in {out_dir}")
        return out_dir
    if force and out_dir.exists():
        shutil.rmtree(out_dir)
        ensure_dir(out_dir)

    image_paths = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}])
    if not image_paths:
        raise RuntimeError(f"No images found in {src_dir}")

    count = 0
    for p in image_paths:
        if count >= limit:
            break
        try:
            img = Image.open(p).convert("RGB")
            img = ImageOps.fit(img, (512, 512), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            img.save(out_dir / f"{count}.png")
            count += 1
        except Exception as exc:
            log(f"[WARN] Skipped {p}: {exc}")

    if count < limit:
        raise RuntimeError(f"Only prepared {count} images, but limit={limit}")
    log(f"[DONE] Prepared {count} 512x512 hosts in {out_dir}")
    return out_dir


def create_default_watermark(path: Path, seed: int = 2026, size: int = 64) -> Path:
    ensure_dir(path.parent)
    if path.exists():
        log(f"[OK] Reusing watermark: {path}")
        return path
    rng = np.random.default_rng(seed)
    wm = (rng.random((size, size)) > 0.5).astype(np.uint8) * 255
    # Add a small deterministic border/cross so the watermark is easy to inspect visually.
    wm[:2, :] = 255
    wm[-2:, :] = 255
    wm[:, :2] = 255
    wm[:, -2:] = 255
    mid = size // 2
    wm[max(0, mid-1):min(size, mid+1), :] = 255 - wm[max(0, mid-1):min(size, mid+1), :]
    wm[:, max(0, mid-1):min(size, mid+1)] = 255 - wm[:, max(0, mid-1):min(size, mid+1)]
    Image.fromarray(wm, mode="L").save(path)
    log(f"[DONE] Created fixed {size}x{size} binary watermark: {path}")
    return path


def primitive_absolute_strength(primitive: str, relative_strength: float) -> float:
    if not 0 <= relative_strength <= 1:
        raise ValueError("strength must be in [0, 1]")
    ranges = {
        "rotation": (0.0, 45.0),
        "resizedcrop": (1.0, 0.5),
        "erasing": (0.0, 0.25),
        "brightness": (1.0, 2.0),
        "contrast": (1.0, 2.0),
        "blurring": (0.0, 20.0),
        "noise": (0.0, 0.1),
        "compression": (90.0, 10.0),
    }
    a, b = ranges[primitive]
    return relative_strength * (b - a) + a


def attack_param_summary(attack: str, relative_strength: float) -> str:
    parts = []
    for primitive in ATTACK_TO_PRIMITIVES[attack]:
        abs_value = primitive_absolute_strength(primitive, relative_strength)
        if primitive == "compression":
            parts.append(f"jpeg_quality={int(round(abs_value))}")
        elif primitive == "rotation":
            parts.append(f"rotation_deg={abs_value:.2f}")
        elif primitive == "resizedcrop":
            parts.append(f"crop_area={abs_value:.3f}")
        elif primitive == "erasing":
            parts.append(f"erase_area={abs_value:.3f}")
        elif primitive in {"brightness", "contrast"}:
            parts.append(f"{primitive}_factor={abs_value:.3f}")
        elif primitive == "blurring":
            parts.append(f"blur_radius={abs_value:.2f}")
        elif primitive == "noise":
            parts.append(f"noise_std={abs_value:.4f}")
    return ";".join(parts)


def apply_pure_pil_primitive(img: Image.Image, primitive: str, relative_strength: float, seed: int) -> Image.Image:
    rng = random.Random(seed)
    img = img.convert("RGB")
    abs_strength = primitive_absolute_strength(primitive, relative_strength)

    if primitive == "rotation":
        return img.rotate(float(abs_strength), resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))

    if primitive == "resizedcrop":
        area_scale = float(abs_strength)
        w, h = img.size
        crop = max(1, int(round(math.sqrt(area_scale) * min(w, h))))
        if crop >= min(w, h):
            return img.copy()
        left = rng.randint(0, w - crop)
        top = rng.randint(0, h - crop)
        return img.crop((left, top, left + crop, top + crop)).resize((w, h), Image.Resampling.BICUBIC)

    if primitive == "erasing":
        erase_area = float(abs_strength)
        if erase_area <= 0:
            return img.copy()
        w, h = img.size
        side = max(1, int(round(math.sqrt(erase_area) * min(w, h))))
        left = rng.randint(0, max(0, w - side))
        top = rng.randint(0, max(0, h - side))
        out = img.copy()
        patch = Image.new("RGB", (side, side), (0, 0, 0))
        out.paste(patch, (left, top))
        return out

    if primitive == "brightness":
        return ImageEnhance.Brightness(img).enhance(float(abs_strength))

    if primitive == "contrast":
        return ImageEnhance.Contrast(img).enhance(float(abs_strength))

    if primitive == "blurring":
        return img.filter(ImageFilter.GaussianBlur(float(abs_strength)))

    if primitive == "noise":
        arr = np.asarray(img, dtype=np.float32) / 255.0
        np_rng = np.random.default_rng(seed)
        arr = np.clip(arr + np_rng.normal(0.0, float(abs_strength), size=arr.shape), 0.0, 1.0)
        return Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="RGB")

    if primitive == "compression":
        import io

        quality = int(round(abs_strength))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    raise ValueError(f"Unknown primitive distortion: {primitive}")


def apply_attack(img: Image.Image, attack: str, strength: float, seed: int, attack_engine: str = "pure") -> Image.Image:
    primitives = ATTACK_TO_PRIMITIVES[attack]
    if attack_engine in {"repo", "auto"} and len(primitives) == 1:
        try:
            from distortions.distortions import apply_distortion

            primitive_for_repo = "compression" if primitives[0] == "compression" else primitives[0]
            return apply_distortion(
                [img.convert("RGB")],
                distortion_type=primitive_for_repo,
                strength=float(strength),
                distortion_seed=seed,
                same_operation=True,
                relative_strength=True,
                return_image=True,
            )[0].convert("RGB")
        except Exception as exc:
            if attack_engine == "repo":
                raise
            log(f"[WARN] Repo distortion engine unavailable; falling back to pure PIL: {exc}")

    out = img.convert("RGB")
    for offset, primitive in enumerate(primitives):
        out = apply_pure_pil_primitive(out, primitive, strength, seed + 1009 * offset)
    return out


def generate_attacks(data_dir: Path, dataset: str, attacks: Sequence[str], strengths: Sequence[str], limit: int, attack_engine: str, seed: int, force: bool = False, sources: Sequence[str] = (METHOD,)) -> None:
    for attack in attacks:
        if attack not in ATTACK_TO_PRIMITIVES:
            raise ValueError(f"Unknown attack {attack}. Valid attacks: {sorted(ATTACK_TO_PRIMITIVES)}")
        for st in strengths:
            st_label = strength_label(st)
            st_float = float(st)
            for source in sources:
                in_dir = data_dir / "main" / dataset / source
                out_dir = data_dir / "attacked" / dataset / f"{attack}-{st_label}-{source}"
                ensure_dir(out_dir)
                existing = list(out_dir.glob("*.png"))
                if len(existing) >= limit and not force:
                    log(f"[OK] Reusing attack folder: {out_dir}")
                    continue
                if force and out_dir.exists():
                    shutil.rmtree(out_dir)
                    ensure_dir(out_dir)
                log(f"[ATTACK] {attack} strength={st_label} source={source}")
                count = 0
                for idx in range(limit):
                    in_path = in_dir / f"{idx}.png"
                    if not in_path.exists():
                        continue
                    img = Image.open(in_path).convert("RGB")
                    out = apply_attack(img, attack, st_float, seed + idx, attack_engine=attack_engine)
                    out.save(out_dir / f"{idx}.png")
                    count += 1
                log(f"[DONE] {out_dir}: {count} images")


def write_clean_metric_placeholder(result_dir: Path, dataset: str, source: str, limit: int) -> None:
    dataset_dir = ensure_dir(result_dir / dataset)
    path = dataset_dir / f"{source}-metric.json"
    if path.exists():
        return
    data = {str(i): {key: None for key in QUALITY_KEYS} for i in range(limit)}
    path.write_text(json.dumps(data), encoding="utf-8")


def decode_all(env: Dict[str, str], data_dir: Path, dataset: str, attacks: Sequence[str], strengths: Sequence[str], limit: int, subset_limit: int, dry_run: bool = False) -> None:
    paths = [
        data_dir / "main" / dataset / "real",
        data_dir / "main" / dataset / METHOD,
    ]
    for attack in attacks:
        for st in strengths:
            st_label = strength_label(st)
            paths.append(data_dir / "attacked" / dataset / f"{attack}-{st_label}-real")
            paths.append(data_dir / "attacked" / dataset / f"{attack}-{st_label}-{METHOD}")
    for path in paths:
        run_cmd(
            [
                sys.executable,
                "scripts/decode.py",
                "--path",
                str(path),
                "--mode",
                METHOD,
                "--limit",
                str(limit),
                "--subset-limit",
                str(subset_limit),
            ],
            env=env,
            dry_run=dry_run,
        )


def compute_metrics(env: Dict[str, str], data_dir: Path, dataset: str, attacks: Sequence[str], strengths: Sequence[str], limit: int, subset_limit: int, dry_run: bool = False) -> None:
    for attack in attacks:
        for st in strengths:
            st_label = strength_label(st)
            # Metrics for watermarked images are required for removal quality.
            # Metrics for real images are also computed so combined/spoofing-style
            # quality tables can be added later without rerunning attacks.
            for source in [METHOD, "real"]:
                path = data_dir / "attacked" / dataset / f"{attack}-{st_label}-{source}"
                run_cmd(
                    [
                        sys.executable,
                        "scripts/metric.py",
                        "--path",
                        str(path),
                        "--metrics",
                        "psnr,ssim,nmi",
                        "--limit",
                        str(limit),
                        "--subset-limit",
                        str(subset_limit),
                    ],
                    env=env,
                    dry_run=dry_run,
                )


def decode_array_from_string_local(encoded_string: str) -> np.ndarray:
    import base64
    import gzip

    try:
        import orjson
        loads = orjson.loads
    except Exception:
        loads = json.loads

    raw = gzip.decompress(base64.b64decode(encoded_string))
    meta_encoded, array_bytes = raw.split(b"\x00", 1)
    meta = loads(meta_encoded)
    return np.frombuffer(array_bytes, dtype=meta["dtype"]).reshape(meta["shape"])


def encode_array_to_string_local(array: np.ndarray) -> str:
    import base64
    import gzip

    try:
        import orjson
        dumps = orjson.dumps
    except Exception:
        dumps = lambda obj: json.dumps(obj).encode("utf-8")

    array = np.asarray(array)
    meta = dumps({"shape": array.shape, "dtype": str(array.dtype)})
    combined = meta + b"\x00" + array.tobytes()
    return base64.b64encode(gzip.compress(combined)).decode("utf-8")


def result_json_path(result_dir: Path, dataset: str, source: str, kind: str = "decode") -> Path:
    return result_dir / dataset / f"{source}-{kind}.json"


def attacked_source_name(attack: str, strength: str, source: str) -> str:
    return f"{attack}-{strength_label(strength)}-{source}"


def decode_dir_arnold(data_dir: Path, result_dir: Path, dataset: str, source_name: str, image_dir: Path, limit: int, quiet: bool = False) -> None:
    from watermarks.arnold_hess_adapter import decode_message

    side_dir = data_dir / "side_info" / dataset / METHOD
    out_path = result_json_path(result_dir, dataset, source_name, "decode")
    ensure_dir(out_path.parent)
    if out_path.exists():
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    else:
        data = {}
    for idx in range(limit):
        data.setdefault(str(idx), {m: None for m in WATERMARK_METHOD_KEYS})

    done = 0
    for idx in range(limit):
        # Always overwrite Arnold-Hess direct decode to avoid stale results when
        # watermark size or side-info parameters change between runs.
        img_path = image_dir / f"{idx}.png"
        side_path = side_dir / f"{idx}.npz"
        if not img_path.exists() or not side_path.exists():
            continue
        msg = decode_message(str(img_path), str(side_path)).astype(bool)
        data[str(idx)][METHOD] = encode_array_to_string_local(msg)
        done += 1
    out_path.write_text(json.dumps(data), encoding="utf-8")
    if not quiet:
        log(f"[DECODE] {source_name}: {done}/{limit} -> {out_path}")


def decode_all_direct(data_dir: Path, result_dir: Path, dataset: str, attacks: Sequence[str], strengths: Sequence[str], limit: int, include_real_attacks: bool = False) -> None:
    decode_dir_arnold(data_dir, result_dir, dataset, "real", data_dir / "main" / dataset / "real", limit)
    decode_dir_arnold(data_dir, result_dir, dataset, METHOD, data_dir / "main" / dataset / METHOD, limit)
    for attack in attacks:
        for st in strengths:
            st_label = strength_label(st)
            decode_dir_arnold(data_dir, result_dir, dataset, attacked_source_name(attack, st_label, METHOD), data_dir / "attacked" / dataset / attacked_source_name(attack, st_label, METHOD), limit)
            if include_real_attacks:
                decode_dir_arnold(data_dir, result_dir, dataset, attacked_source_name(attack, st_label, "real"), data_dir / "attacked" / dataset / attacked_source_name(attack, st_label, "real"), limit)


def compute_pair_metrics(clean_img: Image.Image, attacked_img: Image.Image) -> Dict[str, float]:
    from skimage.metrics import normalized_mutual_information, peak_signal_noise_ratio, structural_similarity

    a = np.asarray(clean_img.convert("RGB"))
    b = np.asarray(attacked_img.convert("RGB"))
    return {
        "psnr": float(peak_signal_noise_ratio(a, b, data_range=255)),
        "ssim": float(structural_similarity(a, b, channel_axis=2, data_range=255)),
        "nmi": float(normalized_mutual_information(a, b)),
    }


def compute_metric_dir_lightweight(data_dir: Path, result_dir: Path, dataset: str, source_name: str, attacked_dir: Path, clean_source: str, limit: int, subset_limit: int) -> None:
    clean_dir = data_dir / "main" / dataset / clean_source
    out_path = result_json_path(result_dir, dataset, source_name, "metric")
    ensure_dir(out_path.parent)
    if out_path.exists():
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    else:
        data = {}
    for idx in range(limit):
        data.setdefault(str(idx), {key: None for key in QUALITY_KEYS})

    measured = 0
    for idx in range(subset_limit):
        if all(data.get(str(idx), {}).get(k) is not None for k in ["psnr", "ssim", "nmi"]):
            measured += 1
            continue
        clean_path = clean_dir / f"{idx}.png"
        attacked_path = attacked_dir / f"{idx}.png"
        if not clean_path.exists() or not attacked_path.exists():
            continue
        vals = compute_pair_metrics(Image.open(clean_path), Image.open(attacked_path))
        data[str(idx)].update(vals)
        measured += 1
    out_path.write_text(json.dumps(data), encoding="utf-8")
    log(f"[METRIC] {source_name}: {measured}/{subset_limit} -> {out_path}")


def compute_metrics_direct(data_dir: Path, result_dir: Path, dataset: str, attacks: Sequence[str], strengths: Sequence[str], limit: int, subset_limit: int, include_real_attacks: bool = False) -> None:
    for attack in attacks:
        for st in strengths:
            st_label = strength_label(st)
            src = attacked_source_name(attack, st_label, METHOD)
            compute_metric_dir_lightweight(data_dir, result_dir, dataset, src, data_dir / "attacked" / dataset / src, METHOD, limit, subset_limit)
            if include_real_attacks:
                real_src = attacked_source_name(attack, st_label, "real")
                compute_metric_dir_lightweight(data_dir, result_dir, dataset, real_src, data_dir / "attacked" / dataset / real_src, "real", limit, subset_limit)


def load_metric_values(metric_json: Path, metric: str, subset_limit: int) -> Tuple[Optional[float], Optional[float]]:
    if not metric_json.exists():
        return None, None
    data = json.loads(metric_json.read_text(encoding="utf-8"))
    vals = []
    for i in range(subset_limit):
        value = data.get(str(i), {}).get(metric)
        if value is not None:
            vals.append(float(value))
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def classify_robustness(tpr: Optional[float]) -> str:
    if tpr is None:
        return "missing"
    if tpr >= 0.95:
        return "very_robust"
    if tpr >= 0.70:
        return "robust"
    if tpr >= 0.40:
        return "weak"
    return "broken"


def classify_quality(psnr: Optional[float]) -> str:
    if psnr is None:
        return "missing"
    if psnr >= 40:
        return "excellent"
    if psnr >= 35:
        return "good"
    if psnr >= 30:
        return "acceptable"
    return "visible_degradation"


def summarize_watermark_recovery(result_dir: Path, dataset: str, attack: str, strength: str, subset_limit: int, watermark_path: Path) -> Dict[str, Optional[float]]:
    from watermarks.arnold_hess_adapter import ground_truth_message

    target = ground_truth_message(str(watermark_path)).astype(bool).reshape(-1)
    decode_json = result_dir / dataset / f"{attack}-{strength_label(strength)}-{METHOD}-decode.json"
    if not decode_json.exists():
        return {"ber_mean": None, "bit_accuracy_mean": None, "nc_mean": None}
    data = json.loads(decode_json.read_text(encoding="utf-8"))
    bers, accs, ncs = [], [], []
    target01 = target.astype(np.float64)
    target_norm = float(np.sqrt(np.sum(target01 * target01)))
    for i in range(subset_limit):
        encoded = data.get(str(i), {}).get(METHOD)
        if encoded is None:
            continue
        pred = decode_array_from_string_local(encoded).astype(bool).reshape(-1)
        if pred.shape != target.shape:
            continue
        ber = float(np.mean(pred != target))
        pred01 = pred.astype(np.float64)
        pred_norm = float(np.sqrt(np.sum(pred01 * pred01)))
        nc = float(np.sum(pred01 * target01) / (pred_norm * target_norm + 1e-12))
        bers.append(ber)
        accs.append(1.0 - ber)
        ncs.append(nc)
    if not bers:
        return {"ber_mean": None, "bit_accuracy_mean": None, "nc_mean": None}
    return {
        "ber_mean": float(np.mean(bers)),
        "bit_accuracy_mean": float(np.mean(accs)),
        "nc_mean": float(np.mean(ncs)),
    }


def export_summary(result_dir: Path, dataset: str, attacks: Sequence[str], strengths: Sequence[str], subset_limit: int, watermark_path: Path) -> Path:
    # Import dev aggregation only after env variables and watermark are set.
    from dev.aggregate import clear_aggregated_cache, get_performance, get_quality

    clear_aggregated_cache()
    out_dir = ensure_dir(result_dir / dataset)
    csv_path = out_dir / f"{METHOD}_waves_distortion_summary.csv"
    rows: List[Dict[str, object]] = []
    for attack in attacks:
        for st in strengths:
            st_label = strength_label(st)
            st_float = float(st_label)
            perf = get_performance(dataset, METHOD, attack, st_float, mode="removal")
            qual = get_quality(dataset, METHOD, attack, st_float, mode="removal")
            recovery = summarize_watermark_recovery(result_dir, dataset, attack, st_label, subset_limit, watermark_path)
            metric_json = result_dir / dataset / f"{attack}-{st_label}-{METHOD}-metric.json"
            psnr_mean, psnr_std = load_metric_values(metric_json, "psnr", subset_limit)
            ssim_mean, ssim_std = load_metric_values(metric_json, "ssim", subset_limit)
            nmi_mean, nmi_std = load_metric_values(metric_json, "nmi", subset_limit)
            tpr001 = perf.get("low1000_1")
            tpr01 = perf.get("low100_1")
            auc = perf.get("auc_1")
            acc = perf.get("acc_1")
            rows.append(
                {
                    "dataset": dataset,
                    "method": METHOD,
                    "evaluation_mode": "removal",
                    "attack": attack,
                    "strength_relative": st_label,
                    "attack_parameters": attack_param_summary(attack, float(st_label)),
                    "tpr_at_0_1pct_fpr": tpr001,
                    "tpr_at_1pct_fpr": tpr01,
                    "auc": auc,
                    "mean_accuracy": acc,
                    "ber_mean": recovery["ber_mean"],
                    "bit_accuracy_mean": recovery["bit_accuracy_mean"],
                    "nc_mean": recovery["nc_mean"],
                    "psnr_mean": psnr_mean,
                    "psnr_std": psnr_std,
                    "ssim_mean": ssim_mean,
                    "ssim_std": ssim_std,
                    "nmi_mean": nmi_mean,
                    "nmi_std": nmi_std,
                    "robustness_class": classify_robustness(tpr001),
                    "visual_quality_class": classify_quality(psnr_mean),
                    "waves_metric_primary": "TPR@0.1%FPR",
                    "notes": f"WAVES-style distortion benchmark; repeat_limit={os.environ.get('ARNOLD_HESS_REPEAT_LIMIT', '1')}; candidate_scoring={'ultrafast' if os.environ.get('ARNOLD_HESS_ULTRA_FAST_CANDIDATE_SCORING') == '1' else 'fast/full'}; regeneration/adversarial attacks are not included in this lightweight runner.",
                }
            )

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log(f"[DONE] Summary CSV: {csv_path}")

    # Also export markdown table for easy manuscript copying.
    md_path = out_dir / f"{METHOD}_waves_distortion_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| Attack | Strength | TPR@0.1%FPR | AUC | BER | NC | PSNR | SSIM | Class |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            def fmt(x, nd=4):
                return "" if x is None else f"{float(x):.{nd}f}"
            f.write(
                f"| {r['attack']} | {r['strength_relative']} | {fmt(r['tpr_at_0_1pct_fpr'])} | {fmt(r['auc'])} | "
                f"{fmt(r['ber_mean'])} | {fmt(r['nc_mean'])} | {fmt(r['psnr_mean'], 2)} | {fmt(r['ssim_mean'])} | {r['robustness_class']} |\n"
            )
    log(f"[DONE] Markdown summary: {md_path}")

    try:
        make_plots(rows, out_dir)
    except Exception as exc:
        log(f"[WARN] Plot generation skipped: {exc}")
    return csv_path


def make_plots(rows: List[Dict[str, object]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    # Plot 1: robustness-vs-quality scatter, comparable to WAVES-style analysis.
    xs, ys, labels = [], [], []
    for r in rows:
        if r.get("psnr_mean") is None or r.get("tpr_at_0_1pct_fpr") is None:
            continue
        xs.append(float(r["psnr_mean"]))
        ys.append(float(r["tpr_at_0_1pct_fpr"]))
        labels.append(f"{r['attack'].replace('distortion_single_', '').replace('distortion_combo_', 'combo_')}:{r['strength_relative']}")
    if xs:
        plt.figure(figsize=(8, 6))
        plt.scatter(xs, ys)
        for x, y, label in zip(xs, ys, labels):
            plt.annotate(label, (x, y), fontsize=6, alpha=0.75)
        plt.xlabel("PSNR after attack (dB)")
        plt.ylabel("TPR@0.1%FPR")
        plt.title("Arnold-Hess WAVES-style robustness vs. quality")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = out_dir / f"{METHOD}_waves_psnr_vs_tpr.png"
        plt.savefig(path, dpi=200)
        plt.close()
        log(f"[DONE] Plot: {path}")

    # Plot 2: NC by strength for each attack.
    attacks = sorted({str(r["attack"]) for r in rows})
    for metric, ylabel, filename in [
        ("nc_mean", "Mean NC", f"{METHOD}_waves_nc_by_strength.png"),
        ("bit_accuracy_mean", "Mean bit accuracy", f"{METHOD}_waves_bit_accuracy_by_strength.png"),
        ("tpr_at_0_1pct_fpr", "TPR@0.1%FPR", f"{METHOD}_waves_tpr_by_strength.png"),
    ]:
        plt.figure(figsize=(8, 6))
        any_line = False
        for attack in attacks:
            sub = [r for r in rows if r["attack"] == attack and r.get(metric) is not None]
            if not sub:
                continue
            sub = sorted(sub, key=lambda r: float(r["strength_relative"]))
            plt.plot([float(r["strength_relative"]) for r in sub], [float(r[metric]) for r in sub], marker="o", label=attack.replace("distortion_single_", "").replace("distortion_combo_", "combo_"))
            any_line = True
        if any_line:
            plt.xlabel("Relative attack strength")
            plt.ylabel(ylabel)
            plt.title(f"Arnold-Hess {ylabel} under WAVES-style attacks")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=7)
            plt.tight_layout()
            path = out_dir / filename
            plt.savefig(path, dpi=200)
            log(f"[DONE] Plot: {path}")
        plt.close()


def resolve_attacks(preset: str, attacks_csv: Optional[str]) -> List[str]:
    if attacks_csv:
        return parse_csv(attacks_csv)
    if preset == "distortion-single":
        return list(DISTORTION_SINGLE_ATTACKS)
    if preset == "distortion-combo":
        return list(DISTORTION_COMBO_ATTACKS)
    if preset == "distortion-full":
        return list(DISTORTION_SINGLE_ATTACKS + DISTORTION_COMBO_ATTACKS)
    if preset == "lite":
        return ["distortion_single_jpeg", "distortion_single_noise", "distortion_single_blurring", "distortion_single_rotation"]
    raise ValueError(f"Unknown attack preset: {preset}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Arnold-Hess under a WAVES-style benchmark with one command.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=["mscoco", "custom", "diffusiondb", "dalle3"], help="WAVES dataset folder name to use.")
    parser.add_argument("--data-source", default="coco-val2017", choices=["coco-val2017", "local", "synthetic"], help="Host image source.")
    parser.add_argument("--input", default=None, help="Local image folder when --data-source local is used.")
    parser.add_argument("--limit", type=int, default=1000, help="Number of host images. Use 5000 for COCO val2017 paper-scale comparison.")
    parser.add_argument("--subset-limit", type=int, default=None, help="Metric subset size. Default: min(limit, 1000), matching WAVES-style quality subsampling.")
    parser.add_argument("--watermark", default=None, help="Optional binary square watermark path. If omitted, a fixed watermark is created automatically.")
    parser.add_argument("--watermark-size", type=int, default=64, help="Watermark width/height. Use 64 for paper-scale Arnold-Hess; smaller values are only for smoke/debug runs.")
    parser.add_argument("--repeat-limit", default="1", help="Maximum structured repetitions per watermark bit. Default 1 makes the benchmark runnable. Use full for the original all-capacity setting.")
    parser.add_argument("--candidate-scoring", default="ultrafast", choices=["ultrafast", "fast", "full"], help="ultrafast uses exact-block scoring only; fast is the original 4-test BSS; full adds extra stress tests.")
    parser.add_argument("--attack-preset", default="distortion-single", choices=["lite", "distortion-single", "distortion-combo", "distortion-full"], help="Which WAVES-style distortion attacks to run.")
    parser.add_argument("--attacks", default=None, help="Comma-separated attack names overriding --attack-preset.")
    parser.add_argument("--strengths", default=",".join(DEFAULT_STRENGTHS), help="Comma-separated relative strengths in [0,1].")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--result-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / "cache"))
    parser.add_argument("--raw-dir", default=str(PROJECT_ROOT / "raw"))
    parser.add_argument("--model-dir", default=str(PROJECT_ROOT / "decoders"))
    parser.add_argument("--attack-engine", default="pure", choices=["pure", "auto", "repo"], help="pure is self-contained; repo uses distortions/distortions.py when available.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-download", action="store_true", help="Do not download COCO; require existing raw/coco/val2017.")
    parser.add_argument("--force-prepare", action="store_true", help="Regenerate prepared 512x512 hosts.")
    parser.add_argument("--force-attacks", action="store_true", help="Regenerate attacked images.")
    parser.add_argument("--include-real-attacks", action="store_true", help="Also attack/decode/metric real images. Needed for spoofing/combined analysis, not for default removal comparison.")
    parser.add_argument("--use-subprocess-tools", action="store_true", help="Use original scripts/decode.py and scripts/metric.py subprocesses instead of faster in-process Arnold-Hess evaluation.")
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--skip-attacks", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    subset_limit = args.subset_limit if args.subset_limit is not None else min(args.limit, 1000)
    if subset_limit <= 0 or subset_limit > args.limit:
        raise ValueError("--subset-limit must be in [1, limit]")

    data_dir = ensure_dir(args.data_dir).resolve()
    result_dir = ensure_dir(args.result_dir).resolve()
    cache_dir = ensure_dir(args.cache_dir).resolve()
    raw_dir = ensure_dir(args.raw_dir).resolve()
    model_dir = ensure_dir(args.model_dir).resolve()

    watermark_path = Path(args.watermark).resolve() if args.watermark else (data_dir / "watermark" / f"arnold_hess_wm{args.watermark_size}.png").resolve()
    if not args.watermark:
        create_default_watermark(watermark_path, seed=args.seed, size=args.watermark_size)
    elif not watermark_path.exists():
        raise RuntimeError(f"Watermark does not exist: {watermark_path}")

    env = dict(os.environ)
    env.update(
        {
            "DATA_DIR": str(data_dir),
            "RESULT_DIR": str(result_dir),
            "CACHE_DIR": str(cache_dir),
            "MODEL_DIR": str(model_dir),
            "ARNOLD_HESS_WM_PATH": str(watermark_path),
            "ARNOLD_HESS_WM_SIZE": str(args.watermark_size),
            "ARNOLD_HESS_REPEAT_LIMIT": str(args.repeat_limit),
            "ARNOLD_HESS_ULTRA_FAST_CANDIDATE_SCORING": "1" if args.candidate_scoring == "ultrafast" else "0",
            "ARNOLD_HESS_FAST_CANDIDATE_SCORING": "0" if args.candidate_scoring == "full" else "1",
            "ARNOLD_HESS_DEBUG": os.environ.get("ARNOLD_HESS_DEBUG", "0"),
            "WAVES_LIMIT": str(args.limit),
            "WAVES_SUBSET_LIMIT": str(subset_limit),
        }
    )
    os.environ.update(env)
    write_env_file(PROJECT_ROOT / ".env", env)

    log("=" * 80)
    log("Arnold-Hess WAVES-style benchmark")
    log(f"dataset={args.dataset}, limit={args.limit}, subset_limit={subset_limit}")
    log(f"DATA_DIR={data_dir}")
    log(f"RESULT_DIR={result_dir}")
    log(f"watermark={watermark_path}")
    log(f"repeat_limit={args.repeat_limit}, candidate_scoring={args.candidate_scoring}")
    log("=" * 80)

    if args.data_source == "coco-val2017":
        src_dir = download_and_extract_coco(raw_dir, skip_download=args.skip_download)
        prepared_dir = prepare_512_hosts(src_dir, raw_dir / f"{args.dataset}512", args.limit, force=args.force_prepare)
    elif args.data_source == "synthetic":
        prepared_dir = create_synthetic_hosts(raw_dir, args.limit, seed=args.seed)
    else:
        if not args.input:
            raise RuntimeError("--input is required when --data-source local is used")
        prepared_dir = prepare_512_hosts(Path(args.input).resolve(), raw_dir / f"{args.dataset}512", args.limit, force=args.force_prepare)

    attacks = resolve_attacks(args.attack_preset, args.attacks)
    strengths = [strength_label(x) for x in parse_csv(args.strengths)]
    for st in strengths:
        if not 0 <= float(st) <= 1:
            raise ValueError("All strengths must be in [0,1]")

    if not args.skip_embed:
        run_cmd(
            [
                sys.executable,
                "scripts/embed_arnold_hess.py",
                "--input",
                str(prepared_dir),
                "--watermark",
                str(watermark_path),
                "--dataset",
                args.dataset,
                "--limit",
                str(args.limit),
            ],
            env=env,
            dry_run=args.dry_run,
        )

    # Clean metric placeholders are needed by WAVES aggregation when only
    # lightweight attack metrics are computed.
    if not args.dry_run:
        write_clean_metric_placeholder(result_dir, args.dataset, "real", args.limit)
        write_clean_metric_placeholder(result_dir, args.dataset, METHOD, args.limit)

    attack_sources = [METHOD] + (["real"] if args.include_real_attacks else [])
    if not args.skip_attacks:
        generate_attacks(data_dir, args.dataset, attacks, strengths, args.limit, args.attack_engine, args.seed, force=args.force_attacks, sources=attack_sources)

    if not args.skip_decode:
        if args.use_subprocess_tools:
            decode_all(env, data_dir, args.dataset, attacks, strengths, args.limit, subset_limit, dry_run=args.dry_run)
        elif not args.dry_run:
            decode_all_direct(data_dir, result_dir, args.dataset, attacks, strengths, args.limit, include_real_attacks=args.include_real_attacks)

    if not args.skip_metrics:
        if args.use_subprocess_tools:
            compute_metrics(env, data_dir, args.dataset, attacks, strengths, args.limit, subset_limit, dry_run=args.dry_run)
        elif not args.dry_run:
            compute_metrics_direct(data_dir, result_dir, args.dataset, attacks, strengths, args.limit, subset_limit, include_real_attacks=args.include_real_attacks)

    if not args.skip_summary and not args.dry_run:
        csv_path = export_summary(result_dir, args.dataset, attacks, strengths, subset_limit, watermark_path)
        log("=" * 80)
        log("Finished. Main comparison table:")
        log(str(csv_path))
        log("Use the columns tpr_at_0_1pct_fpr, psnr_mean, ssim_mean, ber_mean, bit_accuracy_mean, and nc_mean for paper comparison.")
        log("=" * 80)


if __name__ == "__main__":
    main()
