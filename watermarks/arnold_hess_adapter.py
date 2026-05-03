"""Small adapter exposing Arnold-Hess to official WAVES scripts."""
from __future__ import annotations

import numpy as np

from . import arnold_hess as core


def embed_image(host_path: str, watermark_path: str, watermarked_output_path: str, side_info_output_path: str) -> np.ndarray:
    return core.embed_full(
        host_path=host_path,
        watermark_path=watermark_path,
        output_path=watermarked_output_path,
        embed_info_path=side_info_output_path,
        arnold_iterations=core.ARNOLD_ITERATIONS,
    )


def decode_message(image_path: str, embed_info_path: str) -> np.ndarray:
    return core.extract_message_bits(image_path, embed_info_path).astype(bool)


def ground_truth_message(watermark_path: str) -> np.ndarray:
    return core.watermark_bits_from_file(watermark_path).astype(bool)
