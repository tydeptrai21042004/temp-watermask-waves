"""Metrics package.

Patched to avoid importing heavy optional dependencies at package import time.
Import submodules directly, e.g. `from metrics.image import compute_psnr`.
"""
