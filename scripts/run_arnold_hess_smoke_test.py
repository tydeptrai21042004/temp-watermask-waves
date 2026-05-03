"""Run the Arnold-Hess integration smoke test without pytest."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_arnold_hess_smoke.py"]
    print("Running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd, cwd=REPO, env=env))


if __name__ == "__main__":
    main()
