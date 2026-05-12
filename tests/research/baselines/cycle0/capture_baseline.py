#!/usr/bin/env python3
"""Capture the Cycle 0 baseline.

Run from the repo root::

    MPLBACKEND=Agg PYTHONHASHSEED=0 \\
        python tests/research/baselines/cycle0/capture_baseline.py
"""

from __future__ import annotations

import datetime
import shutil
import sys
from pathlib import Path

# Ensure project root is on sys.path
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

_BASELINE_DIR = Path(__file__).parent
_OUTPUT_DIR = _BASELINE_DIR / "_pipeline_output"
_OUTPUT_DIR.mkdir(exist_ok=True)

_START = datetime.date(2022, 1, 1)
_END = datetime.date(2023, 12, 31)
_SEED = 42


def main() -> None:
    from research.cli._main import main as cli_main

    exit_code = 0
    try:
        cli_main(
            seed=_SEED,
            start_date=_START,
            end_date=_END,
            output_dir=_OUTPUT_DIR,
        )
    except SystemExit as exc:
        exit_code = int(exc.code) if exc.code is not None else 0

    # Copy artifacts from _pipeline_output into the baseline dir
    for src in _OUTPUT_DIR.iterdir():
        if src.suffix in {".png", ".svg", ".json", ".csv", ".md", ".txt"}:
            dst = _BASELINE_DIR / src.name
            shutil.copy2(src, dst)
            print(f"  captured: {dst.name}")

    (_BASELINE_DIR / ".exit_code").write_text(str(exit_code) + "\n")
    print(f"\nPipeline exited with code {exit_code}")
    print(f"Baseline directory: {_BASELINE_DIR}")


if __name__ == "__main__":
    main()
