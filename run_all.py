#!/usr/bin/env python3
"""
run_all.py — ABR vs Traditional Pipeline, Full Execution
Metatron Dynamics, Inc.

Calls the working pipeline scripts in declared order.
Each phase gates the next: if a phase fails, execution stops.

Usage:
    python run_all.py [path/to/supermag.csv]
"""

import subprocess
import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()
SRC_DIR = REPO_ROOT / "src"

# Default data path — override with first CLI argument
DEFAULT_CSV = REPO_ROOT / "data" / "supermag.csv"


def run_phase(label: str, script: str, args: list = None):
    """Run a pipeline phase. Exit on failure."""
    script_path = SRC_DIR / script if not Path(script).is_absolute() else Path(script)
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Script: {script}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))

    if result.returncode != 0:
        print(f"\n*** FAILED: {label} (exit code {result.returncode}) ***")
        print(f"*** Execution stopped. Fix {script} before proceeding. ***")
        sys.exit(result.returncode)

    print(f"\n  ✓ {label} complete.\n")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_CSV)

    if not Path(csv_path).exists():
        print(f"Data file not found: {csv_path}")
        print("Provide path as: python run_all.py path/to/supermag.csv")
        sys.exit(1)

    os.makedirs(REPO_ROOT / "output", exist_ok=True)
    os.makedirs(REPO_ROOT / "figures", exist_ok=True)

    # --- Phase 1: Traditional Pipeline ---
    run_phase(
        "Phase 1 — Traditional Pipeline (IDW → SH → Indices)",
        "phase1_traditional.py",
        [csv_path],
    )

    # --- Phase 2.1: ABR Pipeline Verification ---
    run_phase(
        "Phase 2.1 — ABR Pipeline Verification",
        "phase2_1_abr_pipeline.py",
        [csv_path],
    )

    # --- Phase 3: Comparison ---
    run_phase(
        "Phase 3 — Comparison (Temporal, Spatial, Information Loss)",
        "phase3_comparison.py",
    )

    # --- Phase 3: Sensitivity Analysis ---
    run_phase(
        "Phase 3 — Sensitivity Analysis (Edge Threshold × SH Degree)",
        "phase3_sensitivity.py",
        [csv_path],
    )

    # --- Phase 4.1: Null Test ---
    null_test = SRC_DIR / "phase4_1_null_test.py"
    if null_test.exists():
        run_phase(
            "Phase 4.1 — Null Test (Synthetic White Noise → Γ ≈ 0)",
            "phase4_1_null_test.py",
            [csv_path],
        )
    else:
        print("\n  ⊘ Phase 4.1 (null test) not yet implemented — skipped.\n")

    print("\n" + "=" * 60)
    print("  All implemented phases complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
