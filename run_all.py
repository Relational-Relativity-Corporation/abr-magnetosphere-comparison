#!/usr/bin/env python
"""
Master run script — executes the full pipeline in build-plan order.
Metatron Dynamics, Inc.
"""
import subprocess, sys

PHASES = [
    ("Phase 1.1 — Grid Interpolation",    "src/traditional/grid_interpolation.py"),
    ("Phase 1.2 — SH Decomposition",      "src/traditional/sh_decomposition.py"),
    ("Phase 1.3 — Source Separation",      "src/traditional/source_separation.py"),
    ("Phase 1.4 — Standard Diagnostics",   "src/traditional/diagnostics.py"),
    ("Phase 2.1 — Verify ABR Pipeline",    "src/abr/verify_pipeline.py"),
    ("Phase 2.2 — Component Gamma",        "src/abr/component_gamma.py"),
    ("Phase 2.3 — Spatial Localization",   "src/abr/spatial_localization.py"),
    ("Phase 2.4 — Latitudinal Profile",    "src/abr/latitudinal_profile.py"),
    ("Phase 3.1 — Temporal Resolution",    "src/comparison/temporal_resolution.py"),
    ("Phase 3.2 — Spatial Resolution",     "src/comparison/spatial_resolution.py"),
    ("Phase 3.3 — Cross-Scale Coupling",   "src/comparison/cross_scale_coupling.py"),
    ("Phase 3.4 — Information Loss",       "src/comparison/information_loss.py"),
    ("Phase 4.1 — Null Test",             "src/validation/null_test.py"),
    ("Phase 4.2 — Injection Test",        "src/validation/injection_test.py"),
    ("Phase 4.3 — Multi-Storm",           "src/validation/multi_storm.py"),
]

def main():
    for label, script in PHASES:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        result = subprocess.run([sys.executable, script], capture_output=False)
        if result.returncode != 0:
            print(f"  FAILED: {script} (exit {result.returncode})")
            sys.exit(1)
    print("\nAll phases complete.")

if __name__ == "__main__":
    main()