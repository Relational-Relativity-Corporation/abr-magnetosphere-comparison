# ABR vs Traditional Methods — Magnetospheric Storm Analysis

**Metatron Dynamics, Inc.**

Rigorous, reproducible comparison demonstrating that the ABR kernel extracts
relational structure from magnetometer data that traditional analysis pipelines
destroy before processing begins.

## Data

SuperMAG St. Patrick's Day Storm, 2015-03-17.
Download requires a free SuperMAG account: https://supermag.jhuapl.edu/

Place the CSV in `data/supermag_20150317.csv`.

## Build Sequence

| Phase | Script | Description |
|-------|--------|-------------|
| 1.1 | `src/traditional/grid_interpolation.py` | IDW interpolation to 5 x 5 grid |
| 1.2 | `src/traditional/sh_decomposition.py` | Spherical harmonic decomposition |
| 1.3 | `src/traditional/source_separation.py` | Magnetospheric / ionospheric split |
| 1.4 | `src/traditional/diagnostics.py` | SYM-H proxy, AE proxy, power series |
| 2.1 | `src/abr/verify_pipeline.py` | Reproduce published gamma result |
| 2.2 | `src/abr/component_gamma.py` | Component-resolved gamma |
| 2.3 | `src/abr/spatial_localization.py` | Per-station gamma geographic map |
| 2.4 | `src/abr/latitudinal_profile.py` | Gamma binned by magnetic latitude |
| 3.x | `src/comparison/` | Side-by-side comparisons 1-4 |
| 4.x | `src/validation/` | Null test, injection test, multi-storm |

## Run All

`python run_all.py`

## Hardware

ROBIN-1 (NVIDIA GTX 1050 Ti, CUDA 12.6)

---
*All definitions bounded over D. No claim beyond D.*