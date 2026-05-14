"""
Phase 3.5: Topology and Parameter Sensitivity Analysis
Metatron Dynamics, Inc.

Systematically varies the declared parameters to characterize
how comparison metrics depend on:
  - MAX_EDGE_KM (spatial topology threshold)
  - N_MAX (SH truncation degree)
  - IDW k (interpolation neighbors)
  - Window size

Each sweep holds all other parameters at default and varies one.
Results reported as a table showing metric stability or drift.

DECLARATION: The comparison results (39.5% non-preservation,
99.2% component coupling, 1-step Γ lead) are conditioned on
the declared parameter set. This analysis characterizes that
conditioning.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/phase3_sensitivity")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ===================================================================
# RESIDUAL ANALYSIS — standalone function
#
# Computes the fraction of edge-field variance not preserved by
# the traditional pipeline, for a given edge threshold.
# Operates on pre-computed traditional pipeline results.
# ===================================================================

def compute_non_preservation(trad_path, max_edge_km):
    """
    For a given proximity threshold, compute the fraction of
    measured edge-field variance not preserved by the traditional pipeline.
    """
    trad = np.load(trad_path, allow_pickle=True)
    residual = trad["station_residual"]     # (n_times, 3, n_stations)
    measured = trad["station_measured"]
    station_lats = trad["station_lats"]
    station_lons = trad["station_lons"]
    n_times = residual.shape[0]
    n_stations = residual.shape[2]

    EARTH_R = 6371.0

    def haversine(lat1, lon1, lat2, lon2):
        la1, lo1 = np.radians(lat1), np.radians(lon1)
        la2, lo2 = np.radians(lat2), np.radians(lon2)
        dlat = la2 - la1
        dlon = lo2 - lo1
        a = np.sin(dlat/2)**2 + np.cos(la1)*np.cos(la2)*np.sin(dlon/2)**2
        return EARTH_R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    edges = []
    for i in range(n_stations):
        for j in range(i + 1, n_stations):
            d = haversine(station_lats[i], station_lons[i],
                          station_lats[j], station_lons[j])
            if d <= max_edge_km:
                edges.append((i, j))

    if not edges:
        return {"n_edges": 0, "mean_fraction": float("nan"),
                "peak_fraction": float("nan")}

    res_var = np.zeros(n_times)
    meas_var = np.zeros(n_times)

    for t_idx in range(n_times):
        res_vals = []
        meas_vals = []
        for (i, j) in edges:
            for c in range(3):
                ri, rj = residual[t_idx, c, i], residual[t_idx, c, j]
                mi, mj = measured[t_idx, c, i], measured[t_idx, c, j]
                if np.isfinite(ri) and np.isfinite(rj):
                    res_vals.append(ri - rj)
                if np.isfinite(mi) and np.isfinite(mj):
                    meas_vals.append(mi - mj)
        if res_vals:
            res_var[t_idx] = np.var(res_vals)
        if meas_vals:
            meas_var[t_idx] = np.var(meas_vals)

    fraction = res_var / (meas_var + 1e-30)

    return {
        "n_edges": len(edges),
        "mean_fraction": float(np.mean(fraction)),
        "peak_fraction": float(np.max(fraction)),
        "std_fraction": float(np.std(fraction)),
    }


# ===================================================================
# SWEEP 1: MAX_EDGE_KM
# ===================================================================

def sweep_edge_threshold():
    log.info("Sweep 1: Edge threshold (MAX_EDGE_KM)")

    trad_path = "output/phase1/traditional_results.npz"
    thresholds = [300, 500, 750, 1000, 1500, 2000, 3000]

    results = []
    for thresh in thresholds:
        log.info(f"  Threshold: {thresh} km")
        r = compute_non_preservation(trad_path, thresh)
        r["threshold_km"] = thresh
        results.append(r)
        log.info(f"    Edges: {r['n_edges']}, "
                 f"mean non-preserved: {r['mean_fraction']*100:.1f}%, "
                 f"peak: {r['peak_fraction']*100:.1f}%")

    return results


# ===================================================================
# SWEEP 2: SH TRUNCATION (N_MAX)
#
# Re-runs the traditional pipeline at different SH degrees.
# This requires re-fitting, but uses the same gridded field.
# ===================================================================

def sweep_sh_truncation():
    log.info("Sweep 2: SH truncation degree")

    from scipy.special import lpmv
    from math import lgamma

    # Load gridded field from Phase 1
    trad = np.load("output/phase1/traditional_results.npz", allow_pickle=True)
    station_lats = trad["station_lats"]
    station_lons = trad["station_lons"]
    measured = trad["station_measured"]  # (n_times, 3, n_stations)
    n_times = measured.shape[0]
    n_stations = measured.shape[2]

    # Load the gridded field — need to re-run SH at different degrees
    # We'll load it from the Phase 1 output
    gridded = trad["gridded"]  # (n_times, 3, n_lat, n_lon)
    lat_centers = trad["lat_centers"]
    lon_centers = trad["lon_centers"]

    def schmidt_factor(n, m):
        if m == 0:
            return 1.0
        log_f = 0.5 * (np.log(2.0) + lgamma(n - m + 1) - lgamma(n + m + 1))
        return np.exp(log_f)

    def build_design(lats, lons, n_max, is_grid=True):
        if is_grid:
            colat_rad = np.radians(90.0 - lats)
            lon_rad = np.radians(lons)
            cg, lg = np.meshgrid(colat_rad, lon_rad, indexing="ij")
            cos_c = np.cos(cg).ravel()
            lon_f = lg.ravel()
        else:
            cos_c = np.cos(np.radians(90.0 - lats))
            lon_f = np.radians(lons)

        ci = []
        for n in range(1, n_max + 1):
            for m in range(0, n + 1):
                ci.append((n, m, "g"))
                if m > 0:
                    ci.append((n, m, "h"))

        Y = np.zeros((len(cos_c), len(ci)))
        for col, (n, m, kind) in enumerate(ci):
            raw = lpmv(m, n, cos_c)
            Pnm = schmidt_factor(n, m) * raw
            if kind == "g":
                Y[:, col] = Pnm * np.cos(m * lon_f)
            else:
                Y[:, col] = Pnm * np.sin(m * lon_f)
        return Y, ci

    n_max_values = [6, 10, 14, 18, 24]
    results = []

    for n_max in n_max_values:
        log.info(f"  N_MAX = {n_max}")

        Y_grid, ci = build_design(lat_centers, lon_centers, n_max, is_grid=True)
        Y_sta, _ = build_design(station_lats, station_lons, n_max, is_grid=False)
        n_coeffs = len(ci)

        reconstructed = np.full((n_times, 3, n_stations), np.nan)

        for t_idx in range(n_times):
            for c_idx in range(3):
                field = gridded[t_idx, c_idx].ravel()
                valid = ~np.isnan(field)
                if valid.sum() < n_coeffs:
                    continue
                coeffs, _, _, _ = np.linalg.lstsq(Y_grid[valid], field[valid],
                                                    rcond=None)
                reconstructed[t_idx, c_idx] = Y_sta @ coeffs

        residual = measured - reconstructed
        res_rms = float(np.sqrt(np.nanmean(residual**2)))

        # Edge-field non-preservation at default threshold
        r = {"n_max": n_max, "n_coeffs": n_coeffs, "residual_rms_nT": res_rms}

        # Compute edge-field fraction for 1000 km threshold
        EARTH_R = 6371.0
        def haversine(lat1, lon1, lat2, lon2):
            la1, lo1 = np.radians(lat1), np.radians(lon1)
            la2, lo2 = np.radians(lat2), np.radians(lon2)
            dlat = la2 - la1
            dlon = lo2 - lo1
            a = np.sin(dlat/2)**2 + np.cos(la1)*np.cos(la2)*np.sin(dlon/2)**2
            return EARTH_R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        edges = []
        for i in range(n_stations):
            for j in range(i + 1, n_stations):
                if haversine(station_lats[i], station_lons[i],
                             station_lats[j], station_lons[j]) <= 1000:
                    edges.append((i, j))

        fracs = []
        for t_idx in range(n_times):
            rv, mv = [], []
            for (i, j) in edges:
                for c in range(3):
                    ri, rj = residual[t_idx, c, i], residual[t_idx, c, j]
                    mi, mj = measured[t_idx, c, i], measured[t_idx, c, j]
                    if np.isfinite(ri) and np.isfinite(rj):
                        rv.append(ri - rj)
                    if np.isfinite(mi) and np.isfinite(mj):
                        mv.append(mi - mj)
            if rv and mv:
                fracs.append(np.var(rv) / (np.var(mv) + 1e-30))

        r["mean_non_preserved"] = float(np.mean(fracs)) if fracs else float("nan")
        results.append(r)
        log.info(f"    RMS: {res_rms:.1f} nT, "
                 f"non-preserved: {r['mean_non_preserved']*100:.1f}%")

    return results


# ===================================================================
# MAIN
# ===================================================================

def run_sensitivity():
    all_results = {}

    all_results["edge_threshold"] = sweep_edge_threshold()
    all_results["sh_truncation"] = sweep_sh_truncation()

    # Summary table
    log.info("\n" + "=" * 70)
    log.info("SENSITIVITY SUMMARY")
    log.info("=" * 70)

    log.info("\nEdge threshold sweep (SH N_MAX=18, IDW k=4):")
    log.info(f"  {'Threshold':>10}  {'Edges':>6}  {'Mean':>8}  {'Peak':>8}")
    for r in all_results["edge_threshold"]:
        log.info(f"  {r['threshold_km']:>8} km  {r['n_edges']:>6}  "
                 f"{r['mean_fraction']*100:>7.1f}%  "
                 f"{r['peak_fraction']*100:>7.1f}%")

    log.info("\nSH truncation sweep (edge=1000km, IDW k=4):")
    log.info(f"  {'N_MAX':>6}  {'Coeffs':>7}  {'RMS(nT)':>8}  {'Non-pres':>8}")
    for r in all_results["sh_truncation"]:
        log.info(f"  {r['n_max']:>6}  {r['n_coeffs']:>7}  "
                 f"{r['residual_rms_nT']:>7.1f}  "
                 f"{r['mean_non_preserved']*100:>7.1f}%")

    log.info("=" * 70)

    with open(OUTPUT_DIR / "sensitivity_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nSaved: {OUTPUT_DIR / 'sensitivity_results.json'}")

    return all_results


if __name__ == "__main__":
    run_sensitivity()
