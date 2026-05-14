"""
Phase 1: Complete Traditional Magnetospheric Analysis Pipeline
Metatron Dynamics, Inc. — ABR vs Traditional Pipeline Comparison

Single script running all four non-injective transformations:
  1.1 Grid interpolation (IDW, stations → 36×72 grid)
  1.2 Spherical harmonic decomposition (degree 1–18)
  1.3 Source separation (magnetospheric deg 1–3, ionospheric deg 4+)
  1.4 Standard diagnostics (SYM-H proxy, AE proxy, power series)

Operates on a 2-hour window matching the ABR pipeline for direct comparison.

Each transformation declares what it preserves and discards.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import lpmv
from math import lgamma
from pathlib import Path
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output/phase1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMPONENTS = ["N", "E", "Z"]
EARTH_RADIUS_KM = 6371.0
WINDOW_MINUTES = 120

# Grid: 5° × 5°
LAT_CENTERS = np.arange(-87.5, 90, 5.0)  # 36 values
LON_CENTERS = np.arange(2.5, 360, 5.0)   # 72 values
N_LAT = len(LAT_CENTERS)
N_LON = len(LON_CENTERS)

# IDW
K_NEIGHBORS = 4
IDW_POWER = 2

# SH
N_MAX = 18

# Source separation
MAG_DEG_MAX = 3

# Auroral zone
AURORAL_LAT_MIN = 65.0
AURORAL_LAT_MAX = 75.0


# ===================================================================
# DATA LOADING
# ===================================================================

def load_supermag(csv_path):
    log.info(f"Loading {csv_path}")
    df = pd.read_csv(csv_path)
    # Drop rows with HTML artifacts or non-parseable dates
    df = df[~df["Date_UTC"].astype(str).str.contains("<|>|br", na=False)]
    df["Date_UTC"] = pd.to_datetime(df["Date_UTC"], format="ISO8601")

    if "dbn_nez" in df.columns:
        df = df.rename(columns={"dbn_nez": "N", "dbe_nez": "E", "dbz_nez": "Z"})

    df = df.dropna(subset=["N", "E", "Z"])
    df = df.sort_values("Date_UTC").reset_index(drop=True)

    station_meta = df.groupby("IAGA").agg(
        lat=("GEOLAT", "first"),
        lon=("GEOLON", "first")
    )
    station_meta["lon"] = station_meta["lon"] % 360

    log.info(f"  {len(df)} measurements, {len(station_meta)} stations")
    return df, station_meta


def find_storm_peak(df):
    df["_mag2"] = df["N"]**2 + df["E"]**2 + df["Z"]**2
    peak_ts = df.groupby("Date_UTC")["_mag2"].sum().idxmax()
    df.drop(columns=["_mag2"], inplace=True)
    return pd.Timestamp(peak_ts)


def extract_window(df, peak_ts, window_minutes):
    half = pd.Timedelta(minutes=window_minutes // 2)
    mask = (df["Date_UTC"] >= peak_ts - half) & (df["Date_UTC"] < peak_ts + half)
    wdf = df[mask].copy()
    timestamps = sorted(wdf["Date_UTC"].unique())
    log.info(f"  Window: {timestamps[0]} — {timestamps[-1]} ({len(timestamps)} steps)")
    return wdf, timestamps


# ===================================================================
# PHASE 1.1: GRID INTERPOLATION (IDW)
# ===================================================================

def latlon_to_xyz(lat_deg, lon_deg):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    return np.column_stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat)
    ])


def haversine_km(lat1, lon1, lat2, lon2):
    la1, lo1 = np.radians(lat1), np.radians(lon1)
    la2, lo2 = np.radians(lat2), np.radians(lon2)
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = np.sin(dlat/2)**2 + np.cos(la1)*np.cos(la2)*np.sin(dlon/2)**2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def idw_one_step(station_lats, station_lons, station_vals,
                 grid_lats, grid_lons, k=K_NEIGHBORS, power=IDW_POWER):
    """IDW interpolation for one component at one timestep."""
    station_xyz = latlon_to_xyz(station_lats, station_lons)
    tree = cKDTree(station_xyz)

    glat, glon = np.meshgrid(grid_lats, grid_lons, indexing="ij")
    target_xyz = latlon_to_xyz(glat.ravel(), glon.ravel())

    dists, idxs = tree.query(target_xyz, k=k)
    dists = np.maximum(dists, 1e-10)

    weights = 1.0 / dists**power
    weights /= weights.sum(axis=1, keepdims=True)

    neighbor_vals = station_vals[idxs]
    gridded = np.sum(weights * neighbor_vals, axis=1)
    return gridded.reshape(len(grid_lats), len(grid_lons))


def run_grid_interpolation(wdf, timestamps, station_meta):
    """Phase 1.1: stations → regular grid via IDW."""
    log.info("Phase 1.1: Grid interpolation (IDW)")

    n_times = len(timestamps)
    gridded = np.full((n_times, 3, N_LAT, N_LON), np.nan)

    for t_idx, ts in enumerate(timestamps):
        snap = wdf[wdf["Date_UTC"] == ts]
        if len(snap) < K_NEIGHBORS:
            continue

        lats = station_meta.loc[
            station_meta.index.isin(snap["IAGA"]), "lat"
        ].reindex(snap["IAGA"].values).values
        lons = station_meta.loc[
            station_meta.index.isin(snap["IAGA"]), "lon"
        ].reindex(snap["IAGA"].values).values

        for c_idx, comp in enumerate(COMPONENTS):
            vals = snap[comp].values
            valid = np.isfinite(vals) & np.isfinite(lats) & np.isfinite(lons)
            if valid.sum() < K_NEIGHBORS:
                continue
            gridded[t_idx, c_idx] = idw_one_step(
                lats[valid], lons[valid], vals[valid],
                LAT_CENTERS, LON_CENTERS
            )

    log.info(f"  Grid shape: {gridded.shape}, NaN fraction: {np.isnan(gridded).mean():.4f}")
    return gridded


# ===================================================================
# PHASE 1.2: SPHERICAL HARMONIC DECOMPOSITION
# ===================================================================

def schmidt_factor(n, m):
    if m == 0:
        return 1.0
    log_f = 0.5 * (np.log(2.0) + lgamma(n - m + 1) - lgamma(n + m + 1))
    return np.exp(log_f)


def build_sh_design_matrix(lats_deg, lons_deg, n_max):
    colat_rad = np.radians(90.0 - lats_deg)
    lon_rad = np.radians(lons_deg)
    colat_g, lon_g = np.meshgrid(colat_rad, lon_rad, indexing="ij")
    cos_colat = np.cos(colat_g).ravel()
    lon_flat = lon_g.ravel()
    n_pts = len(cos_colat)

    coeff_index = []
    for n in range(1, n_max + 1):
        for m in range(0, n + 1):
            coeff_index.append((n, m, "g"))
            if m > 0:
                coeff_index.append((n, m, "h"))

    n_coeffs = len(coeff_index)
    Y = np.zeros((n_pts, n_coeffs))

    Pnm_cache = {}
    for n in range(1, n_max + 1):
        for m in range(0, n + 1):
            raw = lpmv(m, n, cos_colat)
            Pnm_cache[(n, m)] = schmidt_factor(n, m) * raw

    for col, (n, m, kind) in enumerate(coeff_index):
        Pnm = Pnm_cache[(n, m)]
        if kind == "g":
            Y[:, col] = Pnm * np.cos(m * lon_flat)
        else:
            Y[:, col] = Pnm * np.sin(m * lon_flat)

    log.info(f"  Design matrix: {n_pts} points × {n_coeffs} coefficients")
    return Y, coeff_index


def run_sh_decomposition(gridded, timestamps):
    """Phase 1.2: gridded field → SH coefficients."""
    log.info("Phase 1.2: Spherical harmonic decomposition")

    Y, coeff_index = build_sh_design_matrix(LAT_CENTERS, LON_CENTERS, N_MAX)
    n_times, n_comp = gridded.shape[0], gridded.shape[1]
    n_coeffs = len(coeff_index)

    all_coeffs = np.full((n_times, n_comp, n_coeffs), np.nan)
    all_residuals = np.full((n_times, n_comp), np.nan)

    for t_idx in range(n_times):
        for c_idx in range(n_comp):
            field = gridded[t_idx, c_idx].ravel()
            valid = ~np.isnan(field)
            if valid.sum() < n_coeffs:
                continue
            coeffs, _, _, _ = np.linalg.lstsq(Y[valid], field[valid], rcond=None)
            all_coeffs[t_idx, c_idx] = coeffs
            all_residuals[t_idx, c_idx] = np.linalg.norm(
                Y[valid] @ coeffs - field[valid]
            )

    log.info(f"  Mean residual: {np.nanmean(all_residuals):.2f}")
    return all_coeffs, all_residuals, Y, coeff_index


# ===================================================================
# PHASE 1.3: SOURCE SEPARATION
# ===================================================================

def band_power(coeffs, coeff_index, deg_min, deg_max):
    total = 0.0
    for idx, (n, m, kind) in enumerate(coeff_index):
        if deg_min <= n <= deg_max:
            total += (n + 1) * coeffs[idx]**2
    return total


def reconstruct_band(Y, coeffs, coeff_index, deg_min, deg_max):
    masked = np.zeros_like(coeffs)
    for idx, (n, m, kind) in enumerate(coeff_index):
        if deg_min <= n <= deg_max:
            masked[idx] = coeffs[idx]
    return Y @ masked


def run_source_separation(all_coeffs, Y, coeff_index):
    """Phase 1.3: separate magnetospheric (deg 1-3) from ionospheric (deg 4+)."""
    log.info("Phase 1.3: Source separation")

    n_times, n_comp, n_coeffs = all_coeffs.shape

    mag_power = np.zeros((n_times, n_comp))
    iono_power = np.zeros((n_times, n_comp))
    total_power = np.zeros((n_times, n_comp))

    for t_idx in range(n_times):
        for c_idx in range(n_comp):
            c = all_coeffs[t_idx, c_idx]
            if np.any(np.isnan(c)):
                continue
            mag_power[t_idx, c_idx] = band_power(c, coeff_index, 1, MAG_DEG_MAX)
            iono_power[t_idx, c_idx] = band_power(c, coeff_index, MAG_DEG_MAX + 1, N_MAX)
            total_power[t_idx, c_idx] = band_power(c, coeff_index, 1, N_MAX)

    log.info(f"  Mag band fraction: "
             f"{np.nanmean(mag_power.sum(1) / (total_power.sum(1) + 1e-30)):.3f}")

    return mag_power, iono_power, total_power


# ===================================================================
# PHASE 1.4: STANDARD DIAGNOSTICS
# ===================================================================

def compute_symh_proxy(all_coeffs, coeff_index):
    g10_idx = None
    for idx, (n, m, kind) in enumerate(coeff_index):
        if n == 1 and m == 0 and kind == "g":
            g10_idx = idx
            break
    symh = all_coeffs[:, 0, g10_idx].copy()
    baseline_n = min(10, len(symh) // 4)
    if baseline_n > 0:
        symh -= np.nanmean(symh[:baseline_n])
    return symh


def compute_ae_proxy(gridded):
    auroral_mask_n = (LAT_CENTERS >= AURORAL_LAT_MIN) & (LAT_CENTERS <= AURORAL_LAT_MAX)
    auroral_mask_s = (LAT_CENTERS >= -AURORAL_LAT_MAX) & (LAT_CENTERS <= -AURORAL_LAT_MIN)
    auroral_mask = auroral_mask_n | auroral_mask_s

    n_times = gridded.shape[0]
    auroral_N = gridded[:, 0, auroral_mask, :]
    auroral_flat = auroral_N.reshape(n_times, -1)

    baseline_n = min(10, n_times // 4)
    if baseline_n > 0:
        baseline = np.nanmean(auroral_flat[:baseline_n], axis=0, keepdims=True)
        auroral_flat = auroral_flat - baseline

    au = np.nanmax(auroral_flat, axis=1)
    al = np.nanmin(auroral_flat, axis=1)
    ae = au - al
    return ae, au, al


def reconstruct_at_stations(Y_stations, coeffs):
    """Reconstruct field at station locations from SH coefficients."""
    valid = ~np.isnan(coeffs)
    if not np.all(valid):
        return np.full(Y_stations.shape[0], np.nan)
    return Y_stations @ coeffs


# ===================================================================
# MAIN
# ===================================================================

def run_traditional_pipeline(csv_path, window_minutes=WINDOW_MINUTES):
    # Load
    df, station_meta = load_supermag(csv_path)

    # Find peak and extract window
    peak_ts = find_storm_peak(df)
    log.info(f"Storm peak: {peak_ts}")
    wdf, timestamps = extract_window(df, peak_ts, window_minutes)
    n_times = len(timestamps)

    # 1.1 Grid interpolation
    gridded = run_grid_interpolation(wdf, timestamps, station_meta)

    # 1.2 SH decomposition
    all_coeffs, residuals, Y, coeff_index = run_sh_decomposition(gridded, timestamps)

    # 1.3 Source separation
    mag_power, iono_power, total_power = run_source_separation(
        all_coeffs, Y, coeff_index
    )

    # 1.4 Diagnostics
    log.info("Phase 1.4: Standard diagnostics")
    symh = compute_symh_proxy(all_coeffs, coeff_index)
    ae, au, al = compute_ae_proxy(gridded)

    power_total = total_power.sum(axis=1)
    power_mag = mag_power.sum(axis=1)
    power_iono = iono_power.sum(axis=1)

    log.info(f"  SYM-H range: [{np.nanmin(symh):.1f}, {np.nanmax(symh):.1f}]")
    log.info(f"  AE range: [{np.nanmin(ae):.1f}, {np.nanmax(ae):.1f}]")

    # --- Reconstruct at station locations (for Phase 3 residual comparison) ---
    log.info("Reconstructing SH field at station locations...")
    station_lats = station_meta["lat"].values
    station_lons = station_meta["lon"].values

    # Build design matrix for station points (no meshgrid — just the points)
    n_st = len(station_lats)
    colat_st = np.radians(90.0 - station_lats)
    lon_st = np.radians(station_lons)
    cos_colat_st = np.cos(colat_st)
    n_coeffs = len(coeff_index)

    Pnm_st = {}
    for n in range(1, N_MAX + 1):
        for m in range(0, n + 1):
            raw = lpmv(m, n, cos_colat_st)
            Pnm_st[(n, m)] = schmidt_factor(n, m) * raw

    Y_stations = np.zeros((n_st, n_coeffs))
    for col, (n, m, kind) in enumerate(coeff_index):
        Pnm = Pnm_st[(n, m)]
        if kind == "g":
            Y_stations[:, col] = Pnm * np.cos(m * lon_st)
        else:
            Y_stations[:, col] = Pnm * np.sin(m * lon_st)

    log.info(f"  Station design matrix: {n_st} stations × {n_coeffs} coefficients")

    # For each timestep, reconstruct at stations and compare to measured
    station_codes = list(station_meta.index)
    n_stations = len(station_codes)

    reconstructed = np.full((n_times, 3, n_stations), np.nan)
    measured = np.full((n_times, 3, n_stations), np.nan)

    for t_idx, ts in enumerate(timestamps):
        snap = wdf[wdf["Date_UTC"] == ts]
        for s_idx, code in enumerate(station_codes):
            row = snap[snap["IAGA"] == code]
            if len(row) == 1:
                measured[t_idx, :, s_idx] = row[["N", "E", "Z"]].values[0]

        for c_idx in range(3):
            c = all_coeffs[t_idx, c_idx]
            if not np.any(np.isnan(c)):
                reconstructed[t_idx, c_idx] = Y_stations @ c

    residual = measured - reconstructed
    residual_rms = np.sqrt(np.nanmean(residual**2))
    log.info(f"  Station reconstruction residual RMS: {residual_rms:.2f} nT")

    # --- Save ---
    ts_strings = np.array([str(t) for t in timestamps])

    np.savez_compressed(
        OUTPUT_DIR / "traditional_results.npz",
        gridded=gridded,
        coefficients=all_coeffs,
        sh_residuals=residuals,
        symh=symh,
        ae=ae, au=au, al=al,
        power_total=power_total,
        power_mag=power_mag,
        power_iono=power_iono,
        mag_power_per_comp=mag_power,
        iono_power_per_comp=iono_power,
        total_power_per_comp=total_power,
        timestamps=ts_strings,
        station_reconstructed=reconstructed,
        station_measured=measured,
        station_residual=residual,
        station_codes=np.array(station_codes),
        station_lats=station_lats,
        station_lons=station_lons,
        lat_centers=LAT_CENTERS,
        lon_centers=LON_CENTERS,
        n_max=N_MAX,
        mag_deg_max=MAG_DEG_MAX,
    )
    log.info(f"Saved: {OUTPUT_DIR / 'traditional_results.npz'}")

    summary = {
        "pipeline": "Traditional: IDW → SH → source separation → indices",
        "n_times": n_times,
        "window_start": str(timestamps[0]),
        "window_end": str(timestamps[-1]),
        "grid": f"{N_LAT}×{N_LON} (5°×5°)",
        "idw_k": K_NEIGHBORS,
        "sh_n_max": N_MAX,
        "mag_deg_max": MAG_DEG_MAX,
        "symh_range": [float(np.nanmin(symh)), float(np.nanmax(symh))],
        "ae_range": [float(np.nanmin(ae)), float(np.nanmax(ae))],
        "station_residual_rms_nT": float(residual_rms),
        "non_injective_transformations": [
            "1.1 IDW grid interpolation: imposes spatial coupling not in measurement",
            "1.2 SH decomposition: global aggregation destroys local structure",
            "1.3 Source separation: assumes spectral separability during storms",
            "1.4 Index construction: collapses spatial field to single scalar",
        ],
    }

    with open(OUTPUT_DIR / "phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary")

    return summary


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR / "supermag.csv"
    summary = run_traditional_pipeline(csv_path)

    log.info("\nPhase 1 complete — traditional pipeline.")
    log.info(f"  Window: {summary['window_start']} — {summary['window_end']}")
    log.info(f"  SYM-H: {summary['symh_range']}")
    log.info(f"  AE: {summary['ae_range']}")
    log.info(f"  Station residual RMS: {summary['station_residual_rms_nT']:.2f} nT")
    log.info(f"  Four non-injective transformations applied and declared.")
