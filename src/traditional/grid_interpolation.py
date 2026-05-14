"""
Phase 1.1: Grid Interpolation — IDW on SuperMAG station data
Metatron Dynamics, Inc. — ABR vs Traditional Pipeline Comparison

Transforms 226 irregular station measurements onto a regular 5°×5° lat/lon grid
using inverse distance weighting (k=4 nearest stations).

DECLARATION: This is M_traditional — the first non-injective transformation.
Preserved: broad spatial pattern.
Discarded: station-level detail, local gradients, all structure below 5° resolution.
Pairwise differences between stations are altered by the smoothing kernel.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output/phase1_1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Grid specification: 5° × 5°
LAT_EDGES = np.arange(-90, 91, 5)       # 37 edges → 36 centers
LON_EDGES = np.arange(0, 361, 5)        # 73 edges → 72 centers
LAT_CENTERS = 0.5 * (LAT_EDGES[:-1] + LAT_EDGES[1:])  # 36 values
LON_CENTERS = 0.5 * (LON_EDGES[:-1] + LON_EDGES[1:])  # 72 values

# IDW parameters
K_NEIGHBORS = 4
IDW_POWER = 2           # standard inverse-square weighting
COMPONENTS = ["N", "E", "Z"]

# ---------------------------------------------------------------------------
# Utility: Haversine distance on a sphere (degrees in, km out)
# ---------------------------------------------------------------------------
def haversine_deg(lat1, lon1, lat2, lon2, R=6371.0):
    """Great-circle distance between two points given in degrees."""
    la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))

# ---------------------------------------------------------------------------
# Convert lat/lon to 3D Cartesian for cKDTree (Euclidean proxy for neighbor lookup)
# ---------------------------------------------------------------------------
def latlon_to_xyz(lat_deg, lon_deg, R=1.0):
    """Convert geographic coordinates to unit-sphere Cartesian."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return np.column_stack([x, y, z])

# ---------------------------------------------------------------------------
# IDW Interpolation — single timestep
# ---------------------------------------------------------------------------
def idw_interpolate(station_lats, station_lons, station_values,
                    grid_lats, grid_lons, k=K_NEIGHBORS, power=IDW_POWER):
    """
    Inverse distance weighting interpolation from irregular stations to a regular grid.

    Parameters
    ----------
    station_lats, station_lons : 1D arrays, shape (n_stations,)
    station_values : 1D array, shape (n_stations,) — one component at one timestep
    grid_lats, grid_lons : 1D arrays defining the regular grid axes
    k : number of nearest neighbors
    power : distance weighting exponent

    Returns
    -------
    gridded : 2D array, shape (len(grid_lats), len(grid_lons))
    """
    # Build KD-tree on station positions (unit sphere)
    station_xyz = latlon_to_xyz(station_lats, station_lons)
    tree = cKDTree(station_xyz)

    # Meshgrid of target points
    glat, glon = np.meshgrid(grid_lats, grid_lons, indexing="ij")
    target_xyz = latlon_to_xyz(glat.ravel(), glon.ravel())

    # Query k nearest stations for each grid point
    dists_eucl, idxs = tree.query(target_xyz, k=k)

    # Convert Euclidean chord distances to great-circle for proper weighting
    # (chord ≈ great-circle for nearby points, but we do it right)
    target_lats = glat.ravel()
    target_lons = glon.ravel()
    dists_gc = np.zeros_like(dists_eucl)
    for j in range(k):
        dists_gc[:, j] = haversine_deg(
            target_lats, target_lons,
            station_lats[idxs[:, j]], station_lons[idxs[:, j]]
        )

    # Handle coincident points (distance ~ 0)
    coincident = dists_gc < 1e-6  # less than 1 meter
    dists_gc[coincident] = 1e-6

    # Weights: inverse distance raised to power
    weights = 1.0 / dists_gc ** power
    weights /= weights.sum(axis=1, keepdims=True)

    # Weighted average
    neighbor_vals = station_values[idxs]  # shape (n_targets, k)
    gridded = np.sum(weights * neighbor_vals, axis=1)

    return gridded.reshape(len(grid_lats), len(grid_lons))

# ---------------------------------------------------------------------------
# Load SuperMAG CSV
# ---------------------------------------------------------------------------
def load_supermag(filepath):
    """
    Load SuperMAG CSV. Expected columns include:
    Date_UTC, IAGA, GEOLON, GEOLAT, N, E, Z (at minimum).
    Returns DataFrame sorted by time.
    """
    log.info(f"Loading SuperMAG data from {filepath}")
    df = pd.read_csv(filepath, parse_dates=["Date_UTC"])
    required = {"Date_UTC", "IAGA", "GEOLON", "GEOLAT", "N", "E", "Z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Drop rows with NaN in any component
    n_before = len(df)
    df = df.dropna(subset=["N", "E", "Z"])
    n_after = len(df)
    if n_before != n_after:
        log.info(f"Dropped {n_before - n_after} rows with NaN components")

    df = df.sort_values("Date_UTC").reset_index(drop=True)
    log.info(f"Loaded {len(df)} measurements from {df['IAGA'].nunique()} stations")
    log.info(f"Time range: {df['Date_UTC'].min()} to {df['Date_UTC'].max()}")
    return df

# ---------------------------------------------------------------------------
# Main: Interpolate all timesteps
# ---------------------------------------------------------------------------
def run_interpolation(csv_path):
    df = load_supermag(csv_path)

    timestamps = df["Date_UTC"].unique()
    n_times = len(timestamps)
    log.info(f"Processing {n_times} timesteps across {COMPONENTS} components")

    # Station metadata (positions assumed constant across the event)
    station_meta = df.groupby("IAGA").agg(
        lat=("GEOLAT", "first"),
        lon=("GEOLON", "first")
    )
    # Ensure longitudes in [0, 360)
    station_meta["lon"] = station_meta["lon"] % 360
    log.info(f"Station count: {len(station_meta)}")

    # Pre-allocate output: (n_times, 3, n_lat, n_lon)
    n_lat, n_lon = len(LAT_CENTERS), len(LON_CENTERS)
    gridded = np.full((n_times, len(COMPONENTS), n_lat, n_lon), np.nan, dtype=np.float64)

    # Information loss tracking
    station_counts = []
    all_neighbor_dists_km = []   # every IDW neighbor distance across all grid points & timesteps

    for t_idx, ts in enumerate(timestamps):
        if t_idx % 100 == 0:
            log.info(f"  Timestep {t_idx}/{n_times}: {ts}")

        snapshot = df[df["Date_UTC"] == ts].copy()
        # Merge station positions
        snapshot = snapshot.set_index("IAGA")
        lats = station_meta.loc[snapshot.index, "lat"].values
        lons = station_meta.loc[snapshot.index, "lon"].values
        station_counts.append(len(snapshot))

        # --- Compute neighbor distances for this timestep (once, shared across components) ---
        station_xyz = latlon_to_xyz(lats, lons)
        tree = cKDTree(station_xyz)
        glat_mesh, glon_mesh = np.meshgrid(LAT_CENTERS, LON_CENTERS, indexing="ij")
        target_xyz = latlon_to_xyz(glat_mesh.ravel(), glon_mesh.ravel())
        _, idxs_t = tree.query(target_xyz, k=K_NEIGHBORS)

        # Great-circle distances for all k neighbors at every grid point
        t_lats = glat_mesh.ravel()
        t_lons = glon_mesh.ravel()
        for j in range(K_NEIGHBORS):
            dkm = haversine_deg(t_lats, t_lons, lats[idxs_t[:, j]], lons[idxs_t[:, j]])
            all_neighbor_dists_km.extend(dkm.tolist())

        for c_idx, comp in enumerate(COMPONENTS):
            vals = snapshot[comp].values
            # Mask stations missing this component
            valid = ~np.isnan(vals)
            if valid.sum() < K_NEIGHBORS:
                log.warning(f"  Timestep {ts}, component {comp}: only {valid.sum()} valid stations, skipping")
                continue

            gridded[t_idx, c_idx] = idw_interpolate(
                lats[valid], lons[valid], vals[valid],
                LAT_CENTERS, LON_CENTERS,
                k=K_NEIGHBORS, power=IDW_POWER
            )

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    # Gridded field as compressed numpy archive
    out_npz = OUTPUT_DIR / "gridded_field.npz"
    np.savez_compressed(
        out_npz,
        gridded=gridded,
        timestamps=timestamps.astype(str),
        lat_centers=LAT_CENTERS,
        lon_centers=LON_CENTERS,
        components=np.array(COMPONENTS)
    )
    log.info(f"Saved gridded field: {out_npz} — shape {gridded.shape}")

    # Station metadata for later comparison (Phase 3, Comparison 4)
    meta_out = OUTPUT_DIR / "station_metadata.csv"
    station_meta.to_csv(meta_out)
    log.info(f"Saved station metadata: {meta_out}")

    # Neighbor distance statistics (km)
    dist_arr = np.array(all_neighbor_dists_km)
    neighbor_dist_stats = {
        "mean_km": float(np.mean(dist_arr)),
        "median_km": float(np.median(dist_arr)),
        "p95_km": float(np.percentile(dist_arr, 95)),
        "max_km": float(np.max(dist_arr)),
        "fraction_above_1000km": float(np.mean(dist_arr > 1000)),
        "fraction_above_2000km": float(np.mean(dist_arr > 2000)),
    }
    log.info(f"Neighbor distances — mean: {neighbor_dist_stats['mean_km']:.0f} km, "
             f"p95: {neighbor_dist_stats['p95_km']:.0f} km, "
             f"max: {neighbor_dist_stats['max_km']:.0f} km")

    # Summary statistics
    summary = {
        "n_timesteps": int(n_times),
        "n_stations": int(len(station_meta)),
        "grid_shape": [int(n_lat), int(n_lon)],
        "grid_resolution_deg": 5.0,
        "idw_k": K_NEIGHBORS,
        "idw_power": IDW_POWER,
        "station_count_per_timestep": {
            "min": int(np.min(station_counts)),
            "max": int(np.max(station_counts)),
            "mean": float(np.mean(station_counts))
        },
        "nan_fraction": float(np.isnan(gridded).mean()),
        "neighbor_distance_km": neighbor_dist_stats,
        "declaration": (
            "M_traditional: non-injective. Inverse distance weighting with k=4 "
            "imposes spatial coupling between stations that share no physical coupling. "
            "Pairwise differences between original station measurements are not preserved. "
            "All structure below 5-degree resolution is discarded. "
            "In polar gaps, oceanic regions, and the sparse southern hemisphere, "
            "the k=4 nearest stations may be separated by thousands of kilometers. "
            "The interpolation kernel in these regions creates synthetic coupling "
            "across distances that no local physical process connects. "
            "See neighbor_distance_km statistics for empirical extent."
        )
    }
    summary_out = OUTPUT_DIR / "phase1_1_summary.json"
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary: {summary_out}")

    return gridded, timestamps, summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR / "supermag_20150317.csv"
    gridded, timestamps, summary = run_interpolation(csv_path)

    log.info("Phase 1.1 complete.")
    log.info(f"  Grid shape: {gridded.shape}")
    log.info(f"  NaN fraction: {summary['nan_fraction']:.4f}")
    log.info(f"  Station counts: {summary['station_count_per_timestep']}")
