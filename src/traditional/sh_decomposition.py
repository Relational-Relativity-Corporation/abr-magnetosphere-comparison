"""
Phase 1.2: Spherical Harmonic Decomposition
Metatron Dynamics, Inc. — ABR vs Traditional Pipeline Comparison

Decomposes the gridded field from Phase 1.1 into spherical harmonic coefficients
g_n^m(t), h_n^m(t) for each magnetic field component, degree 1 through N_MAX.

DECLARATION: This is the second non-injective transformation.
Preserved: spectral power per degree.
Discarded: phase relationships between local stations, cross-scale coupling,
all spatial structure that is not representable as a weighted sum of global
orthogonal basis functions. Every coefficient is a weighted integral over the
entire sphere — local relational structure between nearby stations is projected
into global modes and cannot be recovered.
"""

import numpy as np
from scipy.special import lpmv
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_DIR = Path("output/phase1_1")
OUTPUT_DIR = Path("output/phase1_2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_MAX = 18          # maximum SH degree
COMPONENTS = ["N", "E", "Z"]

# ---------------------------------------------------------------------------
# Schmidt Semi-Normalized Associated Legendre Functions
# ---------------------------------------------------------------------------
def schmidt_semi_norm_factor(n, m):
    """
    Conversion factor from scipy's fully-normalized Legendre functions
    to Schmidt semi-normalized, as used in geomagnetism.

    Schmidt semi-normalization:
        S_n^m = sqrt(2 * (n-m)! / (n+m)!)  for m > 0
        S_n^0 = 1

    scipy.special.lpmv returns unnormalized P_n^m(x) with Condon-Shortley phase.
    We compute Schmidt-normalized P_n^m directly.
    """
    if m == 0:
        return 1.0
    # Use log-factorial for numerical stability at high degree
    from math import lgamma
    log_factor = 0.5 * (np.log(2.0) + lgamma(n - m + 1) - lgamma(n + m + 1))
    return np.exp(log_factor)


def schmidt_Pnm(n, m, cos_theta):
    """
    Schmidt semi-normalized associated Legendre function P_n^m(cos(theta)).

    Parameters
    ----------
    n : int, degree
    m : int, order (>= 0)
    cos_theta : array-like, cosine of colatitude

    Returns
    -------
    array of P_n^m values, Schmidt semi-normalized
    """
    # scipy lpmv computes the unnormalized associated Legendre function
    # with Condon-Shortley phase: (-1)^m included
    raw = lpmv(m, n, cos_theta)
    S = schmidt_semi_norm_factor(n, m)
    return S * raw


# ---------------------------------------------------------------------------
# Build design matrix for a single (lat, lon) grid
# ---------------------------------------------------------------------------
def build_sh_design_matrix(lats_deg, lons_deg, n_max):
    """
    Build the design matrix Y such that field = Y @ coeffs.

    For a scalar field on the sphere, the SH expansion is:
        f(theta, phi) = sum_{n=1}^{N} sum_{m=0}^{n}
            [g_n^m * cos(m*phi) + h_n^m * sin(m*phi)] * P_n^m(cos(theta))

    Parameters
    ----------
    lats_deg : 1D array, geographic latitudes in degrees
    lons_deg : 1D array, geographic longitudes in degrees
        (These define a meshgrid; total points = len(lats) * len(lons))
    n_max : int, maximum degree

    Returns
    -------
    Y : 2D array, shape (n_grid_points, n_coeffs)
    coeff_index : list of (n, m, type) tuples where type is 'g' or 'h'
    """
    # Colatitude and longitude in radians
    colat_rad = np.radians(90.0 - lats_deg)
    lon_rad = np.radians(lons_deg)

    # Meshgrid (colat varies along rows, lon along columns)
    colat_grid, lon_grid = np.meshgrid(colat_rad, lon_rad, indexing="ij")
    cos_colat = np.cos(colat_grid).ravel()
    lon_flat = lon_grid.ravel()
    n_pts = len(cos_colat)

    # Count coefficients: for each n from 1..n_max, m from 0..n
    # m=0: one coeff (g_n^0). m>0: two coeffs (g_n^m, h_n^m)
    coeff_index = []
    for n in range(1, n_max + 1):
        for m in range(0, n + 1):
            coeff_index.append((n, m, "g"))
            if m > 0:
                coeff_index.append((n, m, "h"))
    n_coeffs = len(coeff_index)

    log.info(f"Design matrix: {n_pts} grid points x {n_coeffs} coefficients (n_max={n_max})")

    Y = np.zeros((n_pts, n_coeffs), dtype=np.float64)

    # Pre-compute all P_n^m(cos_colat) — cache by (n, m)
    Pnm_cache = {}
    for n in range(1, n_max + 1):
        for m in range(0, n + 1):
            Pnm_cache[(n, m)] = schmidt_Pnm(n, m, cos_colat)

    for col_idx, (n, m, kind) in enumerate(coeff_index):
        Pnm = Pnm_cache[(n, m)]
        if kind == "g":
            Y[:, col_idx] = Pnm * np.cos(m * lon_flat)
        else:  # h
            Y[:, col_idx] = Pnm * np.sin(m * lon_flat)

    return Y, coeff_index


# ---------------------------------------------------------------------------
# Least-squares SH fit for one field snapshot
# ---------------------------------------------------------------------------
def fit_sh_coefficients(Y, field_flat, regularization=0.0):
    """
    Solve Y @ c = f in least-squares sense.

    Parameters
    ----------
    Y : design matrix, (n_pts, n_coeffs)
    field_flat : 1D array, (n_pts,)
    regularization : Tikhonov parameter (0 = no regularization)

    Returns
    -------
    coeffs : 1D array of SH coefficients
    residual_norm : L2 norm of the fit residual
    """
    valid = ~np.isnan(field_flat)
    if valid.sum() < Y.shape[1]:
        log.warning(f"Underdetermined: {valid.sum()} valid points < {Y.shape[1]} coefficients")
        # Return NaN coefficients rather than a garbage fit
        return np.full(Y.shape[1], np.nan), np.nan

    Yv = Y[valid]
    fv = field_flat[valid]

    if regularization > 0:
        # Tikhonov: minimize ||Y c - f||^2 + lambda ||c||^2
        A = Yv.T @ Yv + regularization * np.eye(Yv.shape[1])
        b = Yv.T @ fv
        coeffs = np.linalg.solve(A, b)
    else:
        coeffs, residuals, rank, sv = np.linalg.lstsq(Yv, fv, rcond=None)

    residual_norm = np.linalg.norm(Yv @ coeffs - fv)
    return coeffs, residual_norm


# ---------------------------------------------------------------------------
# Compute spectral power per degree
# ---------------------------------------------------------------------------
def power_per_degree(coeffs, coeff_index, n_max):
    """
    Lowes-Mauersberger spatial power spectrum:
        R_n = (n+1) * sum_{m=0}^{n} [(g_n^m)^2 + (h_n^m)^2]

    Returns dict mapping degree n to power R_n.
    """
    power = {}
    for n in range(1, n_max + 1):
        pn = 0.0
        for idx, (nn, mm, kind) in enumerate(coeff_index):
            if nn == n:
                pn += coeffs[idx] ** 2
        power[n] = (n + 1) * pn
    return power


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_sh_decomposition():
    # Load Phase 1.1 output
    log.info("Loading gridded field from Phase 1.1")
    data = np.load(INPUT_DIR / "gridded_field.npz", allow_pickle=True)
    gridded = data["gridded"]           # (n_times, 3, n_lat, n_lon)
    timestamps = data["timestamps"]
    lat_centers = data["lat_centers"]
    lon_centers = data["lon_centers"]
    components = data["components"]

    n_times, n_comp, n_lat, n_lon = gridded.shape
    n_pts = n_lat * n_lon
    log.info(f"Grid: {n_lat} x {n_lon}, {n_times} timesteps, {n_comp} components")

    # Build design matrix (same for all timesteps — grid doesn't change)
    Y, coeff_index = build_sh_design_matrix(lat_centers, lon_centers, N_MAX)
    n_coeffs = len(coeff_index)

    # Pre-allocate outputs
    all_coeffs = np.full((n_times, n_comp, n_coeffs), np.nan, dtype=np.float64)
    all_residuals = np.full((n_times, n_comp), np.nan, dtype=np.float64)
    all_power = np.zeros((n_times, n_comp, N_MAX), dtype=np.float64)  # degree 1..N_MAX

    # Fit each timestep and component
    for t_idx in range(n_times):
        if t_idx % 100 == 0:
            log.info(f"  Fitting timestep {t_idx}/{n_times}: {timestamps[t_idx]}")

        for c_idx in range(n_comp):
            field = gridded[t_idx, c_idx].ravel()  # (n_pts,)

            if np.all(np.isnan(field)):
                continue

            coeffs, res_norm = fit_sh_coefficients(Y, field)
            all_coeffs[t_idx, c_idx] = coeffs
            all_residuals[t_idx, c_idx] = res_norm

            # Power spectrum
            pwr = power_per_degree(coeffs, coeff_index, N_MAX)
            for n in range(1, N_MAX + 1):
                all_power[t_idx, c_idx, n - 1] = pwr[n]

    # -----------------------------------------------------------------------
    # Information loss: residual statistics
    # -----------------------------------------------------------------------
    # The residual measures what the SH basis cannot represent at this truncation
    valid_res = all_residuals[~np.isnan(all_residuals)]
    residual_stats = {
        "mean": float(np.mean(valid_res)),
        "median": float(np.median(valid_res)),
        "p95": float(np.percentile(valid_res, 95)),
        "max": float(np.max(valid_res)),
    }
    log.info(f"Fit residual L2 — mean: {residual_stats['mean']:.2f}, "
             f"p95: {residual_stats['p95']:.2f}, max: {residual_stats['max']:.2f}")

    # -----------------------------------------------------------------------
    # Reconstruction capability: store design matrix for Phase 3.4
    # -----------------------------------------------------------------------
    # Phase 3.4 will reconstruct station values from SH coefficients and
    # compare to originals. Save what's needed.

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    out_npz = OUTPUT_DIR / "sh_coefficients.npz"
    np.savez_compressed(
        out_npz,
        coefficients=all_coeffs,        # (n_times, n_comp, n_coeffs)
        residuals=all_residuals,         # (n_times, n_comp)
        power_spectrum=all_power,        # (n_times, n_comp, N_MAX)
        timestamps=timestamps,
        components=components,
        coeff_index=np.array([(n, m, k) for n, m, k in coeff_index], dtype=object),
        n_max=N_MAX,
        lat_centers=lat_centers,
        lon_centers=lon_centers,
    )
    log.info(f"Saved SH coefficients: {out_npz}")

    # Save design matrix for reconstruction in Phase 3.4
    Y_out = OUTPUT_DIR / "design_matrix.npz"
    np.savez_compressed(Y_out, Y=Y)
    log.info(f"Saved design matrix: {Y_out}")

    # Summary
    # Total power by band across all timesteps (for quick inspection)
    mag_band = all_power[:, :, 0:3].sum(axis=2)    # degree 1-3
    iono_band = all_power[:, :, 3:].sum(axis=2)     # degree 4+
    total_power = all_power.sum(axis=2)

    summary = {
        "n_max": N_MAX,
        "n_coefficients": n_coeffs,
        "n_timesteps": int(n_times),
        "components": COMPONENTS,
        "fit_residual_L2": residual_stats,
        "power_bands": {
            "magnetospheric_deg_1_3": {
                "mean_fraction_of_total": float(
                    np.nanmean(mag_band / (total_power + 1e-30))
                )
            },
            "ionospheric_deg_4_plus": {
                "mean_fraction_of_total": float(
                    np.nanmean(iono_band / (total_power + 1e-30))
                )
            },
        },
        "declaration": (
            "Non-injective: global aggregation. Every SH coefficient is a weighted "
            "integral over the entire sphere. Local relational structure between "
            "nearby stations — pairwise differences, local gradients, neighborhood "
            "coupling patterns — is projected into global orthogonal modes. "
            "The orthogonality assumption enforces that cross-scale coupling between "
            "degrees is exactly zero by construction. This is an assumption about "
            "the physics, not a measurement of it. During geomagnetic storms, "
            "ring current asymmetry and substorm injection produce localized structure "
            "that violates the spectral separability assumption. The SH basis "
            f"at degree {N_MAX} retains {n_coeffs} coefficients from {n_pts} grid points "
            f"({100 * n_coeffs / n_pts:.1f}% of the gridded information, which is itself "
            "already reduced from station-level data by Phase 1.1)."
        ),
    }
    summary_out = OUTPUT_DIR / "phase1_2_summary.json"
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary: {summary_out}")

    return all_coeffs, all_power, summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    coeffs, power, summary = run_sh_decomposition()
    log.info("Phase 1.2 complete.")
    log.info(f"  Coefficients shape: {coeffs.shape}")
    log.info(f"  Magnetospheric band fraction: "
             f"{summary['power_bands']['magnetospheric_deg_1_3']['mean_fraction_of_total']:.3f}")
    log.info(f"  Fit residual (mean L2): {summary['fit_residual_L2']['mean']:.2f}")
