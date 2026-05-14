"""
Phase 1.3: Source Separation by Spherical Harmonic Degree
Metatron Dynamics, Inc. — ABR vs Traditional Pipeline Comparison

Splits the SH coefficients from Phase 1.2 into magnetospheric (degree 1-3)
and ionospheric (degree 4+) bands. Reconstructs the gridded field for each
band separately.

DECLARATION: This is the third non-injective transformation.
Preserved: power within each band.
Discarded: cross-band coupling — the physical interaction between
magnetospheric and ionospheric current systems during storms. The separation
assumes spectral separability: that degree encodes source identity. During
geomagnetic storms this assumption fails. Ring current asymmetry pushes
magnetospheric power above degree 2. Substorm injection produces localized
structure that leaks across the degree boundary. The separation enforces
zero coupling between bands by construction, not by measurement.
"""

import numpy as np
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_DIR = Path("output/phase1_2")
OUTPUT_DIR = Path("output/phase1_3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMPONENTS = ["N", "E", "Z"]

# Source separation boundaries
MAG_DEG_MIN = 1     # magnetospheric band lower bound
MAG_DEG_MAX = 3     # magnetospheric band upper bound
IONO_DEG_MIN = 4    # ionospheric band lower bound
# IONO_DEG_MAX = N_MAX (loaded from Phase 1.2 output)

# Sensitivity analysis cutoffs
SENSITIVITY_CUTOFFS = [2, 3, 4]  # test magnetospheric ceiling at each

# ---------------------------------------------------------------------------
# Reconstruct gridded field from a subset of SH coefficients
# ---------------------------------------------------------------------------
def reconstruct_from_band(Y, coeffs_all, coeff_index, deg_min, deg_max):
    """
    Reconstruct gridded field using only coefficients in degree range [deg_min, deg_max].

    Parameters
    ----------
    Y : design matrix, (n_pts, n_coeffs)
    coeffs_all : full coefficient vector, (n_coeffs,)
    coeff_index : list of (n, m, kind) tuples
    deg_min, deg_max : inclusive degree range

    Returns
    -------
    field : 1D array (n_pts,), reconstructed field from selected band
    """
    masked = np.zeros_like(coeffs_all)
    for idx, (n, m, kind) in enumerate(coeff_index):
        if deg_min <= n <= deg_max:
            masked[idx] = coeffs_all[idx]
    return Y @ masked


# ---------------------------------------------------------------------------
# Compute band power from coefficients
# ---------------------------------------------------------------------------
def band_power(coeffs, coeff_index, deg_min, deg_max):
    """Lowes-Mauersberger power summed over degree range."""
    total = 0.0
    for idx, (n, m, kind) in enumerate(coeff_index):
        if deg_min <= n <= deg_max:
            total += (n + 1) * coeffs[idx] ** 2
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_source_separation():
    # Load Phase 1.2 outputs
    log.info("Loading SH coefficients from Phase 1.2")
    sh_data = np.load(INPUT_DIR / "sh_coefficients.npz", allow_pickle=True)
    all_coeffs = sh_data["coefficients"]       # (n_times, n_comp, n_coeffs)
    timestamps = sh_data["timestamps"]
    components = sh_data["components"]
    coeff_index_raw = sh_data["coeff_index"]   # array of (n, m, kind)
    n_max = int(sh_data["n_max"])
    lat_centers = sh_data["lat_centers"]
    lon_centers = sh_data["lon_centers"]

    # Reconstruct coeff_index as list of tuples
    coeff_index = [(int(row[0]), int(row[1]), str(row[2])) for row in coeff_index_raw]

    # Load design matrix
    Y_data = np.load(INPUT_DIR / "design_matrix.npz")
    Y = Y_data["Y"]

    n_times, n_comp, n_coeffs = all_coeffs.shape
    n_lat = len(lat_centers)
    n_lon = len(lon_centers)
    log.info(f"{n_times} timesteps, {n_comp} components, n_max={n_max}")

    # -------------------------------------------------------------------
    # Reconstruct separated fields
    # -------------------------------------------------------------------
    mag_field = np.full((n_times, n_comp, n_lat, n_lon), np.nan, dtype=np.float64)
    iono_field = np.full((n_times, n_comp, n_lat, n_lon), np.nan, dtype=np.float64)

    # Power time series per band
    mag_power = np.zeros((n_times, n_comp), dtype=np.float64)
    iono_power = np.zeros((n_times, n_comp), dtype=np.float64)
    total_power = np.zeros((n_times, n_comp), dtype=np.float64)

    # Cross-band power ratio (diagnostic for the separation assumption)
    # If magnetospheric power leaks above MAG_DEG_MAX, this ratio shifts
    cross_band_ratio = np.zeros((n_times, n_comp), dtype=np.float64)

    for t_idx in range(n_times):
        if t_idx % 100 == 0:
            log.info(f"  Separating timestep {t_idx}/{n_times}")

        for c_idx in range(n_comp):
            c = all_coeffs[t_idx, c_idx]
            if np.any(np.isnan(c)):
                continue

            # Reconstruct each band
            mag_flat = reconstruct_from_band(Y, c, coeff_index, MAG_DEG_MIN, MAG_DEG_MAX)
            iono_flat = reconstruct_from_band(Y, c, coeff_index, IONO_DEG_MIN, n_max)

            mag_field[t_idx, c_idx] = mag_flat.reshape(n_lat, n_lon)
            iono_field[t_idx, c_idx] = iono_flat.reshape(n_lat, n_lon)

            # Power per band
            mp = band_power(c, coeff_index, MAG_DEG_MIN, MAG_DEG_MAX)
            ip = band_power(c, coeff_index, IONO_DEG_MIN, n_max)
            tp = band_power(c, coeff_index, 1, n_max)

            mag_power[t_idx, c_idx] = mp
            iono_power[t_idx, c_idx] = ip
            total_power[t_idx, c_idx] = tp
            cross_band_ratio[t_idx, c_idx] = ip / (mp + 1e-30)

    # -------------------------------------------------------------------
    # Sensitivity analysis: vary the degree cutoff
    # -------------------------------------------------------------------
    log.info("Running sensitivity analysis on degree cutoff")
    sensitivity = {}
    for cutoff in SENSITIVITY_CUTOFFS:
        mag_pwr_sens = np.zeros((n_times, n_comp), dtype=np.float64)
        iono_pwr_sens = np.zeros((n_times, n_comp), dtype=np.float64)

        for t_idx in range(n_times):
            for c_idx in range(n_comp):
                c = all_coeffs[t_idx, c_idx]
                if np.any(np.isnan(c)):
                    continue
                mag_pwr_sens[t_idx, c_idx] = band_power(c, coeff_index, 1, cutoff)
                iono_pwr_sens[t_idx, c_idx] = band_power(c, coeff_index, cutoff + 1, n_max)

        # Fraction of total power attributed to magnetosphere at this cutoff
        frac = mag_pwr_sens / (mag_pwr_sens + iono_pwr_sens + 1e-30)

        # Storm peak: maximum total power across all components
        peak_idx = np.argmax(total_power.sum(axis=1))

        sensitivity[f"cutoff_deg_{cutoff}"] = {
            "mag_power_fraction_mean": float(np.nanmean(frac)),
            "mag_power_fraction_storm_peak": float(np.nanmean(frac[peak_idx])),
            "storm_peak_timestep": int(peak_idx),
            "storm_peak_timestamp": str(timestamps[peak_idx]),
        }
        log.info(f"  Cutoff degree {cutoff}: mean mag fraction = "
                 f"{sensitivity[f'cutoff_deg_{cutoff}']['mag_power_fraction_mean']:.3f}")

    # -------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------
    out_npz = OUTPUT_DIR / "separated_fields.npz"
    np.savez_compressed(
        out_npz,
        mag_field=mag_field,             # (n_times, n_comp, n_lat, n_lon)
        iono_field=iono_field,
        mag_power=mag_power,             # (n_times, n_comp)
        iono_power=iono_power,
        total_power=total_power,
        cross_band_ratio=cross_band_ratio,
        timestamps=timestamps,
        components=np.array(COMPONENTS),
        lat_centers=lat_centers,
        lon_centers=lon_centers,
        mag_deg_range=[MAG_DEG_MIN, MAG_DEG_MAX],
        iono_deg_range=[IONO_DEG_MIN, n_max],
    )
    log.info(f"Saved separated fields: {out_npz}")

    # -------------------------------------------------------------------
    # Cross-band ratio analysis — the core diagnostic
    # -------------------------------------------------------------------
    # During quiet times, most power sits in degree 1 (main field residual +
    # symmetric ring current). During storms, if power shifts to higher degrees,
    # the ratio changes — meaning the fixed cutoff misattributes sources.
    cbr_all = cross_band_ratio[~np.isnan(cross_band_ratio.sum(axis=1))]
    cbr_stats = {
        "mean": float(np.mean(cbr_all)),
        "std": float(np.std(cbr_all)),
        "min": float(np.min(cbr_all)),
        "max": float(np.max(cbr_all)),
    }

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    summary = {
        "magnetospheric_band": f"degree {MAG_DEG_MIN}-{MAG_DEG_MAX}",
        "ionospheric_band": f"degree {IONO_DEG_MIN}-{n_max}",
        "n_timesteps": int(n_times),
        "cross_band_power_ratio": cbr_stats,
        "sensitivity_analysis": sensitivity,
        "declaration": (
            "Non-injective: spectral source separation by degree. "
            f"Magnetospheric sources assigned to degree {MAG_DEG_MIN}-{MAG_DEG_MAX}, "
            f"ionospheric to degree {IONO_DEG_MIN}-{n_max}. "
            "Cross-band coupling is set to exactly zero by construction. "
            "This is not a measurement — it is an assumption that source identity "
            "maps to spectral degree. During storms, ring current asymmetry "
            "produces magnetospheric structure at degree 3+, and substorm "
            "injection creates localized features that span the degree boundary. "
            "The separation discards this coupling before any analysis occurs. "
            "Sensitivity analysis at cutoff degrees "
            f"{SENSITIVITY_CUTOFFS} quantifies how the attribution shifts."
        ),
    }
    summary_out = OUTPUT_DIR / "phase1_3_summary.json"
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary: {summary_out}")

    return mag_field, iono_field, summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mag, iono, summary = run_source_separation()
    log.info("Phase 1.3 complete.")
    log.info(f"  Mag field shape: {mag.shape}")
    log.info(f"  Cross-band ratio: {summary['cross_band_power_ratio']}")
    log.info(f"  Sensitivity: {summary['sensitivity_analysis']}")
