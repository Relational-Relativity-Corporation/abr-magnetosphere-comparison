"""
Phase 1.4: Standard Diagnostics — SYM-H Proxy, AE Proxy, Power Time Series
Metatron Dynamics, Inc. — ABR vs Traditional Pipeline Comparison

Constructs the conventional summary indices from the separated SH fields:
  - SYM-H proxy: degree-1 N-component coefficient, scaled
  - AE proxy: max absolute perturbation at auroral latitudes from gridded field
  - Power time series per band and total
  - Diagnostic plots aligned to UT

These are the final outputs of the traditional pipeline. Everything downstream
of this is interpretation. The four non-injective transformations
(IDW → SH → source separation → index construction) are now complete.

DECLARATION: Index construction is the fourth non-injective transformation.
SYM-H collapses the entire magnetospheric field to a single scalar per timestep.
AE collapses auroral zone dynamics to a single scalar per timestep.
Both are projections with massive information loss. The traditional pipeline
treats these scalars as the signal. Everything they cannot represent is
declared noise or ignored.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_1_1 = Path("output/phase1_1")
INPUT_1_2 = Path("output/phase1_2")
INPUT_1_3 = Path("output/phase1_3")
OUTPUT_DIR = Path("output/phase1_4")
FIGURES_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COMPONENTS = ["N", "E", "Z"]

# Auroral zone latitude range (magnetic latitude, degrees)
AURORAL_LAT_MIN = 65.0
AURORAL_LAT_MAX = 75.0

# SYM-H proxy: low-latitude band for ring current detection
SYMH_LAT_MIN = -30.0
SYMH_LAT_MAX = 30.0

# ---------------------------------------------------------------------------
# Parse timestamps robustly
# ---------------------------------------------------------------------------
def parse_timestamps(ts_array):
    """Convert string timestamps to datetime objects."""
    out = []
    for s in ts_array:
        s = str(s)
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                out.append(datetime.strptime(s, fmt))
                break
            except ValueError:
                continue
        else:
            # numpy datetime64 fallback
            out.append(np.datetime64(s).astype("datetime64[s]").astype(datetime))
    return out

# ---------------------------------------------------------------------------
# SYM-H Proxy
# ---------------------------------------------------------------------------
def compute_symh_proxy(coefficients, coeff_index):
    """
    SYM-H proxy from degree-1, order-0 coefficient of the N component.

    The real SYM-H is derived from ~6 low-latitude stations with local-time
    correction. The SH proxy uses g_1^0 of the northward component, which
    captures the symmetric ring current depression — the same physical signal,
    but extracted via global decomposition rather than station selection.

    Parameters
    ----------
    coefficients : (n_times, n_comp, n_coeffs)
    coeff_index : list of (n, m, kind) tuples

    Returns
    -------
    symh : 1D array (n_times,), in nT (same units as input field)
    """
    # Find index of g_1^0
    g10_idx = None
    for idx, (n, m, kind) in enumerate(coeff_index):
        if n == 1 and m == 0 and kind == "g":
            g10_idx = idx
            break
    if g10_idx is None:
        raise ValueError("g_1^0 coefficient not found in coeff_index")

    # N component is index 0
    symh = coefficients[:, 0, g10_idx].copy()

    # Remove quiet-time baseline (first hour mean, assuming storm hasn't started)
    # This mimics the baseline subtraction in real SYM-H construction
    n_baseline = min(60, len(symh) // 4)  # first hour at 1-min cadence
    if n_baseline > 0:
        baseline = np.nanmean(symh[:n_baseline])
        symh -= baseline
        log.info(f"SYM-H proxy baseline removed: {baseline:.1f} nT "
                 f"(mean of first {n_baseline} timesteps)")

    return symh


# ---------------------------------------------------------------------------
# AE Proxy
# ---------------------------------------------------------------------------
def compute_ae_proxy(gridded_field, lat_centers, lon_centers,
                     lat_min=AURORAL_LAT_MIN, lat_max=AURORAL_LAT_MAX):
    """
    AE proxy from maximum absolute N-component perturbation in the auroral zone.

    The real AE index uses ~12 auroral-latitude stations, taking the max positive
    (AU) and max negative (AL) perturbations. AE = AU - AL.

    Our proxy: from the gridded field, select the auroral latitude band and
    compute AU and AL equivalents from the N component.

    Parameters
    ----------
    gridded_field : (n_times, n_comp, n_lat, n_lon) — original gridded field from 1.1
    lat_centers : 1D array of latitudes
    lon_centers : 1D array of longitudes

    Returns
    -------
    ae, au, al : 1D arrays (n_times,)
    """
    n_times = gridded_field.shape[0]

    # Select auroral latitude rows (both hemispheres)
    auroral_mask_north = (lat_centers >= lat_min) & (lat_centers <= lat_max)
    auroral_mask_south = (lat_centers >= -lat_max) & (lat_centers <= -lat_min)
    auroral_mask = auroral_mask_north | auroral_mask_south

    n_auroral = auroral_mask.sum()
    log.info(f"AE proxy: {n_auroral} latitude bins in auroral zone "
             f"([{lat_min},{lat_max}] and [{-lat_max},{-lat_min}])")

    # N component = index 0
    auroral_N = gridded_field[:, 0, auroral_mask, :]  # (n_times, n_auroral_lats, n_lon)

    # Flatten spatial dimensions for each timestep
    auroral_flat = auroral_N.reshape(n_times, -1)  # (n_times, n_auroral_points)

    # Remove quiet-time baseline per grid point
    n_baseline = min(60, n_times // 4)
    if n_baseline > 0:
        baseline = np.nanmean(auroral_flat[:n_baseline], axis=0, keepdims=True)
        auroral_flat = auroral_flat - baseline

    au = np.nanmax(auroral_flat, axis=1)    # max positive perturbation
    al = np.nanmin(auroral_flat, axis=1)    # max negative perturbation
    ae = au - al

    return ae, au, al


# ---------------------------------------------------------------------------
# Power Time Series
# ---------------------------------------------------------------------------
def compute_power_series(sh_power, n_max):
    """
    Aggregate power time series from per-degree power spectrum.

    Parameters
    ----------
    sh_power : (n_times, n_comp, n_max) — Lowes-Mauersberger power per degree

    Returns
    -------
    dict of power time series
    """
    return {
        "total": sh_power.sum(axis=(1, 2)),                          # (n_times,)
        "mag_deg_1_3": sh_power[:, :, 0:3].sum(axis=(1, 2)),        # (n_times,)
        "iono_deg_4_plus": sh_power[:, :, 3:].sum(axis=(1, 2)),     # (n_times,)
        "per_component_total": sh_power.sum(axis=2),                  # (n_times, 3)
        "degree_1": sh_power[:, :, 0].sum(axis=1),                   # (n_times,)
        "degree_2": sh_power[:, :, 1].sum(axis=1),                   # (n_times,)
        "degree_3": sh_power[:, :, 2].sum(axis=1),                   # (n_times,)
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_diagnostics(times, symh, ae, au, al, power, storm_peak_idx):
    """Four-panel diagnostic plot: SYM-H, AE, band power, degree power."""

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle("Traditional Pipeline Diagnostics — St. Patrick's Day Storm 2015",
                 fontsize=14, fontweight="bold")

    # Panel 1: SYM-H proxy
    ax = axes[0]
    ax.plot(times, symh, "b-", linewidth=0.8, label="SYM-H proxy (g₁⁰)")
    ax.axvline(times[storm_peak_idx], color="red", linestyle="--", alpha=0.5, label="Storm peak")
    ax.set_ylabel("SYM-H proxy (nT)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    # Panel 2: AE proxy
    ax = axes[1]
    ax.plot(times, ae, "k-", linewidth=0.8, label="AE proxy")
    ax.plot(times, au, "r-", linewidth=0.5, alpha=0.6, label="AU")
    ax.plot(times, al, "b-", linewidth=0.5, alpha=0.6, label="AL")
    ax.axvline(times[storm_peak_idx], color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("AE / AU / AL proxy (nT)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel 3: Band power
    ax = axes[2]
    ax.semilogy(times, power["mag_deg_1_3"], "b-", linewidth=0.8, label="Magnetospheric (deg 1-3)")
    ax.semilogy(times, power["iono_deg_4_plus"], "r-", linewidth=0.8, label="Ionospheric (deg 4+)")
    ax.semilogy(times, power["total"], "k--", linewidth=0.5, alpha=0.5, label="Total")
    ax.axvline(times[storm_peak_idx], color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("SH Power (nT²)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel 4: Low-degree power (degree 1, 2, 3 individually)
    ax = axes[3]
    ax.semilogy(times, power["degree_1"], "-", linewidth=0.8, label="Degree 1")
    ax.semilogy(times, power["degree_2"], "-", linewidth=0.8, label="Degree 2")
    ax.semilogy(times, power["degree_3"], "-", linewidth=0.8, label="Degree 3")
    ax.axvline(times[storm_peak_idx], color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("SH Power (nT²)")
    ax.set_xlabel("UTC")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    plt.tight_layout()
    out_path = FIGURES_DIR / "phase1_4_traditional_diagnostics.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved diagnostic plot: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_diagnostics():
    # Load Phase 1.1 (gridded field — needed for AE proxy from spatial data)
    log.info("Loading Phase 1.1 gridded field")
    g1 = np.load(INPUT_1_1 / "gridded_field.npz", allow_pickle=True)
    gridded = g1["gridded"]
    lat_centers = g1["lat_centers"]
    lon_centers = g1["lon_centers"]

    # Load Phase 1.2 (SH coefficients and power)
    log.info("Loading Phase 1.2 SH coefficients")
    g2 = np.load(INPUT_1_2 / "sh_coefficients.npz", allow_pickle=True)
    coefficients = g2["coefficients"]
    sh_power = g2["power_spectrum"]
    timestamps_raw = g2["timestamps"]
    coeff_index_raw = g2["coeff_index"]
    n_max = int(g2["n_max"])

    coeff_index = [(int(r[0]), int(r[1]), str(r[2])) for r in coeff_index_raw]
    times = parse_timestamps(timestamps_raw)
    n_times = len(times)

    # Load Phase 1.3 (for cross-referencing power)
    log.info("Loading Phase 1.3 separated fields")
    g3 = np.load(INPUT_1_3 / "separated_fields.npz", allow_pickle=True)
    total_power_1_3 = g3["total_power"]  # (n_times, n_comp)

    # -------------------------------------------------------------------
    # Compute indices
    # -------------------------------------------------------------------
    log.info("Computing SYM-H proxy")
    symh = compute_symh_proxy(coefficients, coeff_index)

    log.info("Computing AE proxy")
    ae, au, al = compute_ae_proxy(gridded, lat_centers, lon_centers)

    log.info("Computing power time series")
    power = compute_power_series(sh_power, n_max)

    # Storm peak from total SH power (consistent with Phase 1.3)
    storm_peak_idx = np.argmax(power["total"])
    storm_peak_time = times[storm_peak_idx]
    log.info(f"Storm peak at timestep {storm_peak_idx}: {storm_peak_time}")

    # -------------------------------------------------------------------
    # Key storm statistics
    # -------------------------------------------------------------------
    symh_min_idx = np.argmin(symh)
    ae_max_idx = np.argmax(ae)

    storm_stats = {
        "storm_peak": {
            "index": int(storm_peak_idx),
            "timestamp": str(storm_peak_time),
            "method": "argmax(total_SH_power)",
        },
        "symh_proxy": {
            "minimum_nT": float(np.nanmin(symh)),
            "minimum_timestamp": str(times[symh_min_idx]),
            "quiet_baseline_nT": 0.0,  # removed by construction
        },
        "ae_proxy": {
            "maximum_nT": float(np.nanmax(ae)),
            "maximum_timestamp": str(times[ae_max_idx]),
            "au_max_nT": float(np.nanmax(au)),
            "al_min_nT": float(np.nanmin(al)),
        },
        "power": {
            "total_dynamic_range": float(np.nanmax(power["total"]) /
                                         (np.nanmin(power["total"]) + 1e-30)),
            "mag_band_peak": float(np.nanmax(power["mag_deg_1_3"])),
            "iono_band_peak": float(np.nanmax(power["iono_deg_4_plus"])),
        },
    }
    log.info(f"SYM-H min: {storm_stats['symh_proxy']['minimum_nT']:.1f} nT "
             f"at {storm_stats['symh_proxy']['minimum_timestamp']}")
    log.info(f"AE max: {storm_stats['ae_proxy']['maximum_nT']:.1f} nT "
             f"at {storm_stats['ae_proxy']['maximum_timestamp']}")

    # -------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------
    log.info("Generating diagnostic plot")
    plot_path = plot_diagnostics(times, symh, ae, au, al, power, storm_peak_idx)

    # -------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------
    out_npz = OUTPUT_DIR / "diagnostics.npz"
    np.savez_compressed(
        out_npz,
        symh=symh,
        ae=ae,
        au=au,
        al=al,
        power_total=power["total"],
        power_mag=power["mag_deg_1_3"],
        power_iono=power["iono_deg_4_plus"],
        power_deg1=power["degree_1"],
        power_deg2=power["degree_2"],
        power_deg3=power["degree_3"],
        power_per_component=power["per_component_total"],
        timestamps=timestamps_raw,
        storm_peak_idx=storm_peak_idx,
    )
    log.info(f"Saved diagnostics: {out_npz}")

    # Summary
    summary = {
        "storm_statistics": storm_stats,
        "n_timesteps": int(n_times),
        "auroral_zone_deg": [AURORAL_LAT_MIN, AURORAL_LAT_MAX],
        "figure": str(plot_path),
        "declaration": (
            "Fourth non-injective transformation: index construction. "
            "SYM-H proxy collapses the magnetospheric field to a single scalar "
            "per timestep via the g_1^0 coefficient — a weighted integral over the "
            "entire sphere projected onto the lowest-degree zonal harmonic. "
            "AE proxy collapses auroral-zone dynamics to a single scalar per "
            "timestep via the maximum absolute perturbation in a latitude band. "
            "Both indices discard all spatial structure, all component coupling, "
            "all local relational information between stations. "
            "The traditional pipeline treats these scalars as the definitive "
            "characterization of storm dynamics. Everything they cannot represent "
            "is invisible to the analysis."
        ),
    }
    summary_out = OUTPUT_DIR / "phase1_4_summary.json"
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary: {summary_out}")

    return symh, ae, power, summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    symh, ae, power, summary = run_diagnostics()
    log.info("Phase 1.4 complete. Traditional pipeline finished.")
    log.info(f"  SYM-H range: [{np.nanmin(symh):.1f}, {np.nanmax(symh):.1f}] nT")
    log.info(f"  AE range: [{np.nanmin(ae):.1f}, {np.nanmax(ae):.1f}] nT")
    log.info(f"  Total power dynamic range: "
             f"{summary['storm_statistics']['power']['total_dynamic_range']:.0f}x")
    log.info("  === Traditional pipeline complete. Four non-injective transformations applied. ===")
