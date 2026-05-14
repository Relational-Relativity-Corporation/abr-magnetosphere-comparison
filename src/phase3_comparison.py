"""
Phase 3: Direct Comparison — ABR vs Traditional Pipeline
Metatron Dynamics, Inc.

Four comparisons, each demonstrating specific relational structure
that ABR preserves and the traditional pipeline destroys.

Requires:
  output/phase1/traditional_results.npz  (from phase1_traditional.py)
  output/phase2_1/abr_results.npz        (from phase2_1_abr_pipeline.py)

Produces:
  figures/comparison_temporal.png     — Γ vs SH power vs indices
  figures/comparison_spatial.png     — per-station σ² vs SH reconstruction
  figures/comparison_components.png  — component Γ vs independent SH power
  figures/comparison_residual.png    — A(residual) edge field analysis
  output/phase3/comparison_summary.json
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

FIGURES_DIR = Path("figures")
OUTPUT_DIR = Path("output/phase3")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMPONENTS = ["N", "E", "Z"]


# ===================================================================
# LOAD BOTH PIPELINE OUTPUTS
# ===================================================================

def load_results():
    log.info("Loading traditional pipeline results...")
    trad = np.load("output/phase1/traditional_results.npz", allow_pickle=True)

    log.info("Loading ABR pipeline results...")
    abr = np.load("output/phase2_1/abr_results.npz", allow_pickle=True)

    return trad, abr


# ===================================================================
# COMPARISON 1: TEMPORAL — Γ vs SH power vs indices
#
# The traditional pipeline produces single scalars per timestep
# (SYM-H, AE, SH power). ABR produces per-step Γ decomposed into
# spatial, component, and temporal contributions.
#
# This comparison shows where they diverge: where does Γ reveal
# structure that the scalar indices cannot represent?
# ===================================================================

def comparison_temporal(trad, abr):
    log.info("Comparison 1: Temporal resolution")

    # Traditional
    symh = trad["symh"]
    ae = trad["ae"]
    power_total = trad["power_total"]
    power_mag = trad["power_mag"]
    power_iono = trad["power_iono"]

    # ABR
    gamma_by_step = abr["gamma_by_step"]
    sigma_e_by_step = abr["sigma_e_by_step"]
    sigma_ba_by_step = abr["sigma_ba_by_step"]

    n_steps = len(gamma_by_step)
    t_axis = np.arange(n_steps)

    # Normalize for overlay (different units)
    def norm01(x):
        mn, mx = np.nanmin(x), np.nanmax(x)
        if mx - mn == 0:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(
        "Temporal Comparison: ABR Γ vs Traditional Diagnostics\n"
        "Same 2-hour window, same station data",
        fontsize=13, fontweight="bold"
    )

    # Panel 1: Raw Γ by step
    ax = axes[0]
    ax.plot(t_axis, gamma_by_step, "k-", linewidth=1.0, label="Γ total (per step)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    peak = np.argmax(gamma_by_step)
    ax.axvline(peak, color="red", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_ylabel("Γ (nT²)")
    ax.legend(loc="upper left")
    ax.set_title("ABR: R-sustained cross-topology circulation per step")
    ax.grid(True, alpha=0.2)

    # Panel 2: σ²(E) and σ²(BA) by step
    ax = axes[1]
    ax.plot(t_axis, sigma_e_by_step, "b-", linewidth=0.8, label="σ²(R∘B∘A)")
    ax.plot(t_axis, sigma_ba_by_step, "r-", linewidth=0.8, label="σ²(B∘A)")
    ax.axvline(peak, color="red", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_ylabel("σ² (nT²)")
    ax.legend(loc="upper left")
    ax.set_title("ABR: Full composition vs composition without R")
    ax.grid(True, alpha=0.2)

    # Panel 3: Traditional indices (normalized overlay)
    ax = axes[2]
    ax.plot(t_axis[:len(ae)], norm01(ae), "k-", linewidth=0.8, label="AE proxy (normalized)")
    ax.plot(t_axis[:len(symh)], norm01(-symh), "b-", linewidth=0.8,
            label="-SYM-H proxy (normalized)")
    ax.plot(t_axis[:len(power_total)], norm01(power_total), "r--", linewidth=0.8,
            label="Total SH power (normalized)")
    ax.axvline(peak, color="red", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_ylabel("Normalized [0,1]")
    ax.legend(loc="upper left")
    ax.set_title("Traditional: scalar indices (each collapses the spatial field to one number)")
    ax.grid(True, alpha=0.2)

    # Panel 4: Overlay — Γ normalized vs AE normalized
    ax = axes[3]
    ax.plot(t_axis, norm01(gamma_by_step), "k-", linewidth=1.2,
            label="Γ (normalized)")
    ax.plot(t_axis[:len(ae)], norm01(ae), "b--", linewidth=0.8,
            label="AE (normalized)")
    ax.plot(t_axis[:len(power_total)], norm01(power_total), "r--", linewidth=0.8,
            label="SH power (normalized)")
    ax.axvline(peak, color="red", linewidth=0.5, linestyle="--", alpha=0.6,
               label=f"Γ peak (step {peak})")
    ax.set_ylabel("Normalized [0,1]")
    ax.set_xlabel("Evolution step (minutes from window start)")
    ax.legend(loc="upper left")
    ax.set_title("Overlay: where do they diverge?")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = FIGURES_DIR / "comparison_temporal.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out}")

    return {
        "gamma_peak_step": int(peak),
        "gamma_peak_value": float(gamma_by_step[peak]),
        "ae_peak_step": int(np.argmax(ae)),
        "ae_peak_value": float(np.max(ae)),
        "temporal_offset_steps": int(peak) - int(np.argmax(ae)),
    }


# ===================================================================
# COMPARISON 2: SPATIAL — per-station σ² vs SH reconstruction
#
# At the storm peak step, map per-station σ² from ABR (where does
# relational structure concentrate?) against the traditional
# pipeline's reconstructed field at station locations (what does
# the SH basis represent there?).
# ===================================================================

def comparison_spatial(trad, abr):
    log.info("Comparison 2: Spatial resolution")

    station_lats = trad["station_lats"]
    station_lons = trad["station_lons"]

    # Traditional: measured vs reconstructed at storm peak
    measured = trad["station_measured"]       # (n_times, 3, n_stations)
    reconstructed = trad["station_reconstructed"]
    residual = trad["station_residual"]

    # Find peak step (from ABR gamma)
    gamma_by_step = abr["gamma_by_step"]
    peak_step = int(np.argmax(gamma_by_step))

    # Residual magnitude at peak (traditional)
    res_peak = residual[peak_step]  # (3, n_stations)
    res_mag = np.sqrt(np.nansum(res_peak**2, axis=0))  # per station

    # Measured magnitude at peak
    meas_peak = measured[peak_step]
    meas_mag = np.sqrt(np.nansum(meas_peak**2, axis=0))

    # Reconstructed magnitude at peak
    recon_peak = reconstructed[peak_step]
    recon_mag = np.sqrt(np.nansum(recon_peak**2, axis=0))

    # Fractional error per station
    frac_error = res_mag / (meas_mag + 1e-30)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Spatial Comparison at Storm Peak (step {peak_step})\n"
        "Left: Measured field | Center: SH reconstruction | "
        "Right: Information destroyed (residual)",
        fontsize=12, fontweight="bold"
    )

    def scatter_map(ax, lats, lons, values, title, cmap="inferno", label="",
                    use_log=True):
        valid = np.isfinite(values) & (values > 0)
        if valid.sum() == 0:
            ax.set_title(title + " (no data)")
            return
        norm = (mcolors.LogNorm(vmin=max(values[valid].min(), 1e-1),
                                vmax=values[valid].max())
                if use_log else None)
        sc = ax.scatter(
            lons[valid], lats[valid], c=values[valid],
            cmap=cmap, s=12, alpha=0.8, norm=norm, edgecolors="none"
        )
        ax.set_xlim(-10, 370)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title, fontsize=10)
        ax.axhline(65, color="cyan", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.axhline(-65, color="cyan", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.set_facecolor("#111111")
        plt.colorbar(sc, ax=ax, label=label, shrink=0.8)

    scatter_map(axes[0], station_lats, station_lons, meas_mag,
                "Measured |B| perturbation", cmap="viridis", label="|B| (nT)")

    scatter_map(axes[1], station_lats, station_lons, recon_mag,
                "SH reconstructed |B|", cmap="viridis", label="|B| (nT)")

    scatter_map(axes[2], station_lats, station_lons, res_mag,
                "Not preserved through traditional pipeline",
                cmap="inferno", label="|residual| (nT)")

    plt.tight_layout()
    out = FIGURES_DIR / "comparison_spatial.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out}")

    return {
        "peak_step": peak_step,
        "trad_residual_max_nT": float(np.nanmax(res_mag)),
        "trad_residual_mean_nT": float(np.nanmean(res_mag)),
        "mean_fractional_error": float(np.nanmean(frac_error)),
        "note": "ABR per-station σ² not yet computed in 3-topology pipeline. "
                "Spatial comparison shows traditional pipeline information loss only.",
    }


# ===================================================================
# COMPARISON 3: COMPONENT COUPLING
#
# Traditional pipeline processes N, E, Z independently through SH.
# Three separate power curves, no coupling between them.
# ABR processes them as a coupled multi-component field.
# Γ_component (99% of total) captures their interaction.
# ===================================================================

def comparison_components(trad, abr):
    log.info("Comparison 3: Component coupling")

    # Traditional: per-component power
    total_per_comp = trad["total_power_per_comp"]  # (n_times, 3)

    # ABR: total Γ and component Γ
    gamma_total = float(abr["gamma_total"])
    gamma_comp = float(abr["gamma_comp"])
    gamma_spatial = float(abr["gamma_spatial"])
    gamma_temporal = float(abr["gamma_temporal"])

    gamma_by_step = abr["gamma_by_step"]
    n_steps = len(gamma_by_step)
    t_axis = np.arange(n_steps)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        "Component Coupling: ABR detects inter-component structure\n"
        "Traditional pipeline processes N, E, Z independently",
        fontsize=13, fontweight="bold"
    )

    # Panel 1: Traditional — three independent power curves
    ax = axes[0]
    for c_idx, comp in enumerate(COMPONENTS):
        ax.plot(t_axis[:len(total_per_comp)], total_per_comp[:, c_idx],
                linewidth=0.8, label=f"{comp} component SH power")
    ax.set_ylabel("SH Power (nT²)")
    ax.legend(loc="upper left")
    ax.set_title("Traditional: N, E, Z processed independently (no coupling)")
    ax.grid(True, alpha=0.2)

    # Panel 2: ABR — Γ by step (dominated by component coupling)
    ax = axes[1]
    ax.plot(t_axis, gamma_by_step, "k-", linewidth=1.0, label="Γ total (per step)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Γ (nT²)")
    ax.set_xlabel("Evolution step")
    ax.legend(loc="upper left")
    ax.set_title(
        f"ABR: Γ decomposition — spatial: {gamma_spatial:.0f}, "
        f"component: {gamma_comp:.0f} ({100*gamma_comp/gamma_total:.1f}%), "
        f"temporal: {gamma_temporal:.0f}"
    )
    ax.grid(True, alpha=0.2)

    # Annotation
    ax.text(
        0.98, 0.92,
        f"Component coupling = {100*gamma_comp/gamma_total:.1f}% of total Γ\n"
        f"This structure is invisible to the traditional pipeline\n"
        f"because it never computes inter-component relations.",
        transform=ax.transAxes, fontsize=9, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#222", alpha=0.8),
        color="white", family="monospace"
    )

    plt.tight_layout()
    out = FIGURES_DIR / "comparison_components.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out}")

    return {
        "gamma_total": gamma_total,
        "gamma_spatial": gamma_spatial,
        "gamma_component": gamma_comp,
        "gamma_temporal": gamma_temporal,
        "component_fraction": gamma_comp / gamma_total if gamma_total != 0 else 0,
    }


# ===================================================================
# COMPARISON 4: INFORMATION LOSS — A(residual)
#
# The traditional pipeline's residual (measured - SH reconstructed
# at station locations) is the information it destroyed.
# Apply operator A to this residual. If the edge field is nontrivial,
# the destroyed information contained relational structure.
# ===================================================================

def comparison_residual(trad, abr):
    log.info("Comparison 4: Information loss — A(residual)")

    residual = trad["station_residual"]    # (n_times, 3, n_stations)
    measured = trad["station_measured"]
    station_lats = trad["station_lats"]
    station_lons = trad["station_lons"]
    station_codes = trad["station_codes"]

    n_times = residual.shape[0]
    n_stations = residual.shape[2]

    # Build proximity graph on station positions (same as ABR pipeline)
    max_edge_km = float(abr["max_edge_km"])
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

    log.info(f"  Residual graph: {len(edges)} edges on {n_stations} stations")

    # Apply A to the residual at each timestep
    # A(residual) = pairwise differences on the residual field
    residual_edge_variance = np.zeros(n_times)
    measured_edge_variance = np.zeros(n_times)

    for t_idx in range(n_times):
        res_vals = []
        meas_vals = []

        for (i, j) in edges:
            for c in range(3):
                rv_i = residual[t_idx, c, i]
                rv_j = residual[t_idx, c, j]
                mv_i = measured[t_idx, c, i]
                mv_j = measured[t_idx, c, j]

                if np.isfinite(rv_i) and np.isfinite(rv_j):
                    res_vals.append(rv_i - rv_j)
                if np.isfinite(mv_i) and np.isfinite(mv_j):
                    meas_vals.append(mv_i - mv_j)

        if res_vals:
            residual_edge_variance[t_idx] = np.var(res_vals)
        if meas_vals:
            measured_edge_variance[t_idx] = np.var(meas_vals)

    # Fraction of relational content destroyed
    fraction_destroyed = residual_edge_variance / (measured_edge_variance + 1e-30)

    t_axis = np.arange(n_times)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        "Representational Fidelity: Relational content of the traditional pipeline's residual\n"
        "A(residual) reveals relational structure not preserved through IDW → SH → source separation",
        fontsize=12, fontweight="bold"
    )

    # Panel 1: σ²(A(measured)) vs σ²(A(residual))
    ax = axes[0]
    ax.semilogy(t_axis, measured_edge_variance, "b-", linewidth=0.8,
                label="σ²(A(measured)) — relational content of raw data")
    ax.semilogy(t_axis, residual_edge_variance, "r-", linewidth=0.8,
                label="σ²(A(residual)) — relational content not preserved")
    ax.set_ylabel("σ² (nT²)")
    ax.legend(loc="upper left")
    ax.set_title("Edge field variance: measured vs destroyed")
    ax.grid(True, alpha=0.2)

    # Panel 2: Fraction destroyed
    ax = axes[1]
    ax.plot(t_axis, fraction_destroyed * 100, "r-", linewidth=0.8)
    ax.set_ylabel("% of relational content")
    ax.set_title("Fraction of edge-field variance not preserved by traditional pipeline")
    ax.grid(True, alpha=0.2)
    ax.axhline(np.mean(fraction_destroyed) * 100, color="red",
               linewidth=0.5, linestyle="--", alpha=0.6,
               label=f"Mean: {np.mean(fraction_destroyed)*100:.1f}%")
    ax.legend()

    # Panel 3: Residual RMS per step
    ax = axes[2]
    res_rms = np.sqrt(np.nanmean(residual**2, axis=(1, 2)))
    ax.plot(t_axis, res_rms, "k-", linewidth=0.8, label="Residual RMS (nT)")
    ax.set_ylabel("RMS (nT)")
    ax.set_xlabel("Evolution step")
    ax.set_title("Traditional pipeline reconstruction error per step")
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = FIGURES_DIR / "comparison_residual.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {out}")

    return {
        "mean_fraction_destroyed": float(np.mean(fraction_destroyed)),
        "peak_fraction_destroyed": float(np.max(fraction_destroyed)),
        "mean_residual_edge_variance": float(np.mean(residual_edge_variance)),
        "mean_measured_edge_variance": float(np.mean(measured_edge_variance)),
        "mean_residual_rms_nT": float(np.mean(res_rms)),
    }


# ===================================================================
# MAIN
# ===================================================================

def run_comparisons():
    trad, abr = load_results()

    results = {}

    results["temporal"] = comparison_temporal(trad, abr)
    log.info(f"  Temporal offset (Γ peak vs AE peak): "
             f"{results['temporal']['temporal_offset_steps']} steps")

    results["spatial"] = comparison_spatial(trad, abr)
    log.info(f"  Mean fractional error at stations: "
             f"{results['spatial']['mean_fractional_error']*100:.1f}%")

    results["components"] = comparison_components(trad, abr)
    log.info(f"  Component fraction of Γ: "
             f"{results['components']['component_fraction']*100:.1f}%")

    results["residual"] = comparison_residual(trad, abr)
    log.info(f"  Mean relational content destroyed: "
             f"{results['residual']['mean_fraction_destroyed']*100:.1f}%")

    # Save summary
    with open(OUTPUT_DIR / "comparison_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved: {OUTPUT_DIR / 'comparison_summary.json'}")

    # Print summary table
    log.info("\n" + "="*60)
    log.info("COMPARISON SUMMARY")
    log.info("="*60)
    log.info(f"Γ peak step: {results['temporal']['gamma_peak_step']} "
             f"(AE peak: {results['temporal']['ae_peak_step']}, "
             f"offset: {results['temporal']['temporal_offset_steps']} steps)")
    log.info(f"Component coupling: {results['components']['component_fraction']*100:.1f}% of Γ")
    log.info(f"Relational content destroyed: "
             f"{results['residual']['mean_fraction_destroyed']*100:.1f}% mean, "
             f"{results['residual']['peak_fraction_destroyed']*100:.1f}% peak")
    log.info(f"Spatial: mean fractional error "
             f"{results['spatial']['mean_fractional_error']*100:.1f}%")
    log.info("="*60)

    return results


if __name__ == "__main__":
    results = run_comparisons()
