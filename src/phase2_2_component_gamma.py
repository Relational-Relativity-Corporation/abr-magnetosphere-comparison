#!/usr/bin/env python3
"""
Phase 2.2 — Component-Resolved Γ Time Series
ABR vs Traditional Pipeline — Magnetospheric Storm Comparison
Metatron Dynamics, Inc.

PURPOSE:
    Decompose Phase 2.1's per-step Γ into per-topology contributions:
      Γ_spatial(t), Γ_comp(t), Γ_temporal(t)

    Uses Phase 2.1's exact operators and data structures. Does not
    reimplement the pipeline — imports and extends it.

METHOD:
    Phase 2.1 already computes gamma_by_step as total Γ per timestep
    (spatial + component + temporal edges pooled at each step).
    Phase 2.2 extends this by computing σ² per edge type per step,
    both with R and without R, using the same dict-based edge fields.

    Γ_spatial(t)  = σ²(R_spatial at step t) − σ²(B_spatial at step t)
    Γ_comp(t)     = σ²(R_comp at step t)    − σ²(B_comp at step t)
    Γ_temporal(t) = σ²(R_temporal at step t) − σ²(B_temporal at step t)

    Where σ² is computed identically to Phase 2.1: flatten all values
    for that edge type at that step, take np.var.

DECLARATION:
    No interpolation, gridding, spectral projection, or smoothing
    applied prior to operator A. Pre-A steps within M: topology
    declaration (proximity graph), NaN exclusion, edge filtering.
    Declared structural commitments within M.

    σ² per edge type per step is a declared projection.
    Preserved: variance within each topology at each timestep.
    Discarded: cross-topology covariance, per-edge detail,
    spatial distribution within each type.

OUTPUT:
    output/phase2_2/component_gamma.npz
    output/phase2_2/component_gamma_summary.json
    figures/phase2_2_component_gamma.png
"""

import sys
import json
import logging
import importlib.util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent.resolve()
OUTPUT_DIR = REPO_ROOT / "output" / "phase2_2"
FIGURES_DIR = REPO_ROOT / "figures"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("phase2.2")


# ===================================================================
# Import Phase 2.1 as a module
# ===================================================================

def import_phase21():
    """Import phase2_1_abr_pipeline.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "phase2_1",
        REPO_ROOT / "src" / "phase2_1_abr_pipeline.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===================================================================
# Per-step per-topology σ² — using Phase 2.1's edge field dicts
# ===================================================================

def decompose_gamma_per_step(r_spatial, r_comp, r_temporal,
                              b_spatial, b_comp, b_temporal,
                              n_steps):
    """
    Compute per-step Γ decomposed by edge type.

    Uses the same σ² method as Phase 2.1: flatten all values for
    that edge type at that step, take np.var.

    Returns arrays of shape (n_steps,).
    """
    gamma_spatial = np.zeros(n_steps)
    gamma_comp = np.zeros(n_steps)
    gamma_temporal = np.zeros(n_steps)

    sigma_r_spatial = np.zeros(n_steps)
    sigma_r_comp = np.zeros(n_steps)
    sigma_r_temporal = np.zeros(n_steps)
    sigma_ba_spatial = np.zeros(n_steps)
    sigma_ba_comp = np.zeros(n_steps)
    sigma_ba_temporal = np.zeros(n_steps)

    for step in range(n_steps):
        # --- Spatial edges at this step ---
        r_s_vals = []
        ba_s_vals = []
        for (s1, s2, st), ev in r_spatial.items():
            if st == step:
                r_s_vals.extend(ev.tolist())
        for (s1, s2, st), ev in b_spatial.items():
            if st == step:
                ba_s_vals.extend(ev.tolist())

        if r_s_vals:
            sigma_r_spatial[step] = float(np.var(r_s_vals))
        if ba_s_vals:
            sigma_ba_spatial[step] = float(np.var(ba_s_vals))

        # --- Component edges at this step ---
        r_c_vals = []
        ba_c_vals = []
        for (s_idx, st, p_idx), val in r_comp.items():
            if st == step:
                r_c_vals.append(val)
        for (s_idx, st, p_idx), val in b_comp.items():
            if st == step:
                ba_c_vals.append(val)

        if r_c_vals:
            sigma_r_comp[step] = float(np.var(r_c_vals))
        if ba_c_vals:
            sigma_ba_comp[step] = float(np.var(ba_c_vals))

        # --- Temporal edges at this step ---
        r_t_vals = []
        ba_t_vals = []
        for (s_idx, st), ev in r_temporal.items():
            if st == step:
                r_t_vals.extend(ev.tolist())
        for (s_idx, st), ev in b_temporal.items():
            if st == step:
                ba_t_vals.extend(ev.tolist())

        if r_t_vals:
            sigma_r_temporal[step] = float(np.var(r_t_vals))
        if ba_t_vals:
            sigma_ba_temporal[step] = float(np.var(ba_t_vals))

        # --- Γ per type ---
        gamma_spatial[step] = sigma_r_spatial[step] - sigma_ba_spatial[step]
        gamma_comp[step] = sigma_r_comp[step] - sigma_ba_comp[step]
        gamma_temporal[step] = sigma_r_temporal[step] - sigma_ba_temporal[step]

        if (step + 1) % 20 == 0:
            log.info(f"  Step {step + 1}/{n_steps}: "
                     f"Γ_s={gamma_spatial[step]:.1f} "
                     f"Γ_c={gamma_comp[step]:.1f} "
                     f"Γ_t={gamma_temporal[step]:.1f}")

    return {
        "gamma_spatial": gamma_spatial,
        "gamma_comp": gamma_comp,
        "gamma_temporal": gamma_temporal,
        "sigma_r_spatial": sigma_r_spatial,
        "sigma_r_comp": sigma_r_comp,
        "sigma_r_temporal": sigma_r_temporal,
        "sigma_ba_spatial": sigma_ba_spatial,
        "sigma_ba_comp": sigma_ba_comp,
        "sigma_ba_temporal": sigma_ba_temporal,
    }


# ===================================================================
# MAIN
# ===================================================================

def run_component_gamma(csv_path=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Import and run Phase 2.1's pipeline to get the edge fields ---
    p21 = import_phase21()

    if csv_path is None:
        csv_path = REPO_ROOT / "data" / "supermag.csv"

    log.info("Running Phase 2.1 pipeline to obtain edge fields...")

    # Load data
    df, station_meta = p21.load_supermag(csv_path)

    # Topology
    spatial_edges, stations, edge_dists, graph_stats = \
        p21.declare_proximity_topology(station_meta, p21.MAX_EDGE_KM)
    n_stations = len(stations)
    station_to_idx = {s: i for i, s in enumerate(stations)}
    adjacency = p21.build_adjacency(spatial_edges, n_stations)

    # Storm window
    timestamps = sorted(df["Date_UTC"].unique())
    peak_idx = p21.find_storm_peak(df, timestamps)
    log.info(f"Storm peak at idx {peak_idx}: {timestamps[peak_idx]}")

    field, n_steps, window_ts, active_per_step = p21.build_windowed_field(
        df, station_to_idx, timestamps, peak_idx, p21.WINDOW_MINUTES
    )

    # A
    log.info("Running operator A...")
    a_spatial, a_comp, a_temporal = p21.operator_a(
        field, spatial_edges, p21.COMP_PAIRS, n_steps, active_per_step
    )

    # ρ
    log.info("Computing ρ...")
    rho = p21.compute_rho(
        a_spatial, a_comp, a_temporal, p21.RHO_BASE,
        n_steps, active_per_step
    )

    # B
    log.info("Running operator B...")
    b_spatial, b_comp, b_temporal = p21.operator_b(
        a_spatial, a_comp, a_temporal, adjacency,
        p21.COMP_PAIRS, n_steps, active_per_step
    )

    # R
    log.info("Running operator R...")
    r_spatial, r_comp, r_temporal = p21.operator_r(
        b_spatial, b_comp, b_temporal, rho, adjacency,
        p21.COMP_PAIRS, n_steps, active_per_step
    )

    # --- Phase 2.2: decompose per-step Γ by topology ---
    log.info("Decomposing Γ per step per topology...")
    results = decompose_gamma_per_step(
        r_spatial, r_comp, r_temporal,
        b_spatial, b_comp, b_temporal,
        n_steps,
    )

    # --- Cross-check: global totals should match Phase 2.1 ---
    # Phase 2.1 computes global σ² across ALL steps at once.
    # Sum of per-step Γ is NOT the same as global Γ (variance of
    # the union ≠ sum of per-step variances). So we also compute
    # global totals using Phase 2.1's method for comparison.
    global_gamma_spatial = (p21.sigma_sq_from_dict_vec(r_spatial)
                            - p21.sigma_sq_from_dict_vec(b_spatial))
    global_gamma_comp = (p21.sigma_sq_from_dict_scalar(r_comp)
                         - p21.sigma_sq_from_dict_scalar(b_comp))
    global_gamma_temporal = (p21.sigma_sq_from_dict_vec(r_temporal)
                             - p21.sigma_sq_from_dict_vec(b_temporal))
    global_gamma_total = global_gamma_spatial + global_gamma_comp + global_gamma_temporal

    # Load Phase 2.1 saved results
    p21_path = REPO_ROOT / "output" / "phase2_1" / "abr_results.npz"
    p21_saved = np.load(p21_path, allow_pickle=True) if p21_path.exists() else None

    log.info(f"\nGlobal Γ (recomputed, should match Phase 2.1):")
    log.info(f"  Γ_spatial:  {global_gamma_spatial:.6f}")
    log.info(f"  Γ_comp:     {global_gamma_comp:.6f}")
    log.info(f"  Γ_temporal: {global_gamma_temporal:.6f}")
    log.info(f"  Γ_total:    {global_gamma_total:.6f}")

    if p21_saved is not None:
        log.info(f"\nPhase 2.1 saved values:")
        log.info(f"  Γ_spatial:  {float(p21_saved['gamma_spatial']):.6f}")
        log.info(f"  Γ_comp:     {float(p21_saved['gamma_comp']):.6f}")
        log.info(f"  Γ_temporal: {float(p21_saved['gamma_temporal']):.6f}")
        log.info(f"  Γ_total:    {float(p21_saved['gamma_total']):.6f}")

        # Verify match
        tol = abs(float(p21_saved['gamma_total'])) * 0.001
        diff = abs(global_gamma_total - float(p21_saved['gamma_total']))
        if diff > tol:
            log.error(f"  MISMATCH: recomputed total differs by {diff:.2f}")
        else:
            log.info(f"  ✓ Global totals match Phase 2.1 (diff={diff:.2f})")

    # --- Per-step summary ---
    gamma_s = results["gamma_spatial"]
    gamma_c = results["gamma_comp"]
    gamma_t = results["gamma_temporal"]
    gamma_tot = gamma_s + gamma_c + gamma_t

    # Per-step fractions (using absolute values to handle negative Γ_spatial)
    total_abs_sum = np.sum(np.abs(gamma_s)) + np.sum(np.abs(gamma_c)) + np.sum(np.abs(gamma_t))
    if total_abs_sum > 0:
        frac_spatial = np.sum(np.abs(gamma_s)) / total_abs_sum
        frac_comp = np.sum(np.abs(gamma_c)) / total_abs_sum
        frac_temporal = np.sum(np.abs(gamma_t)) / total_abs_sum
    else:
        frac_spatial = frac_comp = frac_temporal = 0.0

    # Global fractions (Phase 2.1's method)
    global_abs_sum = abs(global_gamma_spatial) + abs(global_gamma_comp) + abs(global_gamma_temporal)
    if global_abs_sum > 0:
        global_frac_spatial = abs(global_gamma_spatial) / global_abs_sum
        global_frac_comp = abs(global_gamma_comp) / global_abs_sum
        global_frac_temporal = abs(global_gamma_temporal) / global_abs_sum
    else:
        global_frac_spatial = global_frac_comp = global_frac_temporal = 0.0

    peak_step_s = int(np.argmax(np.abs(gamma_s)))
    peak_step_c = int(np.argmax(np.abs(gamma_c)))
    peak_step_t = int(np.argmax(np.abs(gamma_t)))
    peak_step_tot = int(np.argmax(np.abs(gamma_tot)))

    log.info(f"\nPer-step decomposition:")
    log.info(f"  Γ_spatial:  peak={gamma_s[peak_step_s]:.1f} at step {peak_step_s}, "
             f"|fraction|={frac_spatial:.3f}")
    log.info(f"  Γ_comp:     peak={gamma_c[peak_step_c]:.1f} at step {peak_step_c}, "
             f"|fraction|={frac_comp:.3f}")
    log.info(f"  Γ_temporal: peak={gamma_t[peak_step_t]:.1f} at step {peak_step_t}, "
             f"|fraction|={frac_temporal:.3f}")

    log.info(f"\nGlobal fractions (Phase 2.1 method):")
    log.info(f"  Spatial:  {global_frac_spatial:.3f}")
    log.info(f"  Component: {global_frac_comp:.3f}")
    log.info(f"  Temporal: {global_frac_temporal:.3f}")

    # --- Save .npz ---
    ts_labels = np.array([str(t) for t in window_ts])

    np.savez(
        OUTPUT_DIR / "component_gamma.npz",
        # Per-step decomposition
        gamma_spatial=gamma_s,
        gamma_comp=gamma_c,
        gamma_temporal=gamma_t,
        gamma_total_per_step=gamma_tot,
        sigma_r_spatial=results["sigma_r_spatial"],
        sigma_r_comp=results["sigma_r_comp"],
        sigma_r_temporal=results["sigma_r_temporal"],
        sigma_ba_spatial=results["sigma_ba_spatial"],
        sigma_ba_comp=results["sigma_ba_comp"],
        sigma_ba_temporal=results["sigma_ba_temporal"],
        timestamps=ts_labels,
        # Global decomposition (Phase 2.1 method)
        global_gamma_spatial=global_gamma_spatial,
        global_gamma_comp=global_gamma_comp,
        global_gamma_temporal=global_gamma_temporal,
        global_gamma_total=global_gamma_total,
        # Fractions
        per_step_frac_spatial=frac_spatial,
        per_step_frac_comp=frac_comp,
        per_step_frac_temporal=frac_temporal,
        global_frac_spatial=global_frac_spatial,
        global_frac_comp=global_frac_comp,
        global_frac_temporal=global_frac_temporal,
    )
    log.info(f"Saved: {OUTPUT_DIR / 'component_gamma.npz'}")

    # --- Save JSON summary ---
    summary = {
        "phase": "2.2",
        "description": "Component-resolved Γ — per-step and global decomposition",
        "parameters": {
            "max_edge_km": p21.MAX_EDGE_KM,
            "rho_base": p21.RHO_BASE,
            "window_minutes": p21.WINDOW_MINUTES,
            "n_steps": n_steps,
            "n_spatial_edges": len(spatial_edges),
        },
        "global_decomposition": {
            "note": "σ² computed across ALL steps at once (Phase 2.1 method)",
            "gamma_spatial": float(global_gamma_spatial),
            "gamma_comp": float(global_gamma_comp),
            "gamma_temporal": float(global_gamma_temporal),
            "gamma_total": float(global_gamma_total),
            "frac_spatial": float(global_frac_spatial),
            "frac_comp": float(global_frac_comp),
            "frac_temporal": float(global_frac_temporal),
        },
        "per_step_decomposition": {
            "note": "σ² computed per timestep, then differenced",
            "frac_spatial": float(frac_spatial),
            "frac_comp": float(frac_comp),
            "frac_temporal": float(frac_temporal),
            "peaks": {
                "spatial": {"step": peak_step_s, "value": float(gamma_s[peak_step_s])},
                "component": {"step": peak_step_c, "value": float(gamma_c[peak_step_c])},
                "temporal": {"step": peak_step_t, "value": float(gamma_t[peak_step_t])},
                "total": {"step": peak_step_tot, "value": float(gamma_tot[peak_step_tot])},
            },
        },
        "declaration": {
            "two_projections": (
                "Global and per-step decompositions are different declared "
                "projections of the same edge field. Global: σ² across all "
                "steps at once — one number per edge type. Per-step: σ² at "
                "each timestep — a time series per edge type. Variance of "
                "the union ≠ sum of per-step variances. Both are valid; "
                "they answer different questions."
            ),
            "sigma_sq_per_type": (
                "σ² computed separately for spatial, component, and temporal "
                "edge types. Preserved: variance within each topology. "
                "Discarded: cross-topology covariance, per-edge detail."
            ),
            "preprocessing": (
                "No interpolation, gridding, spectral projection, or smoothing "
                "applied prior to operator A. Operators imported from Phase 2.1."
            ),
        },
    }

    with open(OUTPUT_DIR / "component_gamma_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved: {OUTPUT_DIR / 'component_gamma_summary.json'}")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        t = np.arange(n_steps)
        storm_peak_step = peak_idx - max(0, peak_idx - p21.WINDOW_MINUTES // 2)

        # Panel 1: Per-step Γ by topology
        ax = axes[0]
        ax.plot(t, gamma_s, color="steelblue", linewidth=1.5, label="Γ_spatial")
        ax.plot(t, gamma_c, color="firebrick", linewidth=1.5, label="Γ_comp")
        ax.plot(t, gamma_t, color="goldenrod", linewidth=1.5, label="Γ_temporal")
        ax.axvline(storm_peak_step, color="gray", linestyle="--", alpha=0.5,
                    label="Storm peak")
        ax.set_ylabel("Γ per step")
        ax.set_title("Phase 2.2 — Per-Step Γ Decomposition by Topology")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Panel 2: Fractional contribution (stacked)
        ax = axes[1]
        abs_sum = np.abs(gamma_s) + np.abs(gamma_c) + np.abs(gamma_t)
        abs_sum = np.where(abs_sum > 0, abs_sum, 1.0)
        f_s = np.abs(gamma_s) / abs_sum
        f_c = np.abs(gamma_c) / abs_sum
        f_t = np.abs(gamma_t) / abs_sum
        ax.fill_between(t, 0, f_s, color="steelblue", alpha=0.6, label="Spatial")
        ax.fill_between(t, f_s, f_s + f_c, color="firebrick", alpha=0.6, label="Component")
        ax.fill_between(t, f_s + f_c, 1.0, color="goldenrod", alpha=0.6, label="Temporal")
        ax.axvline(storm_peak_step, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("Fraction of |Γ|")
        ax.set_ylim(0, 1)
        ax.legend(loc="right")
        ax.grid(True, alpha=0.3)

        # Panel 3: Global fractions as annotation + total Γ per step
        ax = axes[2]
        ax.plot(t, gamma_tot, color="black", linewidth=1.5, label="Γ_total per step")
        ax.axvline(storm_peak_step, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Timestep (minutes from window start)")
        ax.set_ylabel("Γ total per step")
        ax.set_title(
            f"Global Fractions (Phase 2.1): "
            f"Spatial {global_frac_spatial:.1%}, "
            f"Component {global_frac_comp:.1%}, "
            f"Temporal {global_frac_temporal:.1%}"
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = FIGURES_DIR / "phase2_2_component_gamma.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        log.info(f"Saved: {plot_path}")

    except ImportError:
        log.warning("matplotlib not available — plot skipped.")

    return results, summary


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    results, summary = run_component_gamma(csv_path)
    log.info("Phase 2.2 complete.")
