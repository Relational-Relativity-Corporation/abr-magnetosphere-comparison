"""
Phase 2.1: ABR Pipeline — 3-Topology Edge Field (Spatial × Component × Temporal)
Metatron Dynamics, Inc. — ABR vs Traditional Pipeline Comparison

Processes raw SuperMAG station data as a single 3-topology edge field.
The field has spatial edges (station-to-station gradients), component edges
(N-E, N-Z, E-Z at each station), and temporal edges (evolution step
differences at each station). ABR operators act on all three edge types.

Temporal evolution is strictly one-directional. No backward temporal edges.
No temporal ring. The system evolves forward. Step t produces step t+1.

R cross-couples BETWEEN topologies, not within them. Spatial edge values
couple into component edges. Component edge values couple into temporal
edges. Temporal edge values couple into spatial edges. The antisymmetry
is across topologies — that's circulation. The ring's forward-backward
subtraction was a special case where cross-axis coupling had to be expressed
within a single axis because no other axis existed.

DECLARATION — M_ABR:
  Observable space O: SuperMAG station measurements (N, E, Z) in nT,
    geographic coordinates (GEOLAT, GEOLON), 1-minute cadence.
  Domain D: stations × components × evolution steps, all finite, all nT.
    D does not include inter-station space or unmeasured time.
  Spatial topology: proximity graph, stations within MAX_EDGE_KM.
  Component topology: all-pairs on {N, E, Z}.
  Temporal topology: directed line. Step t adjacent to step t+1 only.
    No backward edges. Evolution is one-directional.
  Units: nT as reported. No scaling, normalization, or baseline removal.
  Pre-A transformation: none.
  Window: 2-hour window (120 steps) centered on storm peak.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output/phase2_1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMPONENTS = ["N", "E", "Z"]
K = 3
RHO_BASE = 0.3
COMP_PAIRS = [(0, 1), (0, 2), (1, 2)]  # (N,E), (N,Z), (E,Z)
MAX_EDGE_KM = 1000.0
WINDOW_MINUTES = 120  # 2-hour window
EARTH_RADIUS_KM = 6371.0


# ===================================================================
# 1. DATA LOADING
# ===================================================================

def load_supermag(csv_path):
    log.info(f"Loading SuperMAG data from {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["Date_UTC"])

    required = {"Date_UTC", "IAGA", "GEOLON", "GEOLAT"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    col_map = {}
    if "N" in df.columns:
        col_map = {"N": "N", "E": "E", "Z": "Z"}
    elif "dbn_nez" in df.columns:
        col_map = {"dbn_nez": "N", "dbe_nez": "E", "dbz_nez": "Z"}
    else:
        raise ValueError(f"Cannot find component columns. Got: {list(df.columns)}")
    df = df.rename(columns=col_map)
    log.info(f"Component columns mapped: {col_map}")

    n_before = len(df)
    df = df.dropna(subset=["N", "E", "Z"])
    n_after = len(df)
    if n_before != n_after:
        log.info(f"Dropped {n_before - n_after} rows with NaN (not in D)")

    for comp in COMPONENTS:
        if not np.all(np.isfinite(df[comp].values)):
            raise ValueError(f"Non-finite values in {comp} — not in D")

    df = df.sort_values("Date_UTC").reset_index(drop=True)

    station_meta = df.groupby("IAGA").agg(
        lat=("GEOLAT", "first"),
        lon=("GEOLON", "first")
    )

    log.info(f"Loaded {len(df)} measurements, {len(station_meta)} stations")
    log.info(f"Time range: {df['Date_UTC'].min()} — {df['Date_UTC'].max()}")

    return df, station_meta


# ===================================================================
# 2. SPATIAL TOPOLOGY — PROXIMITY GRAPH
# ===================================================================

def haversine_km(lat1, lon1, lat2, lon2):
    la1, lo1 = np.radians(lat1), np.radians(lon1)
    la2, lo2 = np.radians(lat2), np.radians(lon2)
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = (np.sin(dlat / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2)
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def declare_proximity_topology(station_meta, max_km):
    stations = list(station_meta.index)
    n = len(stations)
    lats = station_meta["lat"].values
    lons = station_meta["lon"].values

    log.info(f"Declaring proximity topology: {n} stations, "
             f"threshold {max_km:.0f} km")

    edges = []
    edge_distances = []
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(lats[i], lons[i], lats[j], lons[j])
            if d <= max_km:
                edges.append((i, j))
                edge_distances.append(d)

    degree = np.zeros(n, dtype=int)
    for (i, j) in edges:
        degree[i] += 1
        degree[j] += 1

    graph_stats = {
        "n_stations": n,
        "n_edges": len(edges),
        "max_edge_km": max_km,
        "mean_degree": float(np.mean(degree)),
        "isolated_stations": int(np.sum(degree == 0)),
        "edge_distance_km": {
            "mean": float(np.mean(edge_distances)) if edge_distances else 0,
            "max": float(np.max(edge_distances)) if edge_distances else 0,
        },
    }

    log.info(f"  Edges: {len(edges)}, mean degree: {graph_stats['mean_degree']:.1f}, "
             f"isolated: {graph_stats['isolated_stations']}")

    return edges, stations, edge_distances, graph_stats


def build_adjacency(edges, n_stations):
    adj = defaultdict(set)
    for (i, j) in edges:
        adj[i].add(j)
        adj[j].add(i)
    return dict(adj)


# ===================================================================
# 3. BUILD 3-TOPOLOGY NODE FIELD AND EXTRACT WINDOW
# ===================================================================

def build_windowed_field(df, station_to_idx, timestamps, peak_idx,
                         window_minutes):
    """
    Build the node field for the analysis window.

    Returns
    -------
    field : dict mapping (station_idx, step) → np.array([N, E, Z])
        step is 0-indexed within the window.
    n_steps : int
    window_timestamps : list of timestamps in the window
    active_stations_per_step : list of sets of station indices
    """
    half = window_minutes // 2
    start = max(0, peak_idx - half)
    end = min(len(timestamps), peak_idx + half)
    window_ts = timestamps[start:end]
    n_steps = len(window_ts)

    log.info(f"Window: {window_ts[0]} to {window_ts[-1]} ({n_steps} steps)")
    log.info(f"  Peak at window step {peak_idx - start}")

    # Filter to window timestamps
    window_set = set(window_ts)
    wdf = df[df["Date_UTC"].isin(window_set)].copy()
    wdf["step"] = wdf["Date_UTC"].map({t: i for i, t in enumerate(window_ts)})
    wdf["sidx"] = wdf["IAGA"].map(station_to_idx)
    wdf = wdf.dropna(subset=["sidx"])
    wdf["sidx"] = wdf["sidx"].astype(int)

    field = {}
    active_per_step = [set() for _ in range(n_steps)]

    for row in wdf.itertuples(index=False):
        s_idx = row.sidx
        step = int(row.step)
        field[(s_idx, step)] = np.array([row.N, row.E, row.Z], dtype=np.float64)
        active_per_step[step].add(s_idx)

    log.info(f"  Node field entries: {len(field)}")

    return field, n_steps, window_ts, active_per_step


# ===================================================================
# 4. OPERATOR A — 3-TOPOLOGY EDGE EXTRACTION
#
# A extracts three kinds of edges simultaneously:
#   Spatial: x[s1, t] - x[s2, t] for adjacent stations at same step
#   Component: x[s, t][c1] - x[s, t][c2] at each station/step
#   Temporal: x[s, t+1] - x[s, t] at each station (forward only)
#
# No backward temporal edges. Evolution is one-directional.
# ===================================================================

def operator_a(field, spatial_edges, comp_pairs, n_steps,
               active_per_step):
    """
    A: NodeField → 3-topology EdgeField.

    Returns
    -------
    spatial_ef : dict (s1, s2, step) → np.array(3,)
        Directed spatial edge. Both directions: (s1,s2) and (s2,s1).
    comp_ef : dict (station, step, pair_idx) → float
        Component edge.
    temporal_ef : dict (station, step) → np.array(3,)
        Forward temporal edge: field[s, step+1] - field[s, step].
        step ranges 0..n_steps-2.
    """
    spatial_ef = {}
    comp_ef = {}
    temporal_ef = {}

    for step in range(n_steps):
        active = active_per_step[step]

        # Spatial edges at this step
        for (s1, s2) in spatial_edges:
            if s1 in active and s2 in active:
                v1 = field[(s1, step)]
                v2 = field[(s2, step)]
                spatial_ef[(s1, s2, step)] = v1 - v2
                spatial_ef[(s2, s1, step)] = v2 - v1

        # Component edges at this step
        for s_idx in active:
            v = field[(s_idx, step)]
            for p_idx, (c1, c2) in enumerate(comp_pairs):
                comp_ef[(s_idx, step, p_idx)] = v[c1] - v[c2]

        # Temporal edges: forward only, step → step+1
        if step < n_steps - 1:
            next_active = active_per_step[step + 1]
            for s_idx in active:
                if s_idx in next_active:
                    temporal_ef[(s_idx, step)] = (
                        field[(s_idx, step + 1)] - field[(s_idx, step)]
                    )

    log.info(f"  A: {len(spatial_ef)} spatial, {len(comp_ef)} component, "
             f"{len(temporal_ef)} temporal edges")

    return spatial_ef, comp_ef, temporal_ef


# ===================================================================
# 5. COMPUTE ρ FROM A OUTPUT
#
# Per station-step: max absolute edge value incident to that
# station at that step, across all three topologies.
# ===================================================================

def compute_rho(spatial_ef, comp_ef, temporal_ef, rho_base,
                n_steps, active_per_step):
    rho = {}

    for step in range(n_steps):
        for s_idx in active_per_step[step]:
            rho[(s_idx, step)] = 0.0

    for (s1, s2, step), ev in spatial_ef.items():
        mag = float(np.max(np.abs(ev)))
        key = (s1, step)
        rho[key] = max(rho.get(key, 0.0), mag)

    for (s_idx, step, p_idx), val in comp_ef.items():
        key = (s_idx, step)
        rho[key] = max(rho.get(key, 0.0), abs(val))

    for (s_idx, step), ev in temporal_ef.items():
        mag = float(np.max(np.abs(ev)))
        key = (s_idx, step)
        rho[key] = max(rho.get(key, 0.0), mag)
        # Temporal edge also incident to (s_idx, step+1)
        key_next = (s_idx, step + 1)
        rho[key_next] = max(rho.get(key_next, 0.0), mag)

    for key in rho:
        m = rho[key]
        rho[key] = rho_base * m / (1.0 + m)

    return rho


# ===================================================================
# 6. OPERATOR B — 3-TOPOLOGY ACCUMULATION
#
# Each edge accumulates with edges sharing its forward vertex,
# within the same topology.
#
# Spatial: edge (s1→s2, step) accumulates with edges leaving s2
#   at the same step.
# Component: comp edge at (s, step) accumulates with comp edges
#   at adjacent stations at the same step.
# Temporal: edge (s, step→step+1) accumulates with (s, step+1→step+2)
#   if it exists. Strictly forward. One temporal neighbor.
# ===================================================================

def operator_b(spatial_ef, comp_ef, temporal_ef, adjacency,
               comp_pairs, n_steps, active_per_step):

    b_spatial = {}
    for (s1, s2, step), ev in spatial_ef.items():
        acc = ev.copy()
        for nb in adjacency.get(s2, set()):
            if nb != s1 and (s2, nb, step) in spatial_ef:
                acc = acc + spatial_ef[(s2, nb, step)]
        b_spatial[(s1, s2, step)] = acc

    b_comp = {}
    for (s_idx, step, p_idx), val in comp_ef.items():
        acc = val
        for nb in adjacency.get(s_idx, set()):
            key_nb = (nb, step, p_idx)
            if key_nb in comp_ef:
                acc += comp_ef[key_nb]
        b_comp[(s_idx, step, p_idx)] = acc

    b_temporal = {}
    for (s_idx, step), ev in temporal_ef.items():
        acc = ev.copy()
        # Forward temporal neighbor: (s_idx, step+1) → (s_idx, step+2)
        next_key = (s_idx, step + 1)
        if next_key in temporal_ef:
            acc = acc + temporal_ef[next_key]
        b_temporal[(s_idx, step)] = acc

    log.info(f"  B: {len(b_spatial)} spatial, {len(b_comp)} component, "
             f"{len(b_temporal)} temporal edges")

    return b_spatial, b_comp, b_temporal


# ===================================================================
# 7. OPERATOR R — CROSS-TOPOLOGY CIRCULATION
#
# R cross-couples BETWEEN topologies. The antisymmetry is across
# edge types, not within a single topology.
#
# Spatial edges receive temporal edge values:
#   For spatial edge (s1→s2, step), each component c:
#     R += ρ × (temporal_edge at s2 minus temporal_edge at s1)
#   The asymmetry is between what's evolving at s2 vs s1.
#
# Component edges receive spatial edge values:
#   For comp edge (c1-c2) at (s, step):
#     R += ρ × mean(spatial_edge[c1] - spatial_edge[c2])
#       over spatial edges incident to s at this step.
#
# Temporal edges receive component edge values:
#   For temporal edge at (s, step), each component c:
#     R += ρ × (comp asymmetry at step+1 minus comp asymmetry at step)
#   The asymmetry is between component structure at the two ends
#   of the temporal edge.
#
# Each coupling is between two different topologies. No coupling
# within a topology. No forward-backward along a single axis.
# ===================================================================

def operator_r(b_spatial, b_comp, b_temporal, rho, adjacency,
               comp_pairs, n_steps, active_per_step):

    r_spatial = {}
    r_comp = {}
    r_temporal = {}

    # --- Spatial edges receive temporal coupling ---
    for (s1, s2, step), bv in b_spatial.items():
        rv = bv.copy()
        rh = rho.get((s1, step), 0.0)

        if rh > 0.0:
            # Temporal edge at s2 vs temporal edge at s1
            t_s2 = b_temporal.get((s2, step))
            t_s1 = b_temporal.get((s1, step))

            if t_s2 is not None and t_s1 is not None:
                temporal_asym = t_s2 - t_s1
                rv = rv + rh * temporal_asym
            elif t_s2 is not None:
                rv = rv + rh * t_s2
            elif t_s1 is not None:
                rv = rv - rh * t_s1

        r_spatial[(s1, s2, step)] = rv

    # --- Component edges receive spatial coupling ---
    for (s_idx, step, p_idx), bv in b_comp.items():
        rv = bv
        rh = rho.get((s_idx, step), 0.0)

        if rh > 0.0:
            c1, c2 = comp_pairs[p_idx]
            asym_vals = []
            for nb in adjacency.get(s_idx, set()):
                key = (s_idx, nb, step)
                if key in b_spatial:
                    ev = b_spatial[key]
                    asym_vals.append(ev[c1] - ev[c2])

            if asym_vals:
                spatial_asym = np.mean(asym_vals)
                rv = rv + rh * spatial_asym

        r_comp[(s_idx, step, p_idx)] = rv

    # --- Temporal edges receive component coupling ---
    for (s_idx, step), bv in b_temporal.items():
        rv = bv.copy()
        rh = rho.get((s_idx, step), 0.0)

        if rh > 0.0:
            # Component asymmetry at step+1 vs step
            for p_idx, (c1, c2) in enumerate(comp_pairs):
                comp_next = b_comp.get((s_idx, step + 1, p_idx))
                comp_curr = b_comp.get((s_idx, step, p_idx))

                if comp_next is not None and comp_curr is not None:
                    comp_asym = comp_next - comp_curr
                    # Antisymmetric: +ρ for c1, -ρ for c2
                    rv[c1] += rh * comp_asym
                    rv[c2] -= rh * comp_asym

        r_temporal[(s_idx, step)] = rv

    log.info(f"  R: {len(r_spatial)} spatial, {len(r_comp)} component, "
             f"{len(r_temporal)} temporal edges")

    return r_spatial, r_comp, r_temporal


# ===================================================================
# 8. DIAGNOSTICS — DECLARED PROJECTIONS
# ===================================================================

def sigma_sq_from_dict_vec(ef):
    """σ² over all vector-valued edges (spatial or temporal)."""
    vals = []
    for ev in ef.values():
        vals.extend(ev.tolist())
    if not vals:
        return 0.0
    return float(np.var(vals))


def sigma_sq_from_dict_scalar(ef):
    """σ² over all scalar-valued edges (component)."""
    vals = list(ef.values())
    if not vals:
        return 0.0
    return float(np.var(vals))


def compute_diagnostics(r_spatial, r_comp, r_temporal,
                        b_spatial, b_comp, b_temporal,
                        n_steps, active_per_step, adjacency):
    """
    Compute Γ (total and per-topology) and per-station-step σ².

    All σ² values are DECLARED PROJECTIONS.
    Preserved: total relational variance within each topology.
    Discarded: per-edge detail, directional structure, sign.
    """
    # σ² for full ABR output
    s2_e_spatial = sigma_sq_from_dict_vec(r_spatial)
    s2_e_comp = sigma_sq_from_dict_scalar(r_comp)
    s2_e_temporal = sigma_sq_from_dict_vec(r_temporal)
    s2_e_total = s2_e_spatial + s2_e_comp + s2_e_temporal

    # σ² for B∘A output (without R)
    s2_ba_spatial = sigma_sq_from_dict_vec(b_spatial)
    s2_ba_comp = sigma_sq_from_dict_scalar(b_comp)
    s2_ba_temporal = sigma_sq_from_dict_vec(b_temporal)
    s2_ba_total = s2_ba_spatial + s2_ba_comp + s2_ba_temporal

    # Γ per topology
    gamma_spatial = s2_e_spatial - s2_ba_spatial
    gamma_comp = s2_e_comp - s2_ba_comp
    gamma_temporal = s2_e_temporal - s2_ba_temporal
    gamma_total = s2_e_total - s2_ba_total

    # Per-step Γ: compute σ² restricted to each step
    gamma_by_step = np.zeros(n_steps)
    sigma_e_by_step = np.zeros(n_steps)
    sigma_ba_by_step = np.zeros(n_steps)

    for step in range(n_steps):
        # Collect R output edges at this step
        r_vals = []
        ba_vals = []

        for (s1, s2, st), ev in r_spatial.items():
            if st == step:
                r_vals.extend(ev.tolist())
        for (s1, s2, st), ev in b_spatial.items():
            if st == step:
                ba_vals.extend(ev.tolist())

        for (s_idx, st, p_idx), val in r_comp.items():
            if st == step:
                r_vals.append(val)
        for (s_idx, st, p_idx), val in b_comp.items():
            if st == step:
                ba_vals.append(val)

        # Temporal edges at this step (step → step+1)
        for (s_idx, st), ev in r_temporal.items():
            if st == step:
                r_vals.extend(ev.tolist())
        for (s_idx, st), ev in b_temporal.items():
            if st == step:
                ba_vals.extend(ev.tolist())

        if r_vals:
            sigma_e_by_step[step] = float(np.var(r_vals))
        if ba_vals:
            sigma_ba_by_step[step] = float(np.var(ba_vals))
        gamma_by_step[step] = sigma_e_by_step[step] - sigma_ba_by_step[step]

    return {
        "gamma_total": gamma_total,
        "gamma_spatial": gamma_spatial,
        "gamma_comp": gamma_comp,
        "gamma_temporal": gamma_temporal,
        "sigma_e_total": s2_e_total,
        "sigma_ba_total": s2_ba_total,
        "gamma_by_step": gamma_by_step,
        "sigma_e_by_step": sigma_e_by_step,
        "sigma_ba_by_step": sigma_ba_by_step,
    }


# ===================================================================
# 9. FIND STORM PEAK (quick scan — max variance in raw data)
# ===================================================================

def find_storm_peak(df, timestamps):
    """Find the timestep with maximum total perturbation magnitude."""
    df["_mag2"] = df["N"]**2 + df["E"]**2 + df["Z"]**2
    mag_per_ts = df.groupby("Date_UTC")["_mag2"].sum()
    peak_ts = mag_per_ts.idxmax()
    df.drop(columns=["_mag2"], inplace=True)
    peak_idx = list(timestamps).index(peak_ts)
    return peak_idx


# ===================================================================
# 10. MAIN PIPELINE
# ===================================================================

def run_abr_pipeline(csv_path, max_edge_km=MAX_EDGE_KM,
                     rho_base=RHO_BASE, window_minutes=WINDOW_MINUTES):

    # --- M : O → D ---
    df, station_meta = load_supermag(csv_path)

    # --- Declare spatial topology ---
    spatial_edges, stations, edge_dists, graph_stats = \
        declare_proximity_topology(station_meta, max_edge_km)
    n_stations = len(stations)
    station_to_idx = {s: i for i, s in enumerate(stations)}
    adjacency = build_adjacency(spatial_edges, n_stations)

    # --- Find storm peak and extract window ---
    timestamps = sorted(df["Date_UTC"].unique())
    log.info(f"Scanning {len(timestamps)} timesteps for storm peak...")
    peak_idx = find_storm_peak(df, timestamps)
    log.info(f"Storm peak at idx {peak_idx}: {timestamps[peak_idx]}")

    field, n_steps, window_ts, active_per_step = build_windowed_field(
        df, station_to_idx, timestamps, peak_idx, window_minutes
    )

    # --- Active edges per step ---
    # Filter spatial edges to those where both stations are present
    # (may vary by step due to station dropouts)

    # --- A: extract 3-topology edge field ---
    log.info("Running operator A...")
    a_spatial, a_comp, a_temporal = operator_a(
        field, spatial_edges, COMP_PAIRS, n_steps, active_per_step
    )

    # --- ρ from A ---
    log.info("Computing ρ...")
    rho = compute_rho(a_spatial, a_comp, a_temporal, rho_base,
                      n_steps, active_per_step)

    # --- B: accumulate ---
    log.info("Running operator B...")
    b_spatial, b_comp, b_temporal = operator_b(
        a_spatial, a_comp, a_temporal, adjacency,
        COMP_PAIRS, n_steps, active_per_step
    )

    # --- R: cross-topology circulation ---
    log.info("Running operator R...")
    r_spatial, r_comp, r_temporal = operator_r(
        b_spatial, b_comp, b_temporal, rho, adjacency,
        COMP_PAIRS, n_steps, active_per_step
    )

    # --- Diagnostics ---
    log.info("Computing diagnostics...")
    diag = compute_diagnostics(
        r_spatial, r_comp, r_temporal,
        b_spatial, b_comp, b_temporal,
        n_steps, active_per_step, adjacency
    )

    # --- Log results ---
    log.info(f"Γ total: {diag['gamma_total']:.6f}")
    log.info(f"  Γ spatial:  {diag['gamma_spatial']:.6f}")
    log.info(f"  Γ component: {diag['gamma_comp']:.6f}")
    log.info(f"  Γ temporal:  {diag['gamma_temporal']:.6f}")
    log.info(f"  σ²(E): {diag['sigma_e_total']:.6f}")
    log.info(f"  σ²(BA): {diag['sigma_ba_total']:.6f}")

    step_peak = int(np.argmax(diag["gamma_by_step"]))
    log.info(f"  Per-step Γ peak at step {step_peak}: "
             f"{diag['gamma_by_step'][step_peak]:.6f}")

    # --- Save ---
    ts_strings = np.array([str(t) for t in window_ts])

    np.savez_compressed(
        OUTPUT_DIR / "abr_results.npz",
        gamma_total=diag["gamma_total"],
        gamma_spatial=diag["gamma_spatial"],
        gamma_comp=diag["gamma_comp"],
        gamma_temporal=diag["gamma_temporal"],
        sigma_e_total=diag["sigma_e_total"],
        sigma_ba_total=diag["sigma_ba_total"],
        gamma_by_step=diag["gamma_by_step"],
        sigma_e_by_step=diag["sigma_e_by_step"],
        sigma_ba_by_step=diag["sigma_ba_by_step"],
        timestamps=ts_strings,
        station_lats=station_meta["lat"].values,
        station_lons=station_meta["lon"].values,
        station_codes=np.array(stations),
        n_stations=n_stations,
        n_steps=n_steps,
        n_spatial_edges=len(spatial_edges),
        max_edge_km=max_edge_km,
        rho_base=rho_base,
        window_minutes=window_minutes,
    )
    log.info(f"Saved: {OUTPUT_DIR / 'abr_results.npz'}")

    with open(OUTPUT_DIR / "station_graph.json", "w") as f:
        json.dump({
            "stations": stations,
            "n_stations": n_stations,
            "spatial_edges": spatial_edges,
            "n_edges": len(spatial_edges),
            "max_edge_km": max_edge_km,
            "comp_pairs": COMP_PAIRS,
            "comp_pair_labels": ["(N,E)", "(N,Z)", "(E,Z)"],
            "temporal_topology": "Directed line, forward only",
            "graph_stats": graph_stats,
            "declaration": (
                f"Spatial: proximity graph, {max_edge_km:.0f} km threshold. "
                "Component: all-pairs {N, E, Z}. "
                "Temporal: directed line, strictly forward. "
                "No backward temporal edges. "
                "R cross-couples between topologies, not within them."
            ),
        }, f, indent=2)

    summary = {
        "pipeline": "ABR V4, 3-topology (spatial × component × temporal)",
        "n_steps": n_steps,
        "window_minutes": window_minutes,
        "window_start": str(window_ts[0]),
        "window_end": str(window_ts[-1]),
        "n_stations": n_stations,
        "n_spatial_edges": len(spatial_edges),
        "max_edge_km": max_edge_km,
        "rho_base": rho_base,
        "graph_stats": graph_stats,
        "results": {
            "gamma_total": float(diag["gamma_total"]),
            "gamma_spatial": float(diag["gamma_spatial"]),
            "gamma_comp": float(diag["gamma_comp"]),
            "gamma_temporal": float(diag["gamma_temporal"]),
            "sigma_e": float(diag["sigma_e_total"]),
            "sigma_ba": float(diag["sigma_ba_total"]),
            "per_step_gamma_peak": {
                "step": step_peak,
                "timestamp": str(window_ts[step_peak]),
                "gamma": float(diag["gamma_by_step"][step_peak]),
            },
            "gamma_positive_steps": int(np.sum(diag["gamma_by_step"] > 0)),
        },
        "declaration": {
            "M": (
                "M_ABR: SuperMAG perturbation measurements (N, E, Z) in nT. "
                "No interpolation. No preprocessing before A. "
                "Domain D is the sensor network across a 2-hour evolution "
                "window. No claim about the magnetosphere between stations "
                "or outside the window. M_ABR adds no topological structure "
                "beyond what the sensor network and evolution sequence provide."
            ),
            "topologies": {
                "spatial": (
                    f"Proximity graph, {max_edge_km:.0f} km. "
                    "Edges are sensor-resolvable gradients."
                ),
                "component": "All-pairs on {N, E, Z}.",
                "temporal": (
                    "Directed line, strictly forward. "
                    "Evolution is one-directional. "
                    "Step t adjacent to step t+1 only."
                ),
            },
            "R": (
                "R cross-couples between topologies: "
                "spatial edges receive temporal asymmetry, "
                "component edges receive spatial asymmetry, "
                "temporal edges receive component asymmetry. "
                "No coupling within a single topology. "
                "The ring's forward-backward subtraction was a "
                "special case for single-axis systems."
            ),
            "gamma": (
                "Γ = σ²(R∘B∘A) - σ²(B∘A). Empirical measurement of "
                "cross-topology circulation on an irregular sensor graph. "
                "Theorems 5 and 6 (Object Error) are proved for the ring "
                "and do not apply to this topology. Γ > 0 indicates R "
                "detected antisymmetric cross-coupling in the edge field."
            ),
        },
    }

    with open(OUTPUT_DIR / "phase2_1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary")

    return diag, window_ts, summary


# ===================================================================
# Entry point
# ===================================================================
if __name__ == "__main__":
    import sys

    csv_path = (sys.argv[1] if len(sys.argv) > 1
                else DATA_DIR / "supermag.csv")

    diag, window_ts, summary = run_abr_pipeline(csv_path)

    log.info("\nPhase 2.1 complete.")
    log.info(f"  Window: {summary['window_start']} — {summary['window_end']}")
    log.info(f"  Γ total: {summary['results']['gamma_total']:.6f}")
    log.info(f"  Γ spatial: {summary['results']['gamma_spatial']:.6f}")
    log.info(f"  Γ component: {summary['results']['gamma_comp']:.6f}")
    log.info(f"  Γ temporal: {summary['results']['gamma_temporal']:.6f}")
    log.info(f"  Per-step Γ peak: step {summary['results']['per_step_gamma_peak']['step']} "
             f"({summary['results']['per_step_gamma_peak']['timestamp']})")
    log.info(f"  Γ > 0 steps: {summary['results']['gamma_positive_steps']}/{summary['n_steps']}")
