# ABR vs Traditional Pipeline — Magnetospheric Storm Analysis

**Metatron Dynamics, Inc.**
**Data:** SuperMAG St. Patrick's Day Storm, 2015-03-17
**Hardware:** ROBIN-1 (NVIDIA GTX 1050 Ti, CUDA 12.6)

---

## What This Repository Demonstrates

The traditional magnetospheric analysis pipeline applies a sequence of non-injective transformations — grid interpolation, spherical harmonic decomposition, source separation by degree, and index construction — before any analysis occurs. Each transformation preserves some measured relational structure while failing to preserve others.

The ABR operator framework processes the same sensor data on the irregular station graph, extracting relational structure — spatial gradients, inter-component coupling, and temporal evolution — directly from the measurements without interpolation, global basis decomposition, or source separation.

This repository runs both pipelines on the same 2-hour storm window, on the same data, and quantifies the difference.

**The claim is specific and falsifiable:** the traditional pipeline does not preserve a substantial fraction of the measured edge-field variance at the sensor network. The fraction is scale-dependent and truncation-dependent. At every parameter setting tested, it is nonzero. The relational structure ABR operates on is partially or wholly absent from the traditional pipeline's output.

---

## Results Summary

### Comparison Metrics (default parameters: edge 1000 km, SH degree 18)

| Metric | Value |
|---|---|
| Γ peak timing | 1 step before AE peak |
| Component coupling fraction of Γ | 99.2% |
| Edge-field variance not preserved | 39.5% mean, 54.2% at storm peak |
| Station reconstruction RMS | 118.9 nT |
| Mean fractional error at stations | 74.8% |

### Sensitivity Analysis — Edge Threshold

Edge-field variance not preserved by the traditional pipeline, as a function of the spatial scale at which relational structure is evaluated:

| Threshold (km) | Edges | Mean not preserved | Peak not preserved |
|---|---|---|---|
| 300 | 141 | 71.4% | 100.8% |
| 500 | 322 | 63.6% | 90.2% |
| 750 | 607 | 49.6% | 67.8% |
| 1000 | 972 | 39.5% | 54.2% |
| 1500 | 1877 | 27.6% | 37.7% |
| 2000 | 2947 | 22.3% | 30.3% |
| 3000 | 5076 | 18.8% | 26.6% |

At short baselines (300 km), the traditional pipeline fails to preserve over 70% of edge-field variance. The variance ratio exceeds 1.0 at some timesteps at this scale, indicating the metric is highly topology-sensitive at short baselines. At long baselines (3000 km), non-preservation is lower but still 19%.

### Sensitivity Analysis — SH Truncation Degree

Edge-field variance not preserved at 1000 km threshold, varying the number of spherical harmonic basis functions:

| SH Degree | Coefficients | Residual RMS (nT) | Not preserved |
|---|---|---|---|
| 6 | 48 | 187.7 | 85.8% |
| 10 | 120 | 151.6 | 67.4% |
| 14 | 224 | 132.4 | 52.5% |
| 18 | 360 | 118.9 | 39.5% |
| 24 | 624 | 110.4 | 28.3% |

Even at degree 24 (624 coefficients), 28.3% of edge-field variance is not preserved. Convergence is slow because the IDW interpolation (Phase 1.1) already smoothed the field before SH fitting.

---

## The Two Pipelines

### Traditional Pipeline (Phase 1)

```
Raw station data (246 stations, N/E/Z, 1-min cadence)
  → IDW grid interpolation (36×72, 5°×5°, k=4)          [non-injective]
  → Spherical harmonic decomposition (degree 1–18)        [non-injective]
  → Source separation (magnetospheric deg 1–3, iono 4+)   [non-injective]
  → Index construction (SYM-H proxy, AE proxy)            [non-injective]
```

Four non-injective transformations applied before analysis. Each is declared with preserved and discarded invariants in the source code.

### ABR Pipeline (Phase 2)

```
Raw station data (246 stations, N/E/Z, 1-min cadence)
  → Declare spatial topology (proximity graph, 1000 km)
  → Declare component topology (all-pairs: N-E, N-Z, E-Z)
  → Declare temporal topology (directed line, forward only)
  → Operator A: extract 3-topology edge field
  → Operator B: accumulate along each topology
  → Operator R: cross-couple between topologies
  → Γ = σ²(R∘B∘A) - σ²(B∘A)
```

No interpolation. No global basis decomposition. No source separation. The operators process relational structure where the magnetosphere intersects the sensors. Domain D is the sensor network itself. No claim is made about the magnetosphere between stations.

---

## Key Findings

### 1. Component Coupling (99.2% of Γ)

R's cross-topology circulation is dominated by the coupling between spatial gradients and component (directional) structure. The traditional pipeline processes N, E, Z as three independent scalar fields through the same SH basis. It cannot compute inter-component coupling because its architecture separates what the physics couples.

### 2. Temporal Lead (1 step)

Γ peaks one minute before the AE proxy. The relational structure of the perturbation field — how gradients are organized across the network — shifts before the scalar magnitude peaks. Consistent with the physical expectation that wave structure organizes before amplitude maximizes.

### 3. Scale-Dependent Non-Preservation

The traditional pipeline preserves progressively less edge-field variance at shorter relational baselines: 19% not preserved at 3000 km, 71% at 300 km. The traditional pipeline's SH basis has less representational capacity at the spatial scales where storm-driven gradients are strongest.

### 4. Residual Contains Relational Structure

Applying operator A to the traditional pipeline's residual (measured minus reconstructed at station locations) produces a nontrivial edge field. The information not preserved by the traditional pipeline is not unstructured noise — it contains relational gradients between nearby stations.

---

## Structural Declarations

### M_ABR

The ABR pipeline's measurement mapping adds no topological structure beyond what the sensor network provides. The irregular graph is the topology of the measurement. Contrast with the traditional pipeline, which adds a 5°×5° grid (uniform spacing not present in the measurement), SH basis (orthogonality not present in the physics), and source separation by degree (spectral separability not present during storms).

### Operator Admissibility

The ABR operators on this irregular proximity graph are not the canonical ring/torus operators for which Theorems 5 and 6 (Object Error) are proved. Γ on this graph is an empirical measurement of cross-topology coupling in a single pass, not a consequence of the spectral theorems. The ring is the topology of the proof, not the topology of the observable.

### R on the Irregular Graph

R cross-couples between topologies (spatial ↔ component ↔ temporal), not within them. The ring's forward-backward subtraction was a special case for single-axis systems. On the 3-topology field, R's antisymmetry is between edge types: spatial edges receive temporal asymmetry, component edges receive spatial asymmetry, temporal edges receive component asymmetry.

### Temporal Topology

Temporal evolution is strictly one-directional. Step t is adjacent to step t+1 only. No backward temporal edges. The system evolves forward.

### B Accumulation

B sums without degree normalization. High-degree stations produce larger accumulated edges. This is a structural property of sensor density: dense regions accumulate more because M has more information there.

### σ² as Declared Projection

All σ² values are scalar summaries of structured edge fields. Preserved: total relational variance. Discarded: spatial distribution, directional structure, per-edge detail, edge sign, component identity.

---

## Repository Structure

```
data/
  supermag.csv                  — SuperMAG 2015-03-16 to 2015-03-18

src/
  phase1_traditional.py         — Complete traditional pipeline (IDW → SH → separation → indices)
  phase2_1_abr_pipeline.py      — ABR 3-topology pipeline (spatial × component × temporal)
  phase3_comparison.py          — Four direct comparisons with figures
  phase3_sensitivity.py         — Parameter sensitivity analysis

output/
  phase1/                       — Traditional pipeline results
  phase2_1/                     — ABR pipeline results
  phase3/                       — Comparison summary
  phase3_sensitivity/           — Sensitivity sweep results

figures/
  comparison_temporal.png       — Γ vs AE vs SH power
  comparison_spatial.png        — Station-level reconstruction error
  comparison_components.png     — Component coupling vs independent SH
  comparison_residual.png       — A(residual) edge-field analysis
```

---

## Reproduction

```bash
# Phase 1: Traditional pipeline (2-hour window around storm peak)
python src/phase1_traditional.py data/supermag.csv

# Phase 2: ABR pipeline (same window)
python src/phase2_1_abr_pipeline.py data/supermag.csv

# Phase 3: Direct comparison (produces figures/)
python src/phase3_comparison.py

# Phase 3.5: Sensitivity analysis
python src/phase3_sensitivity.py
```

Requirements: numpy, pandas, scipy, matplotlib.

SuperMAG data is free but requires an account: https://supermag.jhuapl.edu/

---

## Open Conditions

1. **Coordinate frame.** The current pipeline uses geographic coordinates. The open condition from the kernel work is that R's component coupling may require source-native coordinates (SM/MLT). SuperMAG provides MLT and MCOLAT columns. A comparison in both frames would determine whether the choice affects Γ structure.

2. **ABR per-station spatial breakdown.** The 3-topology pipeline computes Γ over the full edge field but does not yet decompose per-station. Adding per-station σ² from R output would enable direct spatial comparison with the traditional pipeline's residual map.

3. **Multi-storm consistency.** These results are from a single event. The comparison should be repeated on at least two additional storms to confirm the non-preservation pattern is not event-specific.

4. **IDW parameter sensitivity.** The edge threshold and SH truncation sweeps are complete. IDW k (number of neighbors) and distance weighting exponent have not been swept.

5. **Window size sensitivity.** The 2-hour window was chosen for computational tractability with the 3-topology edge field. Extending to the full 3-day event with the per-timestep approach (not 3-topology) would test whether the results hold at longer timescales.

---

## References

- Macomber, R. (2026). Invariant Relational Evolution over Bounded Domains. arXiv:2601.22389.
- Macomber, R. (2026). The Object Error: A Formal Argument. Metatron Dynamics, Inc.
- Gjerloev, J.W. (2012). The SuperMAG data processing technique. J. Geophys. Res., 117, A09213.

---

*All definitions bounded over D. No claim beyond D. The structure described above does not require adoption. It describes relational admissibility conditions within D.*
