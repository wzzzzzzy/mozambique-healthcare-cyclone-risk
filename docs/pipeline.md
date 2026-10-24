# Pipeline

The stages below reproduce the manuscript results in order. Job scripts are in
`workflow/sherlock/`; submit them from the repository root with

```bash
export INFRACASCADE_PROJECT_DIR=$PWD           # repository root (some jobs read PROJECT_DIR from it)
export LINE_HAZARD_ROOT=$SCRATCH/hybrid_t24_l025_v1   # main-case results folder
sbatch --export=ALL,LINE_HAZARD_ROOT=$LINE_HAZARD_ROOT,INFRACASCADE_PROJECT_DIR=$PWD workflow/sherlock/<job>.sbatch ...
```

Each job's header states its arguments. Folder names below are the ones used
for the manuscript run; the jobs refuse to overwrite existing outputs. Global
track numbering follows `src/common/synthetic_tracks.py`: tracks
0–2,702 run over the ten STORM sets in order (290, 274, 263, 276, 273, 279,
251, 264, 242 and 291 tracks).

## Stage 0: inputs

See [../data/README.md](../data/README.md). The cyclone tracks, network graphs,
facility survey and boundaries are inputs. The reconstructed power,
communication and water graphs (`network_save/`) were built before this
pipeline; the scripts in `src/networks/` rebuild the OSM water graph
(`extract_osm_water_features.py` → `build_osm_water_network.py` →
`build_water_graph_new2_rules.py`, giving `MG_w_hospital_new6.pickle`).

## Stage 1: network and model setup (run once)

| Step | Job / command | Output |
|---|---|---|
| Public-grid hybrid model (MATPOWER case, hybrid model, path cache) | `run_hybrid_cyclone_setup.sbatch` | `$SCRATCH/hybrid_cyclone_v1/model/` |
| All-grid scenario model (every facility connected to the grid) | `run_hybrid_all_grid_setup.sbatch`, run with `HYBRID_CYCLONE_ROOT=$SCRATCH/hybrid_all_grid_final` | `$SCRATCH/hybrid_all_grid_final/model/` |
| Frozen-demand all-grid model | one-off step, see below | `$SCRATCH/hybrid_all_grid_frozen_demand_v1/model/` |
| Water and communication dependency model | `run_hybrid_multisystem_setup_water_graph.sbatch network_save/MG_w_hospital_new6.pickle` | `$SCRATCH/hybrid_multisystem_water_new6_v1/model/water_comm_dependencies.pickle` |
| Line lengths and category 1–2 track list (Figure 2) | `run_hybrid_figure5_prepare.sbatch` | `$SCRATCH/hybrid_figure5_length_v2/` |
| Sensitivity scenario models | `python3 src/models/hybrid_sensitivity.py build ...` | `$SCRATCH/hybrid_sensitivity_v1/models/`, `scenarios.json` |
| Line-hazard geometry and validation | `run_line_hazard_v2_validation.sbatch` (`line_hazard_v2.py geometry`, then checks against earlier inputs) | `$LINE_HAZARD_ROOT/geometry.npz`, `validation/` |

**Frozen-demand all-grid model.** For the counterfactual in Figure 4, the
all-grid model keeps the current system's demand. `hybrid_power_model.npz` of
`hybrid_all_grid_final` was copied, and `node_pd_mw` at the 2,923 nodes that
exist in the public-grid model was set to the public-grid values, matched by
node ID. The 660 added facility nodes keep 0 MW, and every other array is
unchanged. Total demand is 1,064.37 MW and total generation capacity 1,252.2
MW. The result has SHA-256
`f6a2cb150291d34846dc93eaf3053bc5a34f54c27e5e2a9aacc9b10e3eac7893`;
`frozen_demand_provenance.json` in that folder records the check that the
demand equals the baseline and all other arrays equal the all-grid model.

## Stage 2: hazard per track

| Step | Job | Output |
|---|---|---|
| Winds along every line and branch failure probabilities (24 m/s, L0 = 0.25 km, towers + conductor) | `sbatch --array=0-N run_line_hazard_v2_chunk.sbatch TRACKS_PER_TASK` | `$LINE_HAZARD_ROOT/winds/`, `probabilities/{public,all_grid}/` |
| Solar panel damage and irradiance, gust factor 1.64 | `run_solar_cyclone.sbatch STORM_SET_FOLDER` | `$SCRATCH/wind_result/synthetic_solar/` |
| Solar damage with exposure-dependent gust factors | `sbatch --array=0-27 run_solar_exposure_gust.sbatch winds 100`, then `run_solar_exposure_gust.sbatch convert 5` | `$LINE_HAZARD_ROOT/solar_exposure/synthetic_solar_coastal5km/` |
| Facility wind and direct damage inputs | `run_wind_track_func_hospital.sbatch STORM_SET_FOLDER` | `$SCRATCH/wind_result/synthetic_new_hospital/` |
| Regional direct-damage ratios (Eberenz et al. South Indian TDR curve) | `python3 src/hazard/cyclone_direct_damage.py ...` | `$SCRATCH/cyclone_direct_damage_v1/results/damage_ratio_si_tdr.csv.gz` |

Wind fields use the MATLAB Engine (`ER11E04/`).

## Stage 3: service models per track

`sbatch --array=0-N run_hybrid_t24_chunk.sbatch TRACKS_PER_TASK 100` runs, for
each track, from the stage-2 probabilities:

- **Public-grid electricity:** `src/models/hybrid_power_model.py`, written to `public/results/`.
- **Frozen-demand all-grid counterfactual:** `all_grid/results/`.
- **Water and communication:** `src/models/hybrid_multisystem_model.py`, written to `multisystem/`.
- **Province line-length metrics for category 1–2 tracks:** `src/models/hybrid_figure5_metrics.py`, written to `figure5/`.

Every model uses 100 samples per track and seed 1234; the same sampled line
failures are shared across models.

## Stage 4: catalogue summaries

`run_hybrid_t24_summary.sbatch` writes:

- **Public grid:** `public/df_final_hybrid_t24.csv` (facility × track shortfall) and
  `public/hybrid_cyclone_summary.csv`.
- **All-grid counterfactual:** `all_grid/df_final_hybrid_all_grid_t24.csv`.
- **Figure 2 metrics:** `figure5/hybrid_multisystem_track_summary.csv`.
- **Water and communication:** the catalogue tables, with `compat_25/` links
  under the `_25` file names that the figure loaders expect.

## Stage 5: supplementary analyses

| Analysis | Jobs | Output |
|---|---|---|
| Demand-reallocation decomposition of the all-grid result | `run_t24_supplement_chunk.sbatch decomp N`, `run_t24_supplement_summary.sbatch decomp` | `all_grid_reallocated/`, `all_grid/audit_frozen_vs_reallocated/` |
| Sensitivity to demand allocation, demand, generation and line ratings | `... sensitivity N`, `... summary sensitivity` | `sensitivity/summary/` |
| DC power-flow versus power-balance island audit | `... dc N`, `... summary dc` | `dc_audit/summary/` |
| Cyclone Idai line-fragility tests (Figure S3) | `run_idai_threshold_hybrid.sbatch`, `run_idai_length_normalized.sbatch`, `run_idai_tower_conductor.sbatch`, `run_idai_threshold_lengths.sbatch`; summarized with `src/analysis/summarize_idai_tower_conductor.py` | `$SCRATCH/idai_*` |
| Representative cyclones (Figure S5) | `run_select_hybrid_representative_cyclones.sbatch`, then `run_hybrid_representative_cyclones_draw_cascade.sbatch` | figure folder given as argument |

## Stage 6: figures and tables

`run_figures_t24.sbatch OUTPUT_DIR` draws Figures 2B, 3, 4, 5, S2 and S4 with
the gust-factor-1.64 solar results.
`run_figures_t24_solar_exposure.sbatch SOLAR_ROOT OUTPUT_DIR` redraws the
solar-dependent Figures 3, 4 and S4 with the exposure-dependent gust factors.
The remaining figures and tables run on a laptop from the stage-4/5 outputs;
see [figures.md](figures.md) for every command.
