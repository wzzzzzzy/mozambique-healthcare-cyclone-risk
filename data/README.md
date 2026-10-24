# Data

No data are stored in this repository. The facility survey cannot be
redistributed, and the model inputs and outputs are large (about 1.3 GB for
the outputs behind the figures). This file lists what the pipeline reads and
where each input goes.

## Inputs read from the repository root

Several scripts default to these relative paths, so the folders sit at the
repository root (they are listed in `.gitignore`).

| Path | Content | Source and access |
|---|---|---|
| `SARA_processed/` | 1,644 healthcare facilities: location, type, urban/rural class, electricity, water and communication sources (processed from the 2018 survey) | Service Availability and Readiness Assessment (SARA), Mozambique 2018. **Restricted:** obtained by data request; not redistributable |
| `network_save/simplified_hospital_MG.pickle` | Reconstructed power grid with facility connections, as used by the hybrid model | Built from OpenStreetMap, GridFinder, World Bank and government maps (see manuscript Methods) |
| `network_save/MG_hospital.pickle` | Reference power network (generator and line attributes) | as above |
| `network_save/merge_MG_gridfiner.pickle` | Full-resolution power network (line-hazard routes and supports) | as above |
| `network_save/MG_c_0205.pickle` | Communication towers and facility links | OpenCellID-derived tower locations (World Bank) |
| `network_save/MG_w_hospital_new6.pickle` | Water network (water works, towers, covered reservoirs) with facility links | OpenStreetMap; rebuilt with `src/networks/` (pipeline stage 0) |
| `data/matpower_sources/` | Generator capacities and their sources (`generator_capacity.csv`, `sources.json`) for the MATPOWER case | public plant records; see `sources.json` |

## Inputs read from `$SCRATCH` by the job scripts

| Path | Content | Source |
|---|---|---|
| `cyclone_map/STORM/all_tracks/<set>.csv` | STORM synthetic tracks, South Indian basin, ten 1,000-year sets (present climate) | Bloemendaal et al. (2020), *Scientific Data* 7:40; 4TU.ResearchData |
| `cyclone_map/STORM/Moz_cyclone_track/<set>` | The 2,703 tracks whose path crosses Mozambique (row order = track number) | Selected from the STORM tracks |
| `World_Countries` | Country boundaries (Mozambique is row 151) | Esri World Countries (Generalized) |
| `wind_result/notebook_inputs/afr_g2014_2013_1` | Province boundaries (Figure 2B) | FAO GAUL 2014, admin level 1 |
| `wind_result/notebook_inputs/moz_admin_boundaries.shp/moz_admin3.shp` | Admin-3 boundaries (Figure S4) | Mozambique administrative boundaries (source to be confirmed) |
| `idai_*/input/Mozambique_Idai_2019_flip.tif` | Cyclone Idai wind raster (Figure S3) | source to be confirmed |
| `wind_result/synthetic_capacity_v2_public_net`, `wind_result/catalogue0_wind_track.tar.gz` | Node wind layers from the earlier line model, used only by the validation and representative-cyclone jobs | earlier runs of this model |
| `cascade_result/cc_csv_cyclone/…_capacity_v2_public_0_1_25.csv`, `hybrid_cyclone_v1/df_final_hybrid_25.csv`, `wind_result/df_final_all_grid_25.csv` | Earlier model results, used only for facility-cohort checks and comparisons | earlier runs of this model |

The facility-type table also reads the raw survey export
(`SARA_FINALRAWDATA_18Nov2018.xlsx`, restricted, with the questionnaire and
`SARA_DICIONARIO.xlsx`).

## Model outputs

The outputs behind every figure and table (pipeline stages 3–5: per-track
facility results, catalogue summaries, sensitivity and audit summaries, and
both solar variants) are archived separately. Archive: to be added, for
example as a restricted Zenodo record. The folder layout follows
`docs/pipeline.md`.
