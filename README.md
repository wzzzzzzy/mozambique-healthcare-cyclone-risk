# Cyclone risk to healthcare-facility services in Mozambique

Code for the One Earth manuscript *Different cyclone preparedness needs for
urban and rural healthcare in the Global South* (Wei, Jing, He, Bendavid and
Suckale). The model estimates how tropical cyclones disrupt the electricity,
water and communication services of 1,644 Mozambican healthcare facilities. It
combines:

- a 10,000-year synthetic cyclone catalogue (STORM);
- a parametric wind model (Chavas et al. ER11E04);
- wind fragility curves for power lines, towers and solar panels;
- a hybrid connected-component / DC power-flow model of the national grid;
- dependency rules that pass electricity loss on to water and communication.

The code in `src/`, `moz_notebook/`, `ER11E04/` and `matlab/` is byte-identical
to the code that produced the manuscript results (see [PROVENANCE.md](PROVENANCE.md)).
Data are not included; see [data/README.md](data/README.md).

## Repository layout

```
src/
  common/     shared helpers (track bookkeeping, node positions, figure font)
  networks/   power and water network models built from the reconstructed graphs
  hazard/     wind fields, line/tower/solar fragility, direct facility damage
  models/     hybrid power model, water/communication propagation, Figure 2 metrics
  analysis/   catalogue summaries, sensitivity tests, audits, facility tables
  figures/    figure scripts and the generator of the figure notebooks
moz_notebook/ figure notebooks (manuscript Figures 2B, 3 and 5)
ER11E04/      Chavas et al. (2015) ER11E04 wind-profile model (MATLAB)
matlab/       MATLAB helper for the wind footprints in Figure S5
workflow/
  sherlock/   Slurm job scripts for every pipeline stage (Stanford Sherlock)
tests/        unit tests of the network, power and dependency models
docs/         pipeline description and figure-to-code map
data/         README only; data are obtained separately
```

The Python modules import each other by module name, so all six `src/` folders
go on `PYTHONPATH`. The job scripts set this; for interactive use run, from the
repository root:

```bash
export PYTHONPATH=$(printf '%s:' "$PWD"/src/*)$PYTHONPATH
```

Some modules use relative default paths (`SARA_processed/`, `network_save/`,
`ER11E04/`, `matlab/`, `moz_notebook/`), so commands run from the repository
root and the data folders are placed there (see [data/README.md](data/README.md)).

## Reproducing the results

The full pipeline runs on an HPC cluster and needs MATLAB. The job scripts were
written for Stanford's Sherlock cluster: they load Sherlock modules, read
inputs from `$SCRATCH` and use the Python environment named in `ENV_ROOT`. On
another cluster, change those three things. The scripts in `src/` themselves
contain no cluster-specific paths.

1. **Environment.** Python 3.9 with the packages in
   [requirements.txt](requirements.txt), and MATLAB R2022b with the MATLAB Engine
   for Python (wind fields only).
2. **Data.** Obtain the inputs listed in [data/README.md](data/README.md) and
   place them as described there.
3. **Pipeline.** Run the stages in [docs/pipeline.md](docs/pipeline.md):
   1. build the network models;
   2. compute winds and component failure probabilities for every track;
   3. run the power, water and communication models;
   4. summarize the catalogue;
   5. run the supplementary analyses;
   6. draw the figures.

   With the cluster's array jobs, the 2,703 tracks take about a day.
4. **Figures and tables.** [docs/figures.md](docs/figures.md) maps every
   manuscript figure and table to its script, job and inputs.

Run the unit tests with `python -m pytest tests` (with `PYTHONPATH` set as
above).

### Key settings of the main case

| Setting | Value |
|---|---|
| Synthetic catalogue | STORM South Indian basin, 10 × 1,000 years; 2,703 tracks crossing Mozambique |
| Line fragility | conductor q(V) = 2^((V − 24)/30) − 1 per 0.25 km (0 below 24 m/s, 1 above 54 m/s), wind sampled along spans every ≤ 0.5 km; mapped towers and poles with an Emanuel-form curve (threshold 24 m/s, 50% at 110 kn) |
| Power model | connected components with a DC-flow uniform served fraction; 100 Monte Carlo samples per track, seed 1234 |
| Solar panels | lognormal in the 3-s gust (median 80 m/s, β 0.32); gust factor 1.66 inland, 1.38 / 1.52 for onshore / offshore flow within 5 km of the coast (1.64 everywhere in the earlier variant) |
| Annualized index | mean over tracks of the event-level expected fractional shortfall × 2,703 / 10,000 |

## Citation and contact

Please cite the One Earth article (reference to be added on publication).
For questions, open an issue in this repository.
