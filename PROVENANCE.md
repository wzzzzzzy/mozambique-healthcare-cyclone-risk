# Provenance

Every file in `src/`, `moz_notebook/`, `matlab/` and `tests/` is a byte-identical copy of the
file at the listed path in the development repository `wzzzzzzy/InfraCascade`, branch `one-earth`, commit `cd7965e4e513f077c88a9b43449e3a4c58ef257a`.
The files in `ER11E04/` (Chavas et al. wind-profile model) are identical except that the `% email:` comment line
in each file header was removed; no code changed.
That code is the code that ran on Sherlock for the manuscript results; its checksums match the run checkouts.
Exception: `src/models/hybrid_multisystem_model.py` is the version at commit `e3f527c` of the same repository,
the checkout that ran the water/communication stage. (The later version on `one-earth` adds line-length
fields that need metadata the paper's power model does not contain.) It runs with the `one-earth`
`hybrid_power_model.py`, as checked against the stored outputs.

The job scripts in `workflow/sherlock/` are copies of the original job scripts, edited only so that they find
the scripts in this layout:

- script paths point to `src/<stage>/` and `PYTHONPATH` lists the six `src/` folders;
- the water/communication stage uses this repository's code (`MS_CODE`) instead of a separate checkout;
- `run_solar_cyclone.sbatch` and `run_wind_track_func_hospital.sbatch` change to `$INFRACASCADE_PROJECT_DIR` instead of a home-directory path;
- `run_figures_t24.sbatch` no longer draws the historical-versus-synthetic figure, which belongs to a separate study.

## Reproduction check

On 2026-10-24 this repository was copied to Sherlock and its job scripts were
run from scratch for ten synthetic tracks (STORM set 0, tracks 5-9 and 50-54;
track 52 is one of the representative cyclones in Figure S5):

- `run_line_hazard_v2_chunk.sbatch`: winds computed with the MATLAB wind model, and line-failure probabilities;
- `run_hybrid_t24_chunk.sbatch`: public-grid electricity, all-grid counterfactual, water and communication, and Figure 2 metrics.

Every output was compared with the stored manuscript results:

| Stage | Files | Differences |
|---|---|---|
| Winds | 10 | 0 |
| Line-failure probabilities (public, all-grid) | 20 | 0 |
| Public-grid electricity (samples, facilities, summaries) | 30 | 0 |
| All-grid counterfactual | 30 | 0 |
| Water and communication (4,990 rows) | 30 | 0 |
| Figure 2 province metrics | 4 | 0 |

All outputs are identical (maximum absolute difference 0). Cells that are
empty in the stored outputs are also empty in the reproduced ones.

## Files

| File | Source path | MD5 |
|---|---|---|
| `ER11E04/E04_outerwind_nondim.m` | `ER11E04/E04_outerwind_nondim.m` (email comment removed) | `7666e4c6c8e9b83e18a684bef7f2d393` |
| `ER11E04/E04_outerwind_nondim_LargePond.m` | `ER11E04/E04_outerwind_nondim_LargePond.m` (email comment removed) | `bedf685626971a9f06d144555293584f` |
| `ER11E04/E04_outerwind_nondim_MM0.m` | `ER11E04/E04_outerwind_nondim_MM0.m` (email comment removed) | `d021bd4bc40f21e092b76897506cfbcf` |
| `ER11E04/E04_outerwind_r0input_nondim.m` | `ER11E04/E04_outerwind_r0input_nondim.m` (email comment removed) | `35f1cfa9f900f62260d81e681a49a87b` |
| `ER11E04/E04_outerwind_r0input_nondim_MM0.m` | `ER11E04/E04_outerwind_r0input_nondim_MM0.m` (email comment removed) | `d15cc510028c1dc375d1ffa3a680ca0e` |
| `ER11E04/E04_outerwind_r0input_nondim_MM0_LargePond.m` | `ER11E04/E04_outerwind_r0input_nondim_MM0_LargePond.m` (email comment removed) | `8bf4882eff6bfeafa74d1f23606018dd` |
| `ER11E04/ER11E04_nondim_r0input.m` | `ER11E04/ER11E04_nondim_r0input.m` (email comment removed) | `a700c1cde7c4c1caad37023f75f23039` |
| `ER11E04/ER11E04_nondim_rfitinput.m` | `ER11E04/ER11E04_nondim_rfitinput.m` (email comment removed) | `39295e29ce06d4bfbd95cd52c7ec2fe6` |
| `ER11E04/ER11E04_nondim_rmaxinput.m` | `ER11E04/ER11E04_nondim_rmaxinput.m` (email comment removed) | `f19f0597ff99986522bc94f8c180221a` |
| `ER11E04/ER11_radprof.m` | `ER11E04/ER11_radprof.m` (email comment removed) | `571acb2b03990e852a91b03f95021eaa` |
| `ER11E04/ER11_radprof_nondim.m` | `ER11E04/ER11_radprof_nondim.m` (email comment removed) | `02d7d73ba1374313e3ecc59566ac0126` |
| `ER11E04/ER11_radprof_raw.m` | `ER11E04/ER11_radprof_raw.m` (email comment removed) | `e71146223956e466245de5481a94f199` |
| `ER11E04/ER11_radprof_raw_nondim.m` | `ER11E04/ER11_radprof_raw_nondim.m` (email comment removed) | `6713c12da7db93aa392029c04f89dddf` |
| `ER11E04/ER11_radprof_raw_nondim_MMm.m` | `ER11E04/ER11_radprof_raw_nondim_MMm.m` (email comment removed) | `15698d43f0f39ee66dbec8a8fa8692d8` |
| `ER11E04/curveintersect.m` | `ER11E04/curveintersect.m` | `9b9d138d0fe7caad2cfaddc3f8874e47` |
| `ER11E04/radprof_eyeadj.m` | `ER11E04/radprof_eyeadj.m` (email comment removed) | `5c55917fd5c850780e8cac6e6867c105` |
| `ER11E04/testing/ER11E04_testing.m` | `ER11E04/testing/ER11E04_testing.m` | `7a481b49cf8a8dc28464a8cf72b5aebd` |
| `matlab/generate_cyclone_footprints.m` | `matlab/generate_cyclone_footprints.m` | `ba912f52fa18cd5348e5879167490324` |
| `moz_notebook/figure_02_electricity_sources.ipynb` | `moz_notebook/figure_02_electricity_sources.ipynb` | `7bdeac92fb804c35a0ad64a74e1ce66e` |
| `moz_notebook/figure_03_cascading_service_failures.ipynb` | `moz_notebook/figure_03_cascading_service_failures.ipynb` | `5dba1508a0f0dd34b97238e363ccb75a` |
| `moz_notebook/figure_05_grid_cascade.ipynb` | `moz_notebook/figure_05_grid_cascade.ipynb` | `3e6ac347f667f0375f2c309d727b0e11` |
| `src/analysis/audit_all_grid_frozen_demand.py` | `audit_all_grid_frozen_demand.py` | `b6edf3d7b4526c11881763310a9d375f` |
| `src/analysis/audit_dc_balance_gap_v2.py` | `audit_dc_balance_gap_v2.py` | `03abc5680388cf70918505105b824796` |
| `src/analysis/equipment_summed_index.py` | `equipment_summed_index.py` | `40c97f006f2c87f5b9728609e454e9cd` |
| `src/analysis/facility_type_electricity_services.py` | `facility_type_electricity_services.py` | `54b8793e9861ee5a5ba8bfba4aad1492` |
| `src/analysis/hybrid_sensitivity_v2.py` | `hybrid_sensitivity_v2.py` | `fde85501886c33698fa223d73895af1a` |
| `src/analysis/select_hybrid_representative_cyclones.py` | `select_hybrid_representative_cyclones.py` | `06ec7ba9b82e2621d2781af912879528` |
| `src/analysis/summarize_hybrid_cyclones.py` | `summarize_hybrid_cyclones.py` | `10f86164fe4fba19b88ea808fc759b47` |
| `src/analysis/summarize_hybrid_multisystem.py` | `summarize_hybrid_multisystem.py` | `74922a522443ecaf0ff2426440b65dc1` |
| `src/analysis/summarize_idai_threshold_hybrid.py` | `summarize_idai_threshold_hybrid.py` | `ec8b94f4fd8850f7dc75028994852938` |
| `src/analysis/summarize_idai_tower_conductor.py` | `summarize_idai_tower_conductor.py` | `b5560ca0a9495c8048614afe71c0e38e` |
| `src/analysis/urban_rural_electricity_table.py` | `urban_rural_electricity_table.py` | `54f63ffcf73a2d428ccb3a288b01bfdf` |
| `src/analysis/verify_facility_type_text.py` | `verify_facility_type_text.py` | `69b54c48d3c1e49f50ec99567539061c` |
| `src/common/helvetica_style.py` | `helvetica_style.py` | `55b4db593c28b1dbeed832e3b69376ba` |
| `src/common/synthetic_tracks.py` | `synthetic_tracks.py` | `6aa1bd7bb329500df7d391d9e3ba93f0` |
| `src/common/utils.py` | `utils.py` | `d164b93d553a2b1c746fd6c61e88ecc6` |
| `src/figures/capacity_ablation_figure.py` | `capacity_ablation_figure.py` | `016b3aca0fe8df04ac03f317542bd124` |
| `src/figures/draw_cascade.py` | `draw_cascade.py` | `8bbac94add28d7d307e2205e0f115322` |
| `src/figures/figure_S1_regional_direct.py` | `figure_S1_regional_direct.py` | `592f5229038cf1a66f2606b87656c8b2` |
| `src/figures/hybrid_city_electricity_figure.py` | `hybrid_city_electricity_figure.py` | `a73262038e7204a0f8d613fbbb77157b` |
| `src/figures/hybrid_representative_cyclones_figure.py` | `hybrid_representative_cyclones_figure.py` | `0c9de0a49384799519ed87073a040b70` |
| `src/figures/make_manuscript_figure_notebooks.py` | `make_manuscript_figure_notebooks.py` | `d8c985f66860f29ea993729c546ce0a3` |
| `src/figures/plot_figure6_map_style.py` | `plot_figure6_map_style.py` | `8e6d7ff717cd5a670de954af661358e6` |
| `src/figures/plot_figure_S3_line_hazard_v2.py` | `plot_figure_S3_line_hazard_v2.py` | `28d959e8f333f7b05e0829506e0ceaa8` |
| `src/figures/plot_figure_S6_fragility.py` | `plot_figure_S6_fragility.py` | `cf101642108a5b5e992c8d906a792c63` |
| `src/figures/plot_figure_S8_multilayer.py` | `plot_figure_S8_multilayer.py` | `f9c0b7c11d1ab7f8258319bfbf455b14` |
| `src/figures/plot_figure_equipment_disruption_map.py` | `plot_figure_equipment_disruption_map.py` | `817fee77c39715db8318dd7eb91629c0` |
| `src/figures/plot_figure_facility_type_electricity.py` | `plot_figure_facility_type_electricity.py` | `da85220b018c793fee16ba7dd598d6a1` |
| `src/figures/plot_hybrid_sensitivity_overview.py` | `plot_hybrid_sensitivity_overview.py` | `c1c8d6bd3906dce94f436890d6ef1853` |
| `src/figures/plot_idai_threshold_lengths.py` | `plot_idai_threshold_lengths.py` | `b5888bb687e2ed6436d283c8c1d9aee2` |
| `src/figures/prepare_cascade_admin_summary.py` | `prepare_cascade_admin_summary.py` | `0b2253c112c2fb7666a1a2f76b71b9d2` |
| `src/figures/prepare_failure_figure_inputs.py` | `prepare_failure_figure_inputs.py` | `4f6f7ddedff06db40bbb725406d1eb43` |
| `src/figures/prepare_hybrid_cascade_admin_summary.py` | `prepare_hybrid_cascade_admin_summary.py` | `1f921deca33231fdaf9d7a5ce4f13f9f` |
| `src/hazard/add_wind_to_hospital_for_track_func.py` | `add_wind_to_hospital_for_track_func.py` | `ed3eee0bdbb7cec8291e41b6dba89548` |
| `src/hazard/cyclone_direct_damage.py` | `cyclone_direct_damage.py` | `19d9deacd1b6a3e9436389b9623a1cbf` |
| `src/hazard/idai_direct_damage_check.py` | `idai_direct_damage_check.py` | `57ffedab8ee556d51cb2cae1a15f3d7a` |
| `src/hazard/length_normalized_line_hazard.py` | `length_normalized_line_hazard.py` | `6e18d465c3a826ccf4338c8dd3c23490` |
| `src/hazard/line_hazard_v2.py` | `line_hazard_v2.py` | `e8223fb4c876cd9cedb7ebc8dfc36acf` |
| `src/hazard/prepare_historical_wind_track.py` | `prepare_historical_wind_track.py` | `f912643601418d56a5fc5719751cccb0` |
| `src/hazard/prepare_idai_threshold_wind.py` | `prepare_idai_threshold_wind.py` | `8581383d912089e4fd4ae12a5711cccc` |
| `src/hazard/simulation_on_solar.py` | `simulation_on_solar.py` | `c903951acb9ef525229cf08d71c5824c` |
| `src/hazard/simulation_on_wind_graph.py` | `simulation_on_wind_graph.py` | `93f49f80689c22d1819caf8277050613` |
| `src/hazard/solar_exposure_gust.py` | `solar_exposure_gust.py` | `ff0a6e4f616597403b4abbdfbc1f90db` |
| `src/models/dependency_models.py` | `dependency_models.py` | `f607193b4ed72db5681c75b2fc5cc9be` |
| `src/models/hybrid_figure5_metrics.py` | `hybrid_figure5_metrics.py` | `d21055efbfd72bcda12afb9d137530fa` |
| `src/models/hybrid_multisystem_model.py` | `hybrid_multisystem_model.py` at `e3f527c` | `6e8751f0ae6c5d3a23d1610f6a0c9407` |
| `src/models/hybrid_power_model.py` | `hybrid_power_model.py` | `48937a2c75f3bad73f062a2d660ae0d3` |
| `src/models/hybrid_sensitivity.py` | `hybrid_sensitivity.py` | `c6d36e8e19c89c0576159b7c0071b1f8` |
| `src/models/propagate_cyclone_dependencies.py` | `propagate_cyclone_dependencies.py` | `c03290c65a09644778e4a7378249307d` |
| `src/networks/build_hybrid_power_model.py` | `build_hybrid_power_model.py` | `c0af7657f5c1c9b650da0e60088c2056` |
| `src/networks/build_matpower_case.py` | `build_matpower_case.py` | `fd875142e0034d4778d62a6082c3d64d` |
| `src/networks/build_osm_water_network.py` | `build_osm_water_network.py` | `15256cb7752c838a783a3315def7d02a` |
| `src/networks/build_water_graph_new2_rules.py` | `build_water_graph_new2_rules.py` | `3ff32125152d88de5db492b51f5d32c1` |
| `src/networks/extract_osm_water_features.py` | `extract_osm_water_features.py` | `ff96955cec8d6966ca0e06cdc2308e3d` |
| `src/networks/hospital_grid_ablation.py` | `hospital_grid_ablation.py` | `bc4eb56892482bc7c55471b2fee0a849` |
| `src/networks/power_capacity.py` | `power_capacity.py` | `31420a909065408230cefb13ec5340a1` |
| `src/networks/prepare_hybrid_hospital_scenario.py` | `prepare_hybrid_hospital_scenario.py` | `60e9813672c9491d6687a40e37693f56` |
| `src/networks/prepare_matpower_cyclone_input.py` | `prepare_matpower_cyclone_input.py` | `1f9dfa70072b102f5886dbfd4aa93e94` |
| `tests/test_build_matpower_case.py` | `tests/test_build_matpower_case.py` | `f82bca0242c3ba59db3d73e694b4080f` |
| `tests/test_capacity_ablation_figure.py` | `tests/test_capacity_ablation_figure.py` | `7a8d6c008ed61c2394d2e2c01cd0f6a4` |
| `tests/test_cyclone_dependency_propagation.py` | `tests/test_cyclone_dependency_propagation.py` | `109632870eddaf551a55473e58582ac8` |
| `tests/test_dependency_models.py` | `tests/test_dependency_models.py` | `c80844b04361d5f25530b5e993c8e813` |
| `tests/test_draw_cascade.py` | `tests/test_draw_cascade.py` | `87f651b5017f804f1e21c89c949d7b20` |
| `tests/test_hospital_grid_ablation.py` | `tests/test_hospital_grid_ablation.py` | `880ba8bf6a31c1bc41e98d446845f705` |
| `tests/test_hybrid_multisystem_model.py` | `tests/test_hybrid_multisystem_model.py` | `a893871e66f9aa1b89f0dc0d50f3f5df` |
| `tests/test_hybrid_power_model.py` | `tests/test_hybrid_power_model.py` | `469848b5bae663246811699cd901c295` |
| `tests/test_monte_carlo_pairing.py` | `tests/test_monte_carlo_pairing.py` | `37df8a3ea967f28da5742ecc0e184e0f` |
| `tests/test_power_capacity.py` | `tests/test_power_capacity.py` | `df17b74ce3867c9ba742d01fa252a3de` |
| `tests/test_prepare_matpower_cyclone_input.py` | `tests/test_prepare_matpower_cyclone_input.py` | `d00e3ac9668829e74d8d3703057b552b` |
| `workflow/sherlock/run_figures_t24.sbatch` | `run_figures_t24.sbatch` (paths adapted) | `585c255e13ee1c3b41a42d8fa37ab97e` |
| `workflow/sherlock/run_figures_t24_solar_exposure.sbatch` | `run_figures_t24_solar_exposure.sbatch` (paths adapted) | `2f55741ab97ea85f3e078c3d1c8a0462` |
| `workflow/sherlock/run_hybrid_all_grid_setup.sbatch` | `run_hybrid_all_grid_setup.sbatch` (paths adapted) | `5433795d0ffd7410cbf0a6739606a3e7` |
| `workflow/sherlock/run_hybrid_cyclone_setup.sbatch` | `run_hybrid_cyclone_setup.sbatch` (paths adapted) | `86ca724d1bebd3bb599d3f56d07904ac` |
| `workflow/sherlock/run_hybrid_figure5_prepare.sbatch` | `run_hybrid_figure5_prepare.sbatch` (paths adapted) | `78e88a33841955bf3b24a0790a2903b8` |
| `workflow/sherlock/run_hybrid_multisystem_setup_water_graph.sbatch` | `run_hybrid_multisystem_setup_water_graph.sbatch` (paths adapted) | `cf1d81706c7e62c416c58ea8c94d9fed` |
| `workflow/sherlock/run_hybrid_representative_cyclones_draw_cascade.sbatch` | `run_hybrid_representative_cyclones_draw_cascade.sbatch` (paths adapted) | `0b4c8d47e2bab7bcb1a3110114e8a9cb` |
| `workflow/sherlock/run_hybrid_sensitivity.sbatch` | `run_hybrid_sensitivity.sbatch` (paths adapted) | `ac0612058380963608182b8b4042872e` |
| `workflow/sherlock/run_hybrid_t24_chunk.sbatch` | `run_hybrid_t24_chunk.sbatch` (paths adapted) | `fc6b7882676b2c18d36073fb900f9101` |
| `workflow/sherlock/run_hybrid_t24_summary.sbatch` | `run_hybrid_t24_summary.sbatch` (paths adapted) | `d2fe1114696b9b8e0199feeb10661836` |
| `workflow/sherlock/run_idai_length_normalized.sbatch` | `run_idai_length_normalized.sbatch` (paths adapted) | `074161483dd0920e3c430377e43c9676` |
| `workflow/sherlock/run_idai_threshold_hybrid.sbatch` | `run_idai_threshold_hybrid.sbatch` (paths adapted) | `6462c07d12d7bca8f624f2600072f261` |
| `workflow/sherlock/run_idai_threshold_lengths.sbatch` | `run_idai_threshold_lengths.sbatch` (paths adapted) | `32f8bf9189e08b0b962b540cbb111adf` |
| `workflow/sherlock/run_idai_tower_conductor.sbatch` | `run_idai_tower_conductor.sbatch` (paths adapted) | `41dffda2e020a9a729aa928915415ba4` |
| `workflow/sherlock/run_line_hazard_v2_chunk.sbatch` | `run_line_hazard_v2_chunk.sbatch` (paths adapted) | `72d1457403b8361b0adb534719be3058` |
| `workflow/sherlock/run_line_hazard_v2_validation.sbatch` | `run_line_hazard_v2_validation.sbatch` (paths adapted) | `02b96dd63699c6d8f4ba5a61cfdc5f2b` |
| `workflow/sherlock/run_select_hybrid_representative_cyclones.sbatch` | `run_select_hybrid_representative_cyclones.sbatch` (paths adapted) | `8a4b8f58ef58167e4458d44c8fb6bbb0` |
| `workflow/sherlock/run_solar_cyclone.sbatch` | `run_solar_cyclone.sbatch` (paths adapted) | `79b8d894d0918ac917b83297e7060000` |
| `workflow/sherlock/run_solar_exposure_gust.sbatch` | `run_solar_exposure_gust.sbatch` (paths adapted) | `ea74540f077eb132ec48328736c58f8f` |
| `workflow/sherlock/run_t24_supplement_chunk.sbatch` | `run_t24_supplement_chunk.sbatch` (paths adapted) | `03f1e4f1db3ceb6020f199d6d6f2a621` |
| `workflow/sherlock/run_t24_supplement_summary.sbatch` | `run_t24_supplement_summary.sbatch` (paths adapted) | `405758613bc07b391375e85c72904327` |
| `workflow/sherlock/run_wind_track_func_hospital.sbatch` | `run_wind_track_func_hospital.sbatch` (paths adapted) | `6f266617df5ce8ba350bd30dfb462070` |
