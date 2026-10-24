from pathlib import Path

import numpy as np

from hybrid_power_model import HybridPowerModel


def write_model(
    path: Path,
    pmax: float = 100.0,
    demand: float = 50.0,
    rate: float = 40.0,
) -> Path:
    # gen bus -- load bus -- modeled hospital; an unmodeled hospital is kept
    # in the output cohort but not in the simulated topology.
    np.savez_compressed(
        path,
        format_version=np.asarray([2], dtype=np.int16),
        total_edge_count=np.asarray([2], dtype=np.int64),
        all_edge_length_km=np.asarray([10.0, 2.0]),
        all_edge_is_transmission=np.asarray([True, False]),
        node_ids=np.asarray(["gen", "load", "hospital-a"]),
        node_is_power=np.asarray([True, True, False]),
        node_pd_mw=np.asarray([0.0, demand, 0.0]),
        node_pmax_mw=np.asarray([pmax, 0.0, 0.0]),
        edge_uid=np.asarray([1, 2]),
        edge_u=np.asarray([0, 1]),
        edge_v=np.asarray([1, 2]),
        edge_is_power=np.asarray([True, False]),
        edge_is_transmission=np.asarray([True, False]),
        edge_length_km=np.asarray([10.0, 2.0]),
        edge_x_pu=np.asarray([0.1, np.nan]),
        edge_rate_mw=np.asarray([rate, np.nan]),
        matpower_branch_id=np.asarray([1, -1]),
        hospital_ids=np.asarray(["hospital-a", "hospital-unmodeled"]),
        hospital_node=np.asarray([2, -1]),
        hospital_modeled=np.asarray([True, False]),
    )
    return path


def test_flow_limit_reduces_component_service(tmp_path):
    model = HybridPowerModel(write_model(tmp_path / "model.npz"))
    result = model.solve_sample(np.asarray([False, False]))
    assert np.isclose(result.served_mw, 40.0)
    assert np.isclose(result.hospital_failure[0], 0.2)
    assert result.hospital_failure[1] == -1.0
    assert result.flow_constrained_components == 1
    assert result.capacity_constrained_components == 0


def test_capacity_limit_replaces_four_mw_per_hospital_rule(tmp_path):
    model = HybridPowerModel(
        write_model(tmp_path / "model.npz", pmax=25.0, rate=100.0)
    )
    result = model.solve_sample(np.asarray([False, False]))
    assert np.isclose(result.served_mw, 25.0)
    assert np.isclose(result.hospital_failure[0], 0.5)
    assert result.flow_constrained_components == 0
    assert result.capacity_constrained_components == 1


def test_failed_service_edge_is_a_hospital_outage(tmp_path):
    model = HybridPowerModel(
        write_model(tmp_path / "model.npz", rate=100.0)
    )
    result = model.solve_sample(np.asarray([False, True]))
    assert np.isclose(result.served_mw, 50.0)
    assert result.hospital_failure[0] == 1.0
    assert result.hospital_failure[1] == -1.0


def test_sample_state_exposes_component_service_for_retained_nodes(tmp_path):
    model = HybridPowerModel(
        write_model(tmp_path / "model.npz", pmax=25.0, rate=100.0)
    )
    state = model.solve_sample_state(np.asarray([False, False]))

    assert state.result.capacity_constrained_components == 1
    np.testing.assert_array_equal(state.component_by_node, [0, 0, 0])
    np.testing.assert_allclose(state.node_failure, [0.5, 0.5, 0.5])


def test_intact_state_reports_full_service_without_resolving(tmp_path):
    model = HybridPowerModel(write_model(tmp_path / "model.npz"))
    state = model.intact_state()

    np.testing.assert_array_equal(state.component_by_node, [0, 0, 0])
    np.testing.assert_array_equal(state.node_failure, [0.0, 0.0, 0.0])
    assert state.result.hospital_failure[0] == 0.0


def test_length_failures_separate_removed_and_fractionally_unserved_lines(tmp_path):
    model = HybridPowerModel(
        write_model(tmp_path / "model.npz", pmax=25.0, rate=100.0)
    )
    state = model.solve_sample_state(np.asarray([False, False]))
    lengths = model.length_failures(
        np.asarray([], dtype=np.int64), np.asarray([False, False]), state
    )
    assert lengths.initial_km == 0.0
    assert np.isclose(lengths.cascade_km, 5.0)

    failed = np.asarray([True, False])
    state = model.solve_sample_state(failed)
    lengths = model.length_failures(np.asarray([0]), failed, state)
    assert lengths.initial_km == 10.0
    assert lengths.cascade_km == 0.0


def test_zero_service_component_length_is_initial_not_cascade(tmp_path):
    model = HybridPowerModel(
        write_model(tmp_path / "model.npz", pmax=0.0, rate=100.0)
    )
    failed = np.asarray([False, False])
    state = model.solve_sample_state(failed)
    lengths = model.length_failures(
        np.asarray([], dtype=np.int64), failed, state
    )
    assert lengths.initial_km == 10.0
    assert lengths.cascade_km == 0.0
