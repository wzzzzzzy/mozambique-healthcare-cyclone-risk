import math

import networkx as nx
import numpy as np

from build_matpower_case import (
    BASE_MVA,
    BranchRecord,
    assumed_rate_mw,
    build_branches,
    circuit_count,
    component_load_targets,
    dc_opf,
    find_feasible_scale,
    voltage_kv_from_value,
)


def branch(rate_mw: float = 100.0, x_pu: float = 0.1) -> BranchRecord:
    return BranchRecord(
        uid=1,
        from_node="a",
        to_node="b",
        edge_key="0",
        edge_type="line",
        from_bus=1,
        to_bus=2,
        length_km=10,
        voltage_kv=110,
        circuits=1,
        x_pu=x_pu,
        rate_a_mw=rate_mw,
    )


def test_voltage_and_circuit_parsing():
    assert voltage_kv_from_value("110000; 33000") == 110
    assert voltage_kv_from_value(66) == 66
    assert circuit_count({"circuits": "2"}) == 2
    assert circuit_count({"cables": "6"}) == 2
    assert assumed_rate_mw(110) == 250


def test_parallel_circuit_equivalent_impedance_and_rating():
    graph = nx.MultiGraph()
    graph.add_node("a", geometry={"coordinates": [32.0, -20.0]}, power="substation")
    graph.add_node("b", geometry={"coordinates": [33.0, -20.0]}, power="substation")
    graph.add_edge(
        "a",
        "b",
        attribute={"power": "line", "voltage": "110000", "circuits": "2"},
    )
    record = build_branches(graph, {"a": 1, "b": 2}, {1}, {"a": 1, "b": 1})[0]
    expected_x = 0.4 * record.length_km / 2 * BASE_MVA / 110**2
    assert math.isclose(record.x_pu, expected_x)
    assert record.rate_a_mw == 500


def test_dc_opf_uses_matpower_per_unit_flow_equation():
    loads = np.array([0.0, 50.0])
    generators = [{"bus": 1, "status": 1, "pmax_mw": 100.0, "cost": 10.0}]
    result = dc_opf(loads, [branch()], generators, np.array([3, 1]))
    assert result["success"]
    assert math.isclose(result["pg"][0], 50.0, abs_tol=1e-7)
    assert math.isclose(result["flows"][0], 50.0, abs_tol=1e-7)
    assert math.isclose(result["theta"][1], -0.05, abs_tol=1e-7)


def test_feasibility_scaling_is_limited_by_branch_rating():
    loads = np.array([0.0, 50.0])
    generators = [{"bus": 1, "status": 1, "pmax_mw": 100.0, "cost": 10.0}]
    scale, result = find_feasible_scale(
        loads, [branch(rate_mw=40)], generators, np.array([3, 1])
    )
    assert result["success"]
    assert math.isclose(scale, 0.8, rel_tol=0, abs_tol=1e-8)


def test_component_load_allocation_respects_reserve_cap():
    targets = component_load_targets(
        global_target=50,
        activity={1: 1, 2: 1},
        capacity={1: 10, 2: 100},
        reserve_fraction=0.85,
    )
    assert math.isclose(targets[1], 8.5)
    assert math.isclose(targets[2], 41.5)
