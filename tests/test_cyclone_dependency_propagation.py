import tempfile
from pathlib import Path
import unittest

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point

from propagate_cyclone_dependencies import (
    CommunicationDependencyModel,
    ComponentState,
    WaterDependencyModel,
    build_dependency_models,
    component_path,
    evaluate_communication_dependencies,
    evaluate_water_dependencies,
    load_component_state,
    _selected_services,
    _result_frame,
)


class WaterDependencyEvaluationTest(unittest.TestCase):
    def test_combines_distinct_components_once_and_preserves_sentinels(self):
        model = WaterDependencyModel(
            hospital_ids=("h_combined", "h_missing", "h_unevaluated"),
            upstream_power_nodes=(
                ("p1", "p2", "p3"),
                ("missing",),
                None,
            ),
            power_nodes=("p1", "p2", "p3", "missing"),
        )
        state = ComponentState(
            probability_by_node={"p1": 0.2, "p2": 0.2, "p3": 0.5},
            component_by_node={"p1": 0, "p2": 0, "p3": 1},
        )

        result = evaluate_water_dependencies(model, state)

        self.assertAlmostEqual(result[0], 0.6)
        self.assertEqual(result[1], -1.0)
        self.assertEqual(result[2], -1.0)

    def test_result_summary_excludes_unevaluated_samples(self):
        result = _result_frame(
            ("h_missing", "h_evaluated"),
            (
                np.asarray([-1.0, 0.2]),
                np.asarray([-1.0, 0.4]),
            ),
        )

        self.assertEqual(result.loc[0, "mean"], -1.0)
        self.assertEqual(result.loc[0, "evaluated_sample_num"], 0)
        self.assertAlmostEqual(result.loc[1, "mean"], 0.3)
        self.assertEqual(result.loc[1, "evaluated_sample_num"], 2)


class CommunicationDependencyEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.model = CommunicationDependencyModel(
            hospital_ids=("h_redundant", "h_missing", "h_unevaluated"),
            predecessor_towers=(("t1", "t2"), ("t3",), None),
            tower_power_source=(
                ("t1", "p1"),
                ("t2", "p2"),
                ("t3", "missing"),
            ),
            power_nodes=("p1", "p2", "missing"),
        )

    def test_does_not_multiply_a_shared_component_twice(self):
        state = ComponentState(
            probability_by_node={"p1": 0.2, "p2": 0.2},
            component_by_node={"p1": 0, "p2": 0},
        )

        result = evaluate_communication_dependencies(self.model, state)

        self.assertAlmostEqual(result[0], 0.2)
        self.assertEqual(result[1], 1.0)
        self.assertEqual(result[2], -1.0)

    def test_multiplies_independent_tower_component_failures(self):
        state = ComponentState(
            probability_by_node={"p1": 0.2, "p2": 0.5},
            component_by_node={"p1": 0, "p2": 1},
        )

        result = evaluate_communication_dependencies(self.model, state)

        self.assertAlmostEqual(result[0], 0.1)


class ComponentInputContractTest(unittest.TestCase):
    def test_service_selection_supports_water_only_rerun(self):
        self.assertEqual(_selected_services("water"), ("water",))
        self.assertEqual(_selected_services("both"), ("water", "comm"))
        with self.assertRaises(ValueError):
            _selected_services("electricity")

    def test_filters_irrelevant_nodes_and_reads_safe_literals(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "component.csv"
            pd.DataFrame({
                "ccs": ["['p1', 'irrelevant']", "['p2']"],
                "fail_prob": [0.25, 0.5],
            }).to_csv(path, index=False)

            state = load_component_state(path, {"p1", "p2"})

        self.assertEqual(state.probability_by_node, {"p1": 0.25, "p2": 0.5})
        self.assertEqual(state.component_by_node, {"p1": 0, "p2": 1})

    def test_rejects_executable_component_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "component.csv"
            pd.DataFrame({
                "ccs": ["__import__('os').system('false')"],
                "fail_prob": [0.25],
            }).to_csv(path, index=False)

            with self.assertRaises(ValueError):
                load_component_state(path, {"p1"})

    def test_component_path_is_locked_to_non_ablation_public_scenario(self):
        path = component_path(
            Path("/components"),
            "STORM_DATA_IBTRACS_SI_1000_YEARS_4",
            12,
            3,
            25,
        )
        self.assertEqual(
            path.name,
            "STORM_DATA_IBTRACS_SI_1000_YEARS_4__capacity_v2_public_12_3_25.csv",
        )
        self.assertNotIn("all_grid", str(path))


class DependencyModelConstructionTest(unittest.TestCase):
    def test_graph_absent_service_hospitals_remain_unevaluated(self):
        hospitals = gpd.GeoDataFrame(
            {"water_sour": [1], "has_tel": [1]},
            geometry=[Point(35, -18)],
            crs="EPSG:4326",
        )

        models = build_dependency_models(
            hospitals,
            nx.DiGraph(),
            nx.DiGraph(),
        )

        self.assertEqual(models.water.upstream_power_nodes, (None,))
        self.assertEqual(models.communication.predecessor_towers, (None,))


if __name__ == "__main__":
    unittest.main()
