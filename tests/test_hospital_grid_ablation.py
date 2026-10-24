import unittest

import networkx as nx
from shapely.geometry import Point, mapping

from hospital_grid_ablation import (
    add_assigned_hospitals_to_reference_graph,
    attach_hospitals_to_nearest_grid,
)


class AttachHospitalsToNearestGridTest(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_node(
            "grid-a",
            power="gridfinder",
            geometry=mapping(Point(0.0, 0.0)),
        )
        self.graph.add_node(
            "grid-b",
            power="substation",
            geometry=mapping(Point(2.0, 0.0)),
        )
        # A tower is geographically closest to the new hospital but is not a
        # retained connection type in the simplified graph.
        self.graph.add_node(
            "tower",
            power="tower",
            geometry=mapping(Point(1.05, 0.0)),
        )
        self.graph.add_node(
            "hospital-public",
            power="hospital",
            geometry=mapping(Point(0.1, 0.0)),
        )
        self.graph.add_edge(
            "hospital-public",
            "grid-a",
            attribute={"power": "MG-hospital"},
            length=11.1,
            line_fail=0.0,
            tower_fail=0.0,
        )

    def test_adds_only_missing_hospital_to_nearest_connectable_node(self):
        updated, assignments = attach_hospitals_to_nearest_grid(
            self.graph,
            [
                ("hospital-public", Point(0.1, 0.0)),
                ("hospital-solar", Point(1.1, 0.0)),
            ],
        )

        self.assertNotIn("hospital-solar", self.graph)
        self.assertIn("hospital-solar", updated)
        self.assertEqual(set(updated.neighbors("hospital-solar")), {"grid-b"})
        self.assertEqual(updated.nodes["hospital-solar"]["power"], "hospital")

        connector = updated.edges["hospital-solar", "grid-b"]
        self.assertEqual(connector["attribute"]["power"], "MG-hospital")
        self.assertEqual(connector["line_fail"], 0.0)
        self.assertEqual(connector["tower_fail"], 0.0)
        self.assertTrue(connector["ablation"])
        self.assertGreater(connector["length"], 0.0)

        by_hospital = {
            assignment.hospital_node_id: assignment
            for assignment in assignments
        }
        self.assertEqual(by_hospital["hospital-public"].source, "existing")
        self.assertEqual(by_hospital["hospital-public"].grid_node_id, "grid-a")
        self.assertEqual(by_hospital["hospital-solar"].source, "ablation")
        self.assertEqual(by_hospital["hospital-solar"].grid_node_id, "grid-b")

        reference = add_assigned_hospitals_to_reference_graph(
            self.graph,
            [
                ("hospital-public", Point(0.1, 0.0)),
                ("hospital-solar", Point(1.1, 0.0)),
            ],
            assignments,
        )
        self.assertNotIn("hospital-solar", self.graph)
        self.assertEqual(set(reference.neighbors("hospital-solar")), {"grid-b"})
        self.assertTrue(reference.edges["hospital-solar", "grid-b"]["ablation"])

    def test_rejects_existing_hospital_geometry_mismatch(self):
        with self.assertRaisesRegex(ValueError, "row identity may have changed"):
            attach_hospitals_to_nearest_grid(
                self.graph,
                [("hospital-public", Point(1.5, 0.0))],
            )

    def test_rejects_duplicate_hospital_ids(self):
        with self.assertRaisesRegex(ValueError, "Duplicate hospital node id"):
            attach_hospitals_to_nearest_grid(
                self.graph,
                [
                    ("hospital-solar", Point(1.1, 0.0)),
                    ("hospital-solar", Point(1.2, 0.0)),
                ],
            )

    def test_requires_a_geocoded_connectable_node(self):
        graph = nx.Graph()
        graph.add_node(
            "tower",
            power="tower",
            geometry=mapping(Point(0.0, 0.0)),
        )

        with self.assertRaisesRegex(ValueError, "no geocoded connectable nodes"):
            attach_hospitals_to_nearest_grid(
                graph,
                [("hospital-solar", Point(0.0, 0.0))],
            )


if __name__ == "__main__":
    unittest.main()
