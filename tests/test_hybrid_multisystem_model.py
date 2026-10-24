from pathlib import Path
import tempfile
import unittest

import numpy as np

from hybrid_multisystem_model import component_state
from hybrid_power_model import HybridPowerModel
from test_hybrid_power_model import write_model


class HybridMultisystemStateTest(unittest.TestCase):
    def test_component_state_uses_dc_service_and_omits_unretained_nodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = HybridPowerModel(
                write_model(
                    Path(temporary) / "model.npz", pmax=25.0, rate=100.0
                )
            )
            state = model.solve_sample_state(np.asarray([False, False]))

        downstream = component_state(
            model,
            state.component_by_node,
            state.service_by_component,
            frozenset({"gen", "load", "missing", "hospital-a"}),
        )

        self.assertEqual(downstream.component_by_node, {"gen": 0, "load": 0})
        self.assertEqual(
            downstream.probability_by_node, {"gen": 0.5, "load": 0.5}
        )

    def test_intact_state_exposes_zero_failure_for_retained_nodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = HybridPowerModel(write_model(Path(temporary) / "model.npz"))
            state = model.intact_state()

        np.testing.assert_array_equal(state.component_by_node, [0, 0, 0])
        np.testing.assert_array_equal(state.node_failure, [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
