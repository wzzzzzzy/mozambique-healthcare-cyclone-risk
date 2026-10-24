import unittest

import numpy as np
import pandas as pd
import networkx as nx

from simulation_on_wind_graph import (
    draw_edge_random_values,
    generate_df_edge,
)


class MonteCarloPairingTest(unittest.TestCase):
    def test_ablation_edges_are_ordered_after_shared_edges(self):
        graph = nx.MultiGraph()
        graph.add_edge(
            'grid-a',
            'hospital-solar',
            line_fail=0.4,
            tower_fail=0.0,
            ablation=True,
        )
        graph.add_edge(
            'grid-a',
            'grid-b',
            line_fail=0.2,
            tower_fail=0.1,
        )

        edges = generate_df_edge(graph)

        self.assertEqual(edges['ablation'].tolist(), [False, True])
        self.assertEqual(edges['line_fail'].tolist(), [0.2, 0.4])

    def test_ablation_edges_do_not_shift_shared_edge_draws(self):
        public_edges = pd.DataFrame({'ablation': [False, False, False]})
        all_grid_edges = pd.DataFrame(
            {'ablation': [False, False, False, True, True]}
        )
        public_shared_rng = np.random.RandomState(1234)
        public_ablation_rng = np.random.RandomState(1235)
        all_grid_shared_rng = np.random.RandomState(1234)
        all_grid_ablation_rng = np.random.RandomState(1235)

        for _ in range(5):
            public_draws = draw_edge_random_values(
                public_edges,
                public_shared_rng,
                public_ablation_rng,
            )
            all_grid_draws = draw_edge_random_values(
                all_grid_edges,
                all_grid_shared_rng,
                all_grid_ablation_rng,
            )
            np.testing.assert_array_equal(public_draws, all_grid_draws[:3])


if __name__ == '__main__':
    unittest.main()
