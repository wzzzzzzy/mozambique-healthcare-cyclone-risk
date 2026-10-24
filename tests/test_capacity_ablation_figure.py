import io
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from shapely.geometry import Point

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from capacity_ablation_figure import (  # noqa: E402
    build_comparison_table,
    combine_hybrid_baseline,
    expected_track_columns,
    load_power_summary,
    plot_ablation_figure,
)


TRACK_COUNTS = (290, 274, 263, 276, 273, 279, 251, 264, 242, 291)


def _by_node(frame):
    if frame.index.name == "node_id":
        return frame
    return frame.set_index("node_id", drop=False)


def _small_summaries(with_geometry=False):
    tracks = ["0_0", "0_1"]
    public = pd.DataFrame(
        {
            "node_id": ["node/public", "node/overlap"],
            "facility_i": [101, 102],
            tracks[0]: [0.2, 0.3],
            tracks[1]: [0.4, 0.5],
        }
    )
    all_grid = pd.DataFrame(
        {
            "node_id": ["node/public", "node/overlap", "node/solar-only"],
            "facility_i": [101, 102, 103],
            tracks[0]: [0.5, 0.1, 0.2],
            tracks[1]: [0.6, 0.2, 0.3],
        }
    )
    solar = pd.DataFrame(
        {
            "node_id": ["node/overlap", "node/solar-only"],
            "facility_i": [102, 103],
            tracks[0]: [0.95, 0.8],
            tracks[1]: [0.95, 0.6],
        }
    )

    if with_geometry:
        points = {
            "node/public": Point(32.0, -18.0),
            "node/overlap": Point(33.0, -19.0),
            "node/solar-only": Point(34.0, -20.0),
        }
        all_grid = gpd.GeoDataFrame(
            all_grid,
            geometry=all_grid["node_id"].map(points),
            crs="EPSG:4326",
        )
    return public, all_grid, solar


class ExpectedTrackColumnsTest(unittest.TestCase):
    def test_matches_all_ten_storm_sets_in_canonical_order(self):
        columns = list(expected_track_columns())

        self.assertEqual(len(columns), 2703)
        self.assertEqual(columns[:3], ["0_0", "0_1", "0_2"])
        self.assertEqual(columns[289:292], ["0_289", "1_0", "1_1"])
        self.assertEqual(columns[-3:], ["9_288", "9_289", "9_290"])
        self.assertEqual(
            columns,
            [
                f"{storm_set}_{track}"
                for storm_set, count in enumerate(TRACK_COUNTS)
                for track in range(count)
            ],
        )


class LoadPowerSummaryTest(unittest.TestCase):
    def test_accepts_and_canonically_orders_complete_2703_track_schema(self):
        tracks = list(expected_track_columns())
        frame = pd.DataFrame(
            {
                "node_id": ["node/one"],
                "facility_i": [7],
                **{track: [0.25] for track in reversed(tracks)},
            }
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "summary.csv"
            frame.to_csv(path, index=False)
            loaded = load_power_summary(path)

        self.assertEqual(
            list(loaded.columns), ["node_id", "facility_i", *tracks]
        )
        np.testing.assert_allclose(loaded[tracks].to_numpy(), 0.25)

    def test_rejects_a_missing_track_when_complete_results_are_required(self):
        tracks = list(expected_track_columns())
        missing_track = "4_17"
        frame = pd.DataFrame(
            {
                "node_id": ["node/one"],
                "facility_i": [7],
                **{
                    track: [0.25]
                    for track in tracks
                    if track != missing_track
                },
            }
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "incomplete.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(
                ValueError, rf"(?i)(missing|complete|2703).*{missing_track}"
            ):
                load_power_summary(path, require_complete=True)

    def test_accepts_only_explicit_minus_one_unavailable_sentinel(self):
        tracks = list(expected_track_columns())
        frame = pd.DataFrame(
            {
                "node_id": ["node/one"],
                "facility_i": [7],
                **{track: [-1.0] for track in tracks},
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "hybrid.csv"
            frame.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_power_summary(path)
            loaded = load_power_summary(path, allow_unavailable=True)
        self.assertTrue(loaded[tracks].eq(-1.0).all().all())


class CombineHybridBaselineTest(unittest.TestCase):
    def test_uses_all_grid_cohort_with_public_override_and_solar_only_rows(self):
        public, all_grid, solar = _small_summaries()

        baseline = _by_node(
            combine_hybrid_baseline(public, all_grid, solar)
        )

        self.assertEqual(
            set(baseline.index),
            {"node/public", "node/overlap", "node/solar-only"},
        )
        self.assertEqual(baseline.loc["node/public", "facility_i"], 101)
        self.assertEqual(baseline.loc["node/solar-only", "facility_i"], 103)
        np.testing.assert_allclose(
            baseline.loc["node/public", ["0_0", "0_1"]].astype(float),
            [0.2, 0.4],
        )
        # The overlap deliberately has very different solar values. Public
        # outcomes must win because this hospital is part of the public cohort.
        np.testing.assert_allclose(
            baseline.loc["node/overlap", ["0_0", "0_1"]].astype(float),
            [0.3, 0.5],
        )
        np.testing.assert_allclose(
            baseline.loc["node/solar-only", ["0_0", "0_1"]].astype(float),
            [0.8, 0.6],
        )

    def test_rejects_an_all_grid_hospital_missing_from_both_baselines(self):
        public, all_grid, solar = _small_summaries()
        solar = solar[solar["node_id"] != "node/solar-only"]

        with self.assertRaisesRegex(
            ValueError, r"(?i)(missing|baseline).*node/solar-only"
        ):
            combine_hybrid_baseline(public, all_grid, solar)


class BuildComparisonTableTest(unittest.TestCase):
    def test_computes_paired_delta_as_all_grid_minus_baseline(self):
        public, all_grid, solar = _small_summaries(with_geometry=True)

        comparison = _by_node(
            build_comparison_table(public, all_grid, solar)
        )

        self.assertIsInstance(comparison, gpd.GeoDataFrame)
        self.assertEqual(comparison.crs.to_string(), "EPSG:4326")
        self.assertEqual(comparison.loc["node/public", "geometry"], Point(32, -18))
        self.assertAlmostEqual(
            comparison.loc["node/public", "baseline_mean"], 0.3
        )
        self.assertAlmostEqual(
            comparison.loc["node/public", "all_grid_mean"], 0.55
        )
        self.assertAlmostEqual(comparison.loc["node/public", "delta"], 0.25)
        self.assertGreater(comparison.loc["node/public", "delta"], 0.0)

        self.assertAlmostEqual(
            comparison.loc["node/solar-only", "baseline_mean"], 0.7
        )
        self.assertAlmostEqual(
            comparison.loc["node/solar-only", "all_grid_mean"], 0.25
        )
        self.assertAlmostEqual(
            comparison.loc["node/solar-only", "delta"], -0.45
        )
        self.assertLess(comparison.loc["node/solar-only", "delta"], 0.0)

    def test_preserves_unavailable_public_hospital_in_reporting_cohort(self):
        public, all_grid, solar = _small_summaries(with_geometry=True)
        public.loc[public["node_id"].eq("node/public"), ["0_0", "0_1"]] = -1.0

        comparison = _by_node(build_comparison_table(public, all_grid, solar))

        self.assertEqual(comparison.loc["node/public", "baseline_mean"], -1.0)
        self.assertTrue(np.isnan(comparison.loc["node/public", "delta"]))
        self.assertAlmostEqual(comparison.loc["node/public", "all_grid_mean"], 0.55)


class PlotAblationFigureTest(unittest.TestCase):
    def test_renders_headlessly(self):
        public, all_grid, solar = _small_summaries(with_geometry=True)
        comparison = build_comparison_table(public, all_grid, solar)

        plotted = plot_ablation_figure(comparison, boundary_gdf=None)
        figure = plotted[0] if isinstance(plotted, tuple) else plotted
        try:
            self.assertIsInstance(figure, Figure)
            self.assertGreaterEqual(len(figure.axes), 4)
            figure.canvas.draw()
            legend = next(
                axis.get_legend()
                for axis in figure.axes
                if axis.get_legend() is not None
            )
            renderer = figure.canvas.get_renderer()
            legend_bounds = legend.get_window_extent(renderer)
            title_bounds = [
                axis._left_title.get_window_extent(renderer)
                for axis in figure.axes
                if axis.get_title(loc="left")
            ]
            self.assertTrue(title_bounds)
            self.assertFalse(
                any(legend_bounds.overlaps(bounds) for bounds in title_bounds)
            )
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png")
            self.assertGreater(buffer.tell(), 0)
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
