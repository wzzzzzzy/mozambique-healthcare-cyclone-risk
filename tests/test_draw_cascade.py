import tempfile
import unittest
from pathlib import Path

from draw_cascade import (
    PNG_SIGNATURE,
    TRACK_COUNTS,
    TOTAL_TRACKS,
    global_to_track,
    output_path,
    storm_folder,
    valid_png,
)


class CascadePathTests(unittest.TestCase):
    def test_catalogue_size(self):
        self.assertEqual(TRACK_COUNTS, (290, 274, 263, 276, 273, 279, 251, 264, 242, 291))
        self.assertEqual(TOTAL_TRACKS, 2703)

    def test_global_track_mapping(self):
        self.assertEqual(global_to_track(0), (0, 0))
        self.assertEqual(global_to_track(289), (0, 289))
        self.assertEqual(global_to_track(290), (1, 0))
        self.assertEqual(global_to_track(2702), (9, 290))
        with self.assertRaises(ValueError):
            global_to_track(-1)
        with self.assertRaises(ValueError):
            global_to_track(2703)

    def test_paths(self):
        self.assertEqual(storm_folder(9), "STORM_DATA_IBTRACS_SI_1000_YEARS_9")
        self.assertEqual(
            output_path(Path("/maps"), 9, 86),
            Path("/maps/dataset_9/track_0086.png"),
        )

    def test_png_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.png"
            path.write_bytes(PNG_SIGNATURE + b"x" * 10_000)
            self.assertTrue(valid_png(path))
            path.write_bytes(b"invalid")
            self.assertFalse(valid_png(path))


if __name__ == "__main__":
    unittest.main()
