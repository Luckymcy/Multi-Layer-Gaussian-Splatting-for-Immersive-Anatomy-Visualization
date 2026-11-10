import unittest
import numpy as np

from prepare_fat_dataset import rasterize, sampling_mask, patch_positions, box_iou, near_duplicate


class PreparationTests(unittest.TestCase):
    def test_circle_exclusion_wins_independent_of_order(self):
        shapes = [{'label': 'exclude', 'shape_type': 'circle', 'points': [[30, 30], [35, 30]]},
                  {'label': 'fat', 'shape_type': 'polygon', 'points': [[10, 10], [50, 10], [50, 50], [10, 50]]}]
        fat, exclude, valid, _ = rasterize(shapes, (64, 64))
        self.assertTrue(fat[30, 30])
        self.assertTrue(exclude[30, 30])
        self.assertFalse(valid[30, 30])
        self.assertTrue(valid[15, 15])
        np.testing.assert_array_equal(valid, rasterize(shapes[::-1], (64, 64))[2])

    def test_safe_mask_boundary_and_quality_filter(self):
        rgb = np.full((25, 25, 3), 100, dtype=np.uint8)
        rgb[12, 12] = 255
        valid = np.ones((25, 25), dtype=bool)
        valid[5:8, 5:8] = False
        safe, quality = sampling_mask(rgb, valid, 2, 250, 220, 1)
        self.assertFalse(safe[0, 12])
        self.assertFalse(safe[4, 6])
        self.assertFalse(safe[12, 12])
        self.assertTrue(safe[20, 20])
        self.assertTrue(quality[11:14, 11:14].all())
        self.assertTrue(valid[12, 12])  # Manual input never changed.

    def test_integral_sampling_and_exclusive_coordinates(self):
        mask = np.ones((32, 32), dtype=bool)
        mask[4, 4] = False
        boxes = patch_positions(mask, 8, 4)
        self.assertNotIn((0, 0, 8, 8), boxes)
        self.assertIn((24, 24, 32, 32), boxes)
        for x0, y0, x1, y1 in boxes:
            self.assertTrue(mask[y0:y1, x0:x1].all())
        self.assertEqual(box_iou((0, 0, 8, 8), (0, 0, 8, 8)), 1)
        self.assertEqual(box_iou((0, 0, 8, 8), (8, 8, 16, 16)), 0)

    def test_duplicate_threshold_and_invalid_label(self):
        a = np.zeros((16, 16, 3), dtype=np.float32)
        self.assertTrue(near_duplicate(a + 2, [a], 3))
        self.assertFalse(near_duplicate(a + 4, [a], 3))
        with self.assertRaises(ValueError):
            rasterize([{'label': 'unknown', 'points': [[1, 1], [2, 1], [2, 2]]}], (8, 8))


if __name__ == '__main__':
    unittest.main()
