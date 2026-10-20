import unittest
import numpy as np
from planner import search, GridMap


class PlannerTests(unittest.TestCase):
    def test_open_grid_optimal(self):
        free = np.ones((7, 7), bool)
        a = search(free, (0, 0), (6, 4), 'astar')
        d = search(free, (0, 0), (6, 4), 'dijkstra')
        self.assertAlmostEqual(a['cost_cells'], 4 * 2**.5 + 2)
        self.assertAlmostEqual(a['cost_cells'], d['cost_cells'])

    def test_no_corner_cutting(self):
        free = np.eye(3, dtype=bool)
        self.assertIsNone(search(free, (0, 0), (2, 2))['path'])

    def test_same_cell(self):
        result = search(np.ones((2, 2), bool), (1, 1), (1, 1))
        self.assertEqual(result['path'], [(1, 1)])

    def test_world_grid_roundtrip(self):
        grid = GridMap(np.full((10, 20), 255, np.uint8), .05, [-3.58, -9.44, .3], .25, radius=0)
        for cell in [(0, 0), (9, 19), (3, 8)]:
            self.assertEqual(grid.world_to_cell(grid.cell_to_world(cell)), cell)

    def test_checker_clearance_convention(self):
        image = np.full((30, 30), 255, np.uint8)
        image[10, 10] = 0
        grid = GridMap(image, .05, [0, 0, 0], .25, radius=.30)
        self.assertTrue(grid.free[14, 15])  # sqrt(4^2+5^2)*.05 = .320 m
        self.assertFalse(grid.free[10, 15])  # .25 m


if __name__ == '__main__':
    unittest.main()
