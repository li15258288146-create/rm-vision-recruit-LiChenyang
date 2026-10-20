import unittest
import numpy as np
from planner import GridMap, search
from robust_boundaries import metrics

class RobustTests(unittest.TestCase):
    def test_weighted_algorithms_agree_and_avoid_risk(self):
        free=np.ones((9,15),bool); cost=np.ones_like(free,float);cost[4,3:12]=40
        a=search(free,(4,1),(4,13),traversal_cost=cost)
        d=search(free,(4,1),(4,13),algorithm='dijkstra',traversal_cost=cost)
        self.assertAlmostEqual(a['cost_cells'],d['cost_cells'])
        self.assertTrue(all(cost[c]==1 for c in a['path']))
    def test_fusion_preserves_unknown_and_obstacles(self):
        a=np.full((40,40),255,np.uint8);b=a.copy();a[10,10]=128;b[25,25]=0
        g=GridMap(np.minimum(a,b),.05,[0,0,0],.25)
        self.assertFalse(g.raw_free[10,10]);self.assertFalse(g.raw_free[25,25])
    def test_invalid_cost_rejected(self):
        free=np.ones((3,3),bool)
        with self.assertRaises(ValueError): search(free,(0,0),(2,2),traversal_cost=np.zeros((3,3)))
    def test_endpoint_uncertainty_not_hidden_by_safe_interior(self):
        g=GridMap(np.full((40,40),255,np.uint8),.05,[0,0,0],.25)
        p=np.array([g.cell_to_world((20,10)),g.cell_to_world((20,20))])
        masks=np.ones((2,40,40),bool);masks[1,20,10]=False
        m=metrics(p,g,masks)
        self.assertEqual(m['whole_path_passing_scenarios'],1)
        self.assertEqual(m['endpoint_passing_scenarios'],[1,2])

if __name__=='__main__':unittest.main()
