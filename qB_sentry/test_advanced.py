import unittest
import numpy as np


class TerrainTests(unittest.TestCase):
    def test_slope_is_driveable_step_is_not(self):
        from terrain_advanced import classify_surface
        y,x=np.indices((60,80));ramp=x*.05*.2
        free,slope,edge=classify_surface(ramp,np.zeros_like(x,bool),.05)
        self.assertTrue(free[20:40,20:60].all())
        step=np.where(x<40,0.,.3)
        free,_,edge=classify_surface(step,np.zeros_like(x,bool),.05)
        self.assertFalse(free[:,39:41].any())
        self.assertTrue(free[20:40,10:20].all())

    def test_unknown_hole_stays_unknown(self):
        from terrain_advanced import fill_small_holes
        z=np.zeros((40,40));z[10,10]=np.nan;z[20:30,20:30]=np.nan
        filled,mask=fill_small_holes(z,.05)
        self.assertEqual(filled[10,10],0)
        self.assertTrue(np.isnan(filled[24,24]))


class DecisionTests(unittest.TestCase):
    def test_occlusion_and_unreachable_goal(self):
        from decision_sim import visible,route
        from planner import GridMap
        im=np.full((30,30),255,np.uint8);im[:,15]=0
        grid=GridMap(im,.1,[0,0,0],.25)
        self.assertFalse(visible(grid,[.7,1.5],[2.3,1.5]))
        self.assertIsNone(route(grid,[.7,1.5],[2.3,1.5]))

    def test_survival_priority_and_hysteresis(self):
        from decision_sim import decide
        context=dict(hp=30,ammo=0,enemy_distance=2.,remaining=10,outpost_alive=False)
        self.assertEqual(decide('ENGAGE',**context),'RETREAT')
        context['hp']=80
        self.assertEqual(decide('RETREAT',**context),'RETREAT')
        context['hp']=160
        self.assertEqual(decide('RETREAT',**context),'RESUPPLY')

    def test_all_tactical_inputs(self):
        from decision_sim import decide
        context=dict(hp=200,ammo=100,enemy_distance=20.,remaining=120,outpost_alive=True)
        self.assertEqual(decide('PATROL',**context),'PATROL')
        self.assertEqual(decide('PATROL',**(context|dict(enemy_distance=3.))),'ENGAGE')
        self.assertEqual(decide('PATROL',**(context|dict(outpost_alive=False))),'DEFEND')
        self.assertEqual(decide('PATROL',**(context|dict(remaining=15))),'DEFEND')
        self.assertEqual(decide('PATROL',**(context|dict(ammo=3))),'RESUPPLY')


class RegistrationTests(unittest.TestCase):
    def test_known_rigid_transform_recovered(self):
        from register_map import transform,register_points
        points=np.random.default_rng(9).uniform(-3,3,(100,2))
        target=transform(points,.25,[1.1,-.7])
        result=register_points(points,target,.23,np.array([1.,-.6]))
        recovered=transform(points,result.x[0],result.x[1:])
        self.assertLess(np.max(np.linalg.norm(recovered-target,axis=1)),.003)

if __name__=='__main__':unittest.main()
