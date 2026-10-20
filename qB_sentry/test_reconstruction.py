import unittest
import numpy as np


class ReconstructionTests(unittest.TestCase):
    def test_unknown_encoding_rejected_by_config_threshold(self):
        from reconstruct_map import UNKNOWN
        from planner import GridMap
        image=np.full((40,40),255,np.uint8);image[:,20]=UNKNOWN
        grid=GridMap(image,.05,[0,0,0],.25)
        self.assertFalse(grid.raw_free[:,20].any())

    def test_supported_hole_but_not_wide_gap(self):
        from reconstruct_map import interpolate_surface
        y,x=np.indices((80,80));z=x*.002
        z[30:36,30:36]=np.nan;z[50:75,50:75]=np.nan
        result,mask=interpolate_surface(z,.05,max_edge=.8)
        self.assertAlmostEqual(result[32,32],.064,places=5)
        self.assertTrue(np.isnan(result[62,62]))

    def test_no_interpolation_across_height_discontinuity(self):
        from reconstruct_map import interpolate_surface
        y,x=np.indices((40,40));z=np.where(x<20,0.,.3);z[:,19:21]=np.nan
        result,_=interpolate_surface(z,.05)
        self.assertTrue(np.isnan(result[20,20]))

    def test_small_step_gap_gets_height_before_slope_classification(self):
        from reconstruct_map import interpolate_surface
        y,x=np.indices((40,40));z=np.where(x<20,0.,.10);z[:,20]=np.nan
        result,_=interpolate_surface(z,.05)
        self.assertTrue(np.isfinite(result[20,20]))

    def test_small_step_allowed_large_step_blocked(self):
        from reconstruct_map import terrain_free
        y,x=np.indices((40,40));obstacles=np.zeros(x.shape,bool)
        free,_,_=terrain_free(np.where(x<20,0.,.10),obstacles,.05)
        self.assertTrue(free[10:30,18:22].all())
        free,_,_=terrain_free(np.where(x<20,0.,.30),obstacles,.05)
        self.assertFalse(free[:,19:21].any())

    def test_wall_never_removed_by_surface_interpolation(self):
        from reconstruct_map import terrain_free
        z=np.zeros((40,40));wall=np.zeros(z.shape,bool);wall[:,20]=True
        free,_,_=terrain_free(z,wall,.05)
        self.assertFalse(free[:,20].any())

    def test_layer_reconstruction_does_not_overwrite_high_surface(self):
        from reconstruct_map import fill_planar_layers
        y,x=np.indices((90,90));z=np.zeros((90,90));z[35:55,35:55]=.6
        z[30:35,30:35]=np.nan
        result,mask,levels=fill_planar_layers(z,z,.05)
        self.assertTrue(np.isfinite(result[32,32]))
        self.assertEqual(result[45,45],.6)

    def test_layer_reconstruction_leaves_large_void_unknown(self):
        from reconstruct_map import fill_planar_layers
        z=np.zeros((120,120));z[30:90,30:90]=np.nan
        result,_,_=fill_planar_layers(z,z,.05)
        self.assertTrue(np.isnan(result[60,60]))

    def test_resampling_outside_is_unknown_not_free(self):
        from refine_registration import resample
        from planner import GridMap
        cfg=dict(resolution=.05,origin=[0,0,0])
        image=resample(np.full((10,10),255,np.uint8),cfg,(40,40),cfg,0,[0,0])
        grid=GridMap(image,.05,[0,0,0],.25)
        self.assertEqual(int(grid.raw_free.sum()),100)

if __name__=='__main__':unittest.main()
