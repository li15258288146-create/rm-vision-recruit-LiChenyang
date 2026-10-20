import unittest
import numpy as np
from aim import filter_positions
from run_aim import evaluate

class PredictionTests(unittest.TestCase):
    def test_no_future_samples_are_reported_without_nan(self):
        _,_,m=evaluate(np.full((50,3),np.nan),np.full((50,3),np.nan))
        self.assertEqual(m['evaluated_frames'],0)
        self.assertIsNone(m['mean_euclidean_error_m'])
    def test_half_frame_future_alignment(self):
        obs=np.arange(100)[:,None]/150*np.array([[2.,-.2,.5]])
        pred=obs+.25*np.array([2.,-.2,.5])
        ids,future,m=evaluate(obs,pred)
        self.assertEqual(len(ids),62)
        np.testing.assert_allclose(pred[ids],future,atol=1e-12)
    def test_filter_is_causal(self):
        obs=np.c_[np.arange(180)/150,np.zeros(180),np.ones(180)*3]
        changed=obs.copy();changed[100:]+=20
        a=filter_positions(obs)[1];b=filter_positions(changed)[1]
        np.testing.assert_array_equal(a[:100],b[:100])
    def test_outlier_and_long_gap(self):
        obs=np.tile([0.,0.,3.],(180,1));obs[40]=[100,100,100];obs[50:140]=np.nan
        filtered,pred,accepted=filter_positions(obs)
        self.assertFalse(accepted[40]);self.assertLess(np.linalg.norm(filtered[40]-[0,0,3]),.01)
        self.assertTrue(np.isnan(pred[130]).all());self.assertTrue(accepted[140])
if __name__=='__main__':unittest.main()
