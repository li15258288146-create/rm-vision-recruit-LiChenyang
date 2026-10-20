import unittest
import cv2
import numpy as np
from aim import detect, pose, OBJECT_POINTS

class AimTests(unittest.TestCase):
    def test_red_pair(self):
        image=np.zeros((200,300,3),np.uint8)
        cv2.rectangle(image,(95,65),(101,105),(0,30,255),-1)
        cv2.rectangle(image,(193,65),(199,105),(0,30,255),-1)
        detections,_=detect(image)
        self.assertEqual(len(detections),1)
        p=detections[0]['points']
        self.assertLess(p[0,1],p[1,1]);self.assertLess(p[0,0],p[3,0])

    def test_pnp_units(self):
        K=np.array([[1000.,0,640],[0,1000.,360],[0,0,1]])
        t=np.array([.1,.05,3.]);d=np.zeros(5)
        points,_=cv2.projectPoints(OBJECT_POINTS,np.array([.1,.2,0]),t,K,d)
        result=pose(points.reshape(4,2),K,d)
        np.testing.assert_allclose(result['position'],t,atol=1e-5)

if __name__=='__main__':unittest.main()
