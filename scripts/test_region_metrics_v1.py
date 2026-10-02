import unittest
import numpy as np
from region_metrics_v1 import pixel_statistics, region_statistics


class RegionMetricsTests(unittest.TestCase):
    def test_error_outside_region_does_not_affect_masked_error(self):
        a=np.zeros((20,20,3),dtype=np.uint8);b=a.copy();m=np.zeros((20,20),bool);m[5:15,5:15]=True
        a[~m]=255
        self.assertEqual(pixel_statistics(a,b,m)['mae'],0)
        self.assertEqual(pixel_statistics(a,b,m)['psnr_status'],'perfect_infinite')
        self.assertEqual(pixel_statistics(a,b,~m)['psnr'],0)

    def test_empty_region_is_not_perfect(self):
        a=np.zeros((20,20,3),dtype=np.uint8)
        self.assertEqual(pixel_statistics(a,a,np.zeros((20,20),bool))['psnr_status'],'empty_region')

    def test_thin_mask_ssim_is_unavailable(self):
        a=np.zeros((40,40,3),dtype=np.uint8);m=np.zeros((40,40),bool);m[20,20]=True
        result=region_statistics(None,a,a,m,np.ones((40,40)))
        self.assertIsNone(result['ssim']);self.assertIsNone(result['bbox_lpips'])


if __name__=='__main__':unittest.main()
