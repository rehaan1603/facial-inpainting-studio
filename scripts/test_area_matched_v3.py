import unittest
import numpy as np
from area_matched_v3 import canonical,translate,corrupt


class AreaMatchedTests(unittest.TestCase):
    def test_counts_and_translation(self):
        for count in [1966,3932,6554]:
            e=round(count*.2);true,masks=canonical(256,count,e,42)
            for name,m in masks.items():
                self.assertEqual(int((m&~true).sum()),e if name in ['over','mixed'] else 0)
                self.assertEqual(int((~m&true).sum()),e if name in ['under','mixed'] else 0)
                shifted=translate(m,7,-11)
                self.assertIsNotNone(shifted)
                np.testing.assert_array_equal(translate(shifted,-7,11),m)
            self.assertEqual(int(true.sum()),count)
            self.assertIsNone(translate(true,250,0))

    def test_hidden_pixel_independence(self):
        true,_=canonical(256,1966,393,5)
        zero=np.zeros((256,256,3),np.float32);one=np.ones_like(zero)
        a=corrupt(zero,true,19);b=corrupt(one,true,19)
        np.testing.assert_array_equal(a[true],b[true])
        np.testing.assert_array_equal(b[~true],one[~true])

    def test_matched_texture(self):
        true,_=canonical(256,3932,786,12);shifted=translate(true,9,-30)
        target=np.zeros((256,256,3),np.float32)
        a=corrupt(target,true,22,[128,128]);b=corrupt(target,shifted,22,[137,98])
        np.testing.assert_array_equal(a[true],b[shifted])


if __name__=='__main__':unittest.main()
