import unittest
import numpy as np
from PIL import Image
import torch
from src.preservation.refface_baseline import RefFaceBaseline, conservative_mask


class ReffaceInputTests(unittest.TestCase):
    def test_one_pixel_mask_survives_native_reduction(self):
        mask=np.zeros((512,512),dtype=bool);mask[301,247]=True
        reduced=conservative_mask(mask)
        restored=torch.nn.functional.interpolate(reduced,size=(512,512),mode='nearest')[0,0].numpy().astype(bool)
        self.assertTrue(restored[mask].all())

    def test_erased_payload_invariant_and_visible_bytes_exact(self):
        # A deterministic stand-in isolates preprocessing from the learned network.
        class Fake(RefFaceBaseline):
            def __init__(self):self.device='cpu'
            def reference_labels(self,reference):return torch.zeros(1,1,256,256,dtype=torch.long)
            def native(self,observed,mask,reference,labels):return observed*.5+reference*.5
        rng=np.random.default_rng(17)
        image=rng.integers(0,256,size=(512,512,3),dtype=np.uint8)
        mask=np.zeros((512,512),dtype=bool);mask[123:187,231:303]=True
        changed=image.copy();changed[mask]=255-image[mask]
        ref=Image.fromarray(np.full_like(image,75))
        first=np.asarray(Fake().reconstruct(Image.fromarray(image),mask,ref)[0])
        second=np.asarray(Fake().reconstruct(Image.fromarray(changed),mask,ref)[0])
        np.testing.assert_array_equal(first,second)
        np.testing.assert_array_equal(first[~mask],image[~mask])


if __name__=='__main__':unittest.main()
