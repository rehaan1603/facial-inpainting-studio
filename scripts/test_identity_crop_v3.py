import unittest
import torch
from identity_crop_v3 import crop_tensor,standardize

class CropTests(unittest.TestCase):
    def test_gradients_only_inside_frozen_box(self):
        x=torch.rand(1,3,32,32,requires_grad=True)
        standardize(crop_tensor(x,[8,8,24,24])).sum().backward()
        self.assertGreater(float(x.grad[:,:,8:24,8:24].abs().sum()),0)
        self.assertEqual(float(x.grad[:,:,:8].abs().sum()),0)
        self.assertEqual(float(x.grad[:,:,:,24:].abs().sum()),0)

    def test_invalid_crop_rejected(self):
        with self.assertRaises(ValueError):crop_tensor(torch.zeros(1,3,32,32),[4,4,4,20])

if __name__=='__main__':unittest.main()
