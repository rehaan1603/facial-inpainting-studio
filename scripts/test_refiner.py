"""Analytic invariants for the learned controls and new object protocol."""
import unittest
import numpy as np
import torch
from train_refiner import sample
from refiner import MaskRefiner,refinement_loss

class RefinerTests(unittest.TestCase):
    def test_training_hidden_target_independence(self):
        texture=np.full((128,128,3),77,np.uint8)
        for seed in range(12):
            a=np.zeros_like(texture);b=np.full_like(texture,255)
            oa,ma,ta=sample(a,texture,seed);ob,mb,tb=sample(b,texture,seed)
            true=ta[0]>0
            self.assertTrue(np.array_equal(ma,mb));self.assertTrue(np.array_equal(ta,tb))
            self.assertTrue(np.array_equal(oa[true],ob[true]))
            self.assertTrue(np.array_equal(oa[~true],a[~true]/255))
    def test_loss_penalizes_false_positive_more(self):
        truth=torch.tensor([[[[1.,0.]]]])
        good=torch.tensor([[[[5.,-5.]]]])
        fp=good.clone();fp[0,0,0,1]=5
        fn=good.clone();fn[0,0,0,0]=-5
        self.assertGreater(float(refinement_loss(fp,truth,4)),float(refinement_loss(fn,truth,4)))
        self.assertAlmostEqual(float(refinement_loss(fp,truth,1)),float(refinement_loss(fn,truth,1)),places=6)
    def test_empty_regions_finite_gradients(self):
        for fill in [0.,1.]:
            logits=torch.zeros((2,1,16,16),requires_grad=True);truth=torch.full_like(logits,fill)
            loss=refinement_loss(logits,truth,4);loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(logits.grad).all())
    def test_cpu_forward_backward(self):
        torch.set_num_threads(2);model=MaskRefiner(16)
        pred=model(torch.zeros(2,3,32,32),torch.zeros(2,1,32,32))
        self.assertEqual(tuple(pred.shape),(2,1,32,32));pred.mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
if __name__=='__main__':unittest.main()
