import unittest
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from stable_attention_difference_v2 import attention_difference


class DifferenceTests(unittest.TestCase):
    def test_nonzero_derivative_and_finite_difference(self):
        torch.manual_seed(421)
        torch.set_num_threads(2)
        shapes=[(1,2,3,4),(1,2,5,4),(1,2,5,4),(1,2,3,4),(1,2,5,4)]
        xs=[torch.randn(s,dtype=torch.float64,requires_grad=True) for s in shapes]
        q,k,v,dq,dv=xs
        with sdpa_kernel(SDPBackend.MATH):
            expected=F.scaled_dot_product_attention(q+dq,k,v+dv)-F.scaled_dot_product_attention(q,k,v)
            actual=attention_difference(*xs)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            w=torch.randn_like(actual)
            ea=torch.autograd.grad(expected,xs,w)
            aa=torch.autograd.grad(actual,xs,w)
            for a,b in zip(ea,aa):
                torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
            self.assertTrue(torch.autograd.gradcheck(attention_difference,tuple(xs),fast_mode=True))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required for GPU check')
    def test_gpu_zero_and_nonzero(self):
        torch.manual_seed(421)
        device='cuda'
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.MATH]):
            for dtype in [torch.float16,torch.bfloat16]:
                q=torch.randn(1,10,16,64,device=device,dtype=dtype,requires_grad=True)
                k=torch.randn(1,10,64,64,device=device,dtype=dtype,requires_grad=True)
                v=torch.randn_like(k,requires_grad=True)
                dq=torch.zeros_like(q,requires_grad=True);dv=torch.zeros_like(v,requires_grad=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    out=attention_difference(q,k,v,dq,dv)
                self.assertEqual(torch.count_nonzero(out).item(),0)
                gs=torch.autograd.grad(out,(q,k,v,dq,dv),torch.randn_like(out))
                for g in gs[:3]: self.assertEqual(torch.count_nonzero(g).item(),0)
                for g in gs[3:]: self.assertTrue(torch.isfinite(g).all() and g.abs().sum()>0)
                dq=(torch.randn_like(q)*.02).requires_grad_(True)
                dv=(torch.randn_like(v)*.02).requires_grad_(True)
                out=attention_difference(q,k,v,dq,dv)
                self.assertGreater(out.float().abs().sum().item(),0)
                for g in torch.autograd.grad(out,(q,k,v,dq,dv),torch.randn_like(out)):
                    self.assertTrue(torch.isfinite(g).all())


if __name__=='__main__':unittest.main()
