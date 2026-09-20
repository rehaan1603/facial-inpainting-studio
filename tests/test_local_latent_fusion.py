import unittest
import numpy as np
from src.local_correspondence.latent_fusion import region_weights,fuse_latents,LocalLatentPipeline


class LocalFusionTests(unittest.TestCase):
    def test_weights_and_spatial_reference_choice(self):
        q=np.array([[1.,0.],[0.,1.]])
        weights=region_weights(q,np.zeros_like(q),np.zeros(2),'single')
        refs=np.stack([np.ones((4,64,64)),np.full((4,64,64),3.)])
        basis=np.zeros((2,64,64));basis[0,:,:32]=1;basis[1,:,32:]=1
        result,maps=fuse_latents(refs,weights,basis)
        self.assertTrue(np.all(result[:,:,:32]==1));self.assertTrue(np.all(result[:,:,32:]==3));self.assertTrue(np.allclose(maps.sum(0),1))
        for policy in ['equal','quality','damage']:
            w=region_weights(q,q,np.array([.2,.8]),policy);self.assertTrue(np.allclose(w.sum(0),1))
            permutation=region_weights(q[::-1],q[::-1],np.array([.2,.8]),policy)
            self.assertTrue(np.allclose(permutation,w[::-1]))

    def test_zero_gain_is_exact_passthrough(self):
        class Fake:
            def __call__(self,**kwargs):return kwargs
        callback=lambda *a:None
        result=LocalLatentPipeline(Fake(),None,np.zeros((64,64)))(callback_on_step_end=callback,seed=17)
        self.assertIs(result['callback_on_step_end'],callback);self.assertNotIn('callback_on_step_end_tensor_inputs',result)

    def test_injection_only_first_step(self):
        import torch
        class Fake:
            def __call__(self,**kwargs):
                tensor=torch.zeros((2,4,64,64));cb=kwargs['callback_on_step_end']
                for step in [0,1]:tensor=cb(self,step,100,{'masked_image_latents':tensor,'latents':torch.zeros_like(tensor)})['masked_image_latents']
                return tensor
        gate=np.zeros((64,64));gate[:32]=.25
        result=LocalLatentPipeline(Fake(),np.ones((4,64,64)),gate)()
        self.assertTrue(torch.all(result[:,:,:32]==.25));self.assertTrue(torch.all(result[:,:,32:]==0))


if __name__=='__main__':unittest.main()
