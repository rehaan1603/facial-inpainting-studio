import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.degradation.face_region_masks import REGIONS,region_masks
from src.reference_fusion.regional_fusion import build_masks,POLICIES

def bank():
    records=[]
    for i in range(2):
        regions={k:dict(quality=.5,visibility_proxy=1.,exposure=1.) for k in REGIONS}
        regions['left_eye']['quality']=1-i;regions['right_eye']['quality']=i
        records.append(dict(index=i,valid=True,regions=regions,identity_compatibility=.5,global_quality=.5,pose_similarity=.5,mask_aware_score=.5))
    return {'references':records}

class RoutingTests(unittest.TestCase):
    def test_partition_and_known_context(self):
        binary=np.zeros((64,64),bool);binary[10:50,10:50]=True
        for policy in POLICIES:
            maps,_=build_masks(bank(),binary,{},policy)
            self.assertTrue(np.allclose(maps.sum(0),1));self.assertTrue((maps>=0).all());self.assertTrue(np.allclose(maps[:,~binary],.5))
    def test_regional_preference_and_permutation(self):
        mask=np.ones((64,64),bool);b=bank();maps,_=build_masks(b,mask,{},sigma=1.)
        regions=region_masks(mask.shape)
        self.assertGreater(maps[0,regions['left_eye']].mean(),maps[1,regions['left_eye']].mean())
        self.assertLess(maps[0,regions['right_eye']].mean(),maps[1,regions['right_eye']].mean())
        reverse,_=build_masks({'references':b['references'][::-1]},mask,{},sigma=1.)
        self.assertTrue(np.allclose(maps,reverse[::-1]))
    def test_one_reference_and_invalid(self):
        b=bank();b['references'][1]['valid']=False
        maps,_=build_masks(b,np.ones((32,32),bool),{})
        self.assertEqual(maps.shape,(1,32,32));self.assertTrue(np.all(maps==1))
        b['references'][0]['valid']=False
        with self.assertRaises(ValueError):build_masks(b,np.ones((32,32),bool),{})
    def test_stock_attention_weighted_sum(self):
        try:
            import torch
            from diffusers.models.attention_processor import Attention,IPAdapterAttnProcessor2_0
        except ImportError:self.skipTest('Run this test in reference_env_v2 for stock attention verification')
        torch.manual_seed(17)
        torch.set_num_threads(4)
        attn=Attention(query_dim=8,cross_attention_dim=8,heads=2,dim_head=4)
        processor=IPAdapterAttnProcessor2_0(hidden_size=8,cross_attention_dim=8,num_tokens=[3],scale=.8)
        hidden=torch.randn(1,16,8);text=torch.randn(1,5,8);ip=torch.randn(1,2,3,8)
        weights=torch.ones(1,2,4,4);weights[:,0]*=.25;weights[:,1]*=.75
        mixed=processor(attn,hidden,encoder_hidden_states=(text,[ip]),ip_adapter_masks=[weights])
        singles=[processor(attn,hidden,encoder_hidden_states=(text,[ip[:,i:i+1]]),ip_adapter_masks=[torch.ones(1,1,4,4)]) for i in range(2)]
        self.assertTrue(torch.allclose(mixed,.25*singles[0]+.75*singles[1],atol=1e-6))
        joint_one=processor(attn,hidden,encoder_hidden_states=(text,[ip[:,:1]]))
        self.assertTrue(torch.allclose(joint_one,singles[0],atol=1e-6))
        # Actual spatial routing: first half selects reference 0, second half reference 1.
        spatial=torch.zeros(1,2,4,4);spatial[:,0,:2]=1;spatial[:,1,2:]=1
        routed=processor(attn,hidden,encoder_hidden_states=(text,[ip]),ip_adapter_masks=[spatial])
        expected=torch.cat([singles[0][:,:8],singles[1][:,8:]],dim=1)
        self.assertTrue(torch.allclose(routed,expected,atol=1e-6))
    def test_gpu_proxy_moves_masks_and_preserves_values(self):
        import torch
        from src.reference_fusion.gpu_masks import GPUInputPipeline
        if not torch.cuda.is_available():self.skipTest('CUDA unavailable')
        class Capture:
            _execution_device=torch.device('cuda:0')
            def __call__(self,**kwargs):return kwargs
        source=torch.rand(1,4,8,8)
        result=GPUInputPipeline(Capture())(cross_attention_kwargs={'ip_adapter_masks':[source]})
        target=result['cross_attention_kwargs']['ip_adapter_masks'][0]
        self.assertEqual(target.device.type,'cuda');self.assertEqual(target.dtype,source.dtype)
        self.assertTrue(torch.equal(target.cpu(),source));self.assertEqual(source.device.type,'cpu')

if __name__=='__main__':unittest.main()
