"""Training-only gradient/memory feasibility probe, not an accuracy experiment."""
import json
import sys
import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from reference_intervention_core import ROOT,load_model,predict,set_cache,sha
from target_region_compatibility_v1 import Compatibility,features,install
from train_target_compatibility_v1 import load_bundles

def main():
    out=ROOT/'outputs/decoded_identity_probe_v2';out.mkdir(exist_ok=False)
    receipt={'scope':'One training identity, no optimizer update or performance claim','identity':'8570',
        'seed':20261002,'timestep':100,'identity_crop':'Full aligned 512 image resized bilinearly to 160; feasibility only, differs from evaluation MTCNN crop',
        'script_sha256':sha(__file__)}
    start=time.monotonic()
    try:
        torch.set_num_threads(4);torch.manual_seed(20261002)
        cache=ROOT/'outputs/reference_intervention_pilot_v1'
        config=json.loads((ROOT/'configs/local.json').read_text())
        from pathlib import Path
        weights=Path(config['cache'])/'identity_eval_v1'
        sys.path.insert(0,str(weights))
        from facenet_pytorch import InceptionResnetV1
        encoder=InceptionResnetV1(classify=True,num_classes=8631)
        encoder.load_state_dict(torch.load(weights/'vggface2.pt',map_location='cpu',weights_only=True))
        encoder.classify=False;encoder=encoder.eval().requires_grad_(False).cuda()
        receipt['identity_weights_sha256']=sha(weights/'vggface2.pt')
        model=load_model().cuda();adapter=Compatibility().cuda();install(model,adapter)
        record=next(r for r in load_bundles() if r['identity']=='8570' and r['role']=='train')
        b={k:v.cuda() for k,v in record['old'].items()};refs=b['references']
        adapter.context=features(b['lq'],refs,torch.tensor(record['diagnostics']['conditions']['clean'],device='cuda'))
        set_cache(model,list(refs.split(1)))
        target=np.asarray(Image.open(cache/'images/8570/target.png').convert('RGB')).copy()
        target=torch.from_numpy(target).permute(2,0,1)[None].cuda().float()/255
        # FaceNet's standardization is (pixel - 127.5)/128.
        def embed(rgb):return encoder((F.interpolate(rgb,size=(160,160),mode='bilinear',align_corners=False)*255-127.5)/128)
        with torch.no_grad():target_vector=embed(target)
        t=torch.tensor([100],device='cuda');noise=torch.randn_like(b['target']);noisy=model.q_sample(b['target'],t,noise)
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            adapter.enabled=False
            with torch.no_grad():baseline=predict(model,noisy,b['lq'],t)
            adapter.enabled=True;prediction=predict(model,noisy,b['lq'],t)
            receipt['neutral_exact']=torch.equal(baseline,prediction)
            x0=model.predict_start_from_noise(noisy,t,prediction)
            decoded=checkpoint(model._decode_first_stage,x0,use_reentrant=False)
            rgb=((decoded.float()+1)/2).clamp(0,1)
        # Recognition forward in fp32, frozen encoder but differentiable input.
        identity=(1-F.cosine_similarity(embed(rgb),target_vector)).mean()
        identity.backward()
        grads=[p.grad for p in adapter.parameters() if p.grad is not None]
        norm=float(torch.sqrt(sum(g.float().square().sum() for g in grads)))
        receipt.update(status='pass' if norm>0 and np.isfinite(norm) else 'failed_zero_gradient',
            identity_loss=float(identity.detach()),identity_gradient_norm=norm,
            backbone_frozen=not any(p.grad is not None for p in model.parameters()),
            encoder_frozen=not any(p.grad is not None for p in encoder.parameters()),
            peak_mib=torch.cuda.max_memory_allocated()/2**20)
    except Exception as error:
        receipt.update(status='failed',error=repr(error))
        if torch.cuda.is_available():receipt['peak_mib']=torch.cuda.max_memory_allocated()/2**20
    receipt['seconds']=time.monotonic()-start
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':main()
