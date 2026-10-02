"""Small matched objective ablation; unchanged 193-parameter architecture."""
import json,sys,time,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from reference_intervention_core import ROOT,load_model,predict,set_cache,sha
from target_region_compatibility_v1 import Compatibility,features,install
from train_target_compatibility_v1 import load_bundles

OUT=ROOT/'outputs/identity_compatibility_v2'

def main():
    OUT.mkdir(exist_ok=False)
    data=[r for r in load_bundles() if r['role']=='train']
    rng=np.random.default_rng(20261002)
    # Each identity sees each condition once; independently shuffled timestep assignments.
    schedule=[dict(identity=r['identity'],condition=c) for r in data for c in ['clean','blur','noise','wrong_patch']]
    rng.shuffle(schedule)
    times=np.tile([100,200,300,400],16);rng.shuffle(times)
    for row,t in zip(schedule,times):row['timestep']=int(t)
    config=dict(steps=64,seed=20261002,lr=.001,arms=['diffusion','identity'],parameters=193,
        identity_ids=[r['identity'] for r in data],schedule=schedule,
        objective='Global .25 plus face ROI .75 epsilon MSE. Identity arm adds decoded frozen FaceNet cosine loss every fourth update; weight calibrated from first identity-step gradient ratio, capped at 10.',
        identity_crop='Entire aligned face resized bilinearly to160, fixed standardization; surrogate differs from MTCNN evaluation crop.',
        evaluation=dict(identities=['263','6681','7400','10149'],seeds=[17,29,43],conditions=['clean','wrong_patch'],steps=50,guidance=1.5,resolution=512),
        promotion='All three seeds must preserve PSNR/SSIM/FaceNet/ArcFace and eye crop LPIPS at 1e-6 numeric tolerance, improve LPIPS overall, complete paired support; no practical degradation margin established.',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),script_sha256=sha(__file__),
        architecture_sha256=sha(ROOT/'scripts/target_region_compatibility_v1.py'),
        previous_protocol_sha256=sha(ROOT/'outputs/target_compatibility_v1/protocol.json'))
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    torch.set_num_threads(4);torch.manual_seed(config['seed'])
    cache=Path(json.loads((ROOT/'configs/local.json').read_text())['cache'])
    sys.path.insert(0,str(cache/'identity_eval_v1'))
    from facenet_pytorch import InceptionResnetV1
    enc=InceptionResnetV1(classify=True,num_classes=8631)
    enc.load_state_dict(torch.load(cache/'identity_eval_v1/vggface2.pt',map_location='cpu',weights_only=True))
    enc.classify=False;enc=enc.eval().requires_grad_(False).cuda()
    def embed(rgb):return enc((F.interpolate(rgb,size=(160,160),mode='bilinear',align_corners=False)*255-127.5)/128)
    targets={}
    for r in data:
        rgb=np.asarray(Image.open(ROOT/'outputs/reference_intervention_pilot_v1/images'/r['identity']/'target.png').convert('RGB')).copy()
        with torch.no_grad():targets[r['identity']]=embed(torch.from_numpy(rgb).permute(2,0,1)[None].cuda().float()/255).detach()
    model=load_model().cuda();torch.manual_seed(config['seed']);adapter=Compatibility().cuda();layers=install(model,adapter)
    initial={k:v.detach().clone() for k,v in adapter.state_dict().items()};params=list(adapter.parameters());records=[]
    for arm in config['arms']:
        adapter.load_state_dict(initial);optimizer=torch.optim.AdamW(params,lr=config['lr']);torch.manual_seed(config['seed'])
        start=time.monotonic();torch.cuda.reset_peak_memory_stats();trace=[];weight=None
        for step,s in enumerate(schedule):
            r=next(r for r in data if r['identity']==s['identity']);b={k:v.cuda() for k,v in r['old'].items()}
            refs=b['references'].clone();refs[:1]=r['new']['variants'][s['condition']].cuda()
            adapter.context=features(b['lq'],refs,torch.tensor(r['diagnostics']['conditions'][s['condition']],device='cuda'))
            set_cache(model,list(refs.split(1)));t=torch.tensor([s['timestep']],device='cuda')
            noise=torch.randn_like(b['target']);noisy=model.q_sample(b['target'],t,noise);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=predict(model,noisy,b['lq'],t)
                err=(pred.float()-noise).square();diff=.25*err.mean()+.75*err[:,:,10:58,12:52].mean()
            total=diff;extra={}
            if arm=='identity' and step%4==0:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    x0=model.predict_start_from_noise(noisy,t,pred)
                    decoded=checkpoint(model._decode_first_stage,x0,use_reentrant=False)
                rgb=((decoded.float()+1)/2).clamp(0,1)
                identity=(1-F.cosine_similarity(checkpoint(embed,rgb,use_reentrant=False),targets[s['identity']])).mean()
                if weight is None:
                    def gradnorm(loss):
                        g=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
                        return float(torch.sqrt(sum(v.float().square().sum() for v in g if v is not None)))
                    dn,ino=gradnorm(diff),gradnorm(identity)
                    weight=min(10.,dn/max(ino,1e-12));extra.update(diffusion_grad_norm=dn,identity_grad_norm=ino)
                total=diff+weight*identity
                extra.update(identity_raw=float(identity.detach()),identity_weight=weight,identity_weighted=float(identity.detach())*weight)
            total.backward();gn=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
            assert not any(p.grad is not None for p in model.parameters()) and not any(p.grad is not None for p in enc.parameters())
            trace.append(dict(step=step+1,**s,diffusion_raw=float(diff.detach()),total=float(total.detach()),gradient_norm=float(gn),**extra))
            (OUT/f'{arm}_trace.json').write_text(json.dumps(trace,indent=2))
            print(arm,step+1,trace[-1]['total'],flush=True)
            if arm=='identity' and step%4==0:del identity,decoded,rgb,x0
            del total,diff,pred,err
        path=OUT/f'{arm}.pt';torch.save(adapter.state_dict(),path)
        records.append(dict(arm=arm,steps=len(trace),checkpoint_sha256=sha(path),seconds=time.monotonic()-start,
            peak_mib=torch.cuda.max_memory_allocated()/2**20,identity_weight=weight,backbone_frozen=True,recognition_frozen=True))
        (OUT/'training.json').write_text(json.dumps(dict(arms=records,layers=layers,config_sha256=sha(OUT/'config.json')),indent=2))

if __name__=='__main__':main()
