"""Isolate fixed face-crop supervision with unchanged v2 schedule and loss weight."""
import json,time,sys,subprocess,argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from reference_intervention_core import ROOT,load_model,predict,set_cache,sha
from target_region_compatibility_v1 import Compatibility,features,install
from train_target_compatibility_v1 import load_bundles
from identity_crop_v3 import crop_tensor,standardize

OUT=ROOT/'outputs/identity_timestep_v5'
ARM='timestep_scaled'

def main():
    assert not (OUT/'config.json').exists()
    prior=ROOT/'outputs/identity_compatibility_v2'
    oldcfg=json.loads((prior/'config.json').read_text());oldtr=json.loads((prior/'training.json').read_text())
    weight=next(a['identity_weight'] for a in oldtr['arms'] if a['arm']=='identity')
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'crops.json').write_bytes((ROOT/'outputs/identity_crop_v3/crops.json').read_bytes())
    crops={r['identity']:r for r in json.loads((OUT/'crops.json').read_text())['records']}
    cfg=dict(oldcfg);cfg.update(arms=[ARM],identity_weight=weight,date='2026-10-04',
        identity_crop='Fixed zero-margin training target MTCNN bounds, differentiable antialiased bilinear160 resize, same standardization; evaluator independently detects each output.',
        objective='Single-change comparison against v4 balanced_fixed: multiply inherited identity weight by author sqrt_alphas_cumprod[t]. FaceNet and adapter differ from author full training; not an exact reproduction.', gradient_cap_ratio=None, identity_scaling='sqrt_alphas_cumprod[t]',
        prior_config_sha256=sha(prior/'config.json'),crops_sha256=sha(OUT/'crops.json'),crop_helper_sha256=sha(ROOT/'scripts/identity_crop_v3.py'),
        script_sha256=sha(__file__),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (OUT/'config.json').write_text(json.dumps(cfg,indent=2))
    torch.set_num_threads(4);torch.manual_seed(cfg['seed'])
    cache=Path(json.loads((ROOT/'configs/local.json').read_text())['cache'])
    sys.path.insert(0,str(cache/'identity_eval_v1'))
    from facenet_pytorch import InceptionResnetV1
    enc=InceptionResnetV1(classify=True,num_classes=8631)
    enc.load_state_dict(torch.load(cache/'identity_eval_v1/vggface2.pt',map_location='cpu',weights_only=True))
    enc.classify=False;enc=enc.eval().requires_grad_(False).cuda()
    def embed(rgb,box):return enc(standardize(crop_tensor(rgb,box)))
    data=[r for r in load_bundles() if r['role']=='train'];targets={}
    for r in data:
        cr=crops[r['identity']]
        p=ROOT/'outputs/reference_intervention_pilot_v1/images'/r['identity']/'target.png'
        assert sha(p)==cr['source_sha256']
        if cr['status']!='ok':continue
        rgb=np.asarray(Image.open(p).convert('RGB')).copy()
        with torch.no_grad():targets[r['identity']]=embed(torch.from_numpy(rgb).permute(2,0,1)[None].cuda().float()/255,cr['box']).detach()
    model=load_model().cuda();torch.manual_seed(cfg['seed']);adapter=Compatibility().cuda();layers=install(model,adapter)
    params=list(adapter.parameters());optimizer=torch.optim.AdamW(params,lr=cfg['lr']);torch.manual_seed(cfg['seed'])
    trace=[];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    def grads(loss):
        g=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if v is None else v).float().flatten() for p,v in zip(params,g)])
    for step,s in enumerate(cfg['schedule']):
        r=next(r for r in data if r['identity']==s['identity']);b={k:v.cuda() for k,v in r['old'].items()}
        refs=b['references'].clone();refs[:1]=r['new']['variants'][s['condition']].cuda()
        adapter.context=features(b['lq'],refs,torch.tensor(r['diagnostics']['conditions'][s['condition']],device='cuda'))
        set_cache(model,list(refs.split(1)));t=torch.tensor([s['timestep']],device='cuda')
        noise=torch.randn_like(b['target']);noisy=model.q_sample(b['target'],t,noise);optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred=predict(model,noisy,b['lq'],t)
            err=(pred.float()-noise).square();diff=.25*err.mean()+.75*err[:,:,10:58,12:52].mean()
        total=diff;extra={};has_identity=s['identity'] in targets
        if has_identity:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                x0=model.predict_start_from_noise(noisy,t,pred);decoded=checkpoint(model._decode_first_stage,x0,use_reentrant=False)
            rgb=((decoded.float()+1)/2).clamp(0,1)
            identity=(1-F.cosine_similarity(checkpoint(embed,rgb,crops[s['identity']]['box'],use_reentrant=False),targets[s['identity']])).mean()
            dg,ig=grads(diff),grads(identity)
            timestep_scale=float(model.sqrt_alphas_cumprod[t].detach().item())
            assert 0 < timestep_scale <= 1
            effective_weight=weight*timestep_scale
            extra.update(timestep_scale=timestep_scale,identity_raw=float(identity.detach()),identity_weight=effective_weight,
                diffusion_grad_norm=float(dg.norm()),identity_grad_norm=float(ig.norm()),
                weighted_identity_grad_norm=float((effective_weight*ig).norm()),
                gradient_cosine=float(F.cosine_similarity(dg[None],ig[None]).item()))
            total=diff+effective_weight*identity
        elif step%4==0:extra['identity_skipped']='Training target detection is unavailable; diffusion update retained.'
        total.backward();gn=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
        assert not any(p.grad is not None for p in model.parameters()) and not any(p.grad is not None for p in enc.parameters())
        trace.append(dict(step=step+1,**s,diffusion_raw=float(diff.detach()),total=float(total.detach()),gradient_norm=float(gn),**extra))
        (OUT/'trace.json').write_text(json.dumps(trace,indent=2))
        print(ARM,step+1,trace[-1]['total'],flush=True)
        if has_identity:del identity,decoded,rgb,x0,dg,ig
        del total,diff,pred,err
    p=OUT/'adapter.pt';torch.save(adapter.state_dict(),p)
    (OUT/'training.json').write_text(json.dumps(dict(config_sha256=sha(OUT/'config.json'),layers=layers,
        parameters=sum(p.numel() for p in params),arms=[dict(arm=ARM,steps=len(trace),checkpoint_sha256=sha(p),
            seconds=time.monotonic()-start,peak_mib=torch.cuda.max_memory_allocated()/2**20,identity_weight=weight,
            supervised_steps=sum('identity_raw' in r for r in trace),backbone_frozen=True,recognition_frozen=True)]),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['timestep_scaled'],required=True)
    ARM=parser.parse_args().arm;OUT=OUT/ARM;main()
