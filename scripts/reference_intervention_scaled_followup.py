"""One training-log-calibrated follow-up; reused pilot holdout is now development."""
import json
import time
import numpy as np
import torch
from PIL import Image
from reference_intervention_core import ROOT, ReferenceBias, install_adapter, load_model, predict, set_cache, sha


def main():
    base = ROOT/'outputs/reference_intervention_pilot_v1'
    out = base/'scaled_followup'
    out.mkdir(exist_ok=False)
    manifest = json.loads((base/'manifest.json').read_text())
    cache = json.loads((base/'cache_receipt.json').read_text())
    trace_path = base/'training/intervention_trace.json'
    trace = json.loads(trace_path.read_text())
    ratio = np.median([r['consistency']/(0.5*(r['clean_loss']+r['corrupt_loss'])) for r in trace])
    weight = float(0.1/ratio)
    protocol = dict(scope='Training-derived loss-scale follow-up, not independent validation',
        consistency_weight=weight, rule='Set weighted consistency to 10% of reconstruction at the median previous training-step ratio',
        calibration_trace_sha256=sha(trace_path), seed=manifest['seed'], steps=64, lr=0.001,
        checkpoint_selection='last step only', script_sha256=sha(__file__), core_sha256=sha(ROOT/'scripts/reference_intervention_core.py'),
        manifest_sha256=sha(base/'manifest.json'), final_test_used=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    bundles = {}
    for row in cache['records']:
        path = base/'latents'/(row['identity']+'.pt')
        assert sha(path) == row['sha256']
        bundles[row['identity']] = torch.load(path, weights_only=True)
    model = load_model().cuda()
    model.first_stage_model.cpu()
    torch.cuda.empty_cache()
    torch.manual_seed(manifest['seed'])
    adapter = ReferenceBias().cuda()
    install_adapter(model, adapter)
    optimizer = torch.optim.AdamW(adapter.parameters(),lr=0.001)
    torch.manual_seed(manifest['seed'])
    training = [r for r in cache['records'] if r['role']=='train']
    logs = []
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    for step in range(64):
        bundle = {k:v.cuda() for k,v in bundles[training[step%16]['identity']].items()}
        refs = list(bundle['references'].split(1))
        t = torch.tensor([[100,300,500,700][step//16]],device='cuda')
        noise = torch.randn_like(bundle['target'])
        noisy = model.q_sample(bundle['target'],t,noise)
        optimizer.zero_grad(set_to_none=True)
        set_cache(model,refs)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            clean = predict(model,noisy,bundle['lq'],t)
            clean_loss = (clean.float()-noise).square().mean()
        teacher = clean.detach().float()
        (0.5*clean_loss).backward()
        del clean
        refs[0] = bundle['corrupt_reference']
        set_cache(model,refs)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            corrupt = predict(model,noisy,bundle['lq'],t)
            corrupt_loss = (corrupt.float()-noise).square().mean()
            consistency = (corrupt.float()-teacher).square().mean()
            loss = 0.5*corrupt_loss + weight*consistency
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step()
        assert not any(p.grad is not None for p in model.parameters())
        logs.append(dict(step=step+1,clean_loss=clean_loss.item(),corrupt_loss=corrupt_loss.item(),consistency=consistency.item(),gradient_norm=gradient.item()))
        if (step+1)%8 == 0:
            print('scaled training',step+1,flush=True)
            (out/'trace.json').write_text(json.dumps(logs,indent=2))
    torch.save(adapter.state_dict(),out/'adapter.pt')
    (out/'training_receipt.json').write_text(json.dumps(dict(protocol_sha256=sha(out/'protocol.json'),
        checkpoint_sha256=sha(out/'adapter.pt'), peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        elapsed_seconds=time.monotonic()-start),indent=2))
    # Release training graphs before restoring the image decoder.
    del optimizer, loss, clean_loss, corrupt_loss, consistency, corrupt, teacher, bundle
    from ldm import cache_kv
    cache_kv.clear_cache()
    torch.cuda.empty_cache()
    model.first_stage_model.cuda()
    from ldm.models.diffusion.ddim import DDIMSampler
    sampler = DDIMSampler(model,print_tqdm=False,schedule='uniform_trailing')
    rows = []
    for record in cache['records']:
        if record['role'] != 'adapter_holdout':
            continue
        identity = record['identity']
        bundle = bundles[identity]
        for condition in ['clean','corrupt']:
            refs = bundle['references'].clone()
            if condition=='corrupt':
                refs[:1] = bundle['corrupt_reference']
            cond = dict(lq_image=bundle['lq'].cuda(),ref_image=torch.cat(list(refs.split(1)),-1).cuda())
            null = dict(lq_image=cond['lq_image'],ref_image=torch.zeros_like(cond['ref_image']))
            torch.manual_seed(17)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                latent,_ = sampler.sample(S=50,unconditional_guidance_scale=1.5,conditioning=cond,
                    unconditional_conditioning=null,shape=[8,64,64],x_T=torch.randn(1,8,64,64,device='cuda'),batch_size=1,verbose=False)
                decoded = ((model.decode_first_stage(latent)+1)/2).clamp(0,1)
            assert torch.isfinite(decoded).all()
            array = (decoded[0].permute(1,2,0).float().cpu().numpy()*255).round().astype(np.uint8)
            path = out/(identity+'_'+condition+'.png')
            Image.fromarray(array).save(path)
            rows.append(dict(identity=identity,arm='scaled_intervention',condition=condition,file=path.name,sha256=sha(path)))
            (out/'rows.json').write_text(json.dumps(rows,indent=2))
            print(path.name,flush=True)


if __name__ == '__main__':
    main()
