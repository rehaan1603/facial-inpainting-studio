"""Equal-budget compatibility pilot with separate loss-gradient diagnostics."""
import json
import time
import torch
from reference_intervention_core import load_model,predict,set_cache,sha
from target_region_compatibility_v1 import Compatibility,features,install
from prepare_target_compatibility_v1 import OUT,BASE


def load_bundles():
    old=json.loads((BASE/'cache_receipt.json').read_text())
    new=json.loads((OUT/'cache.json').read_text())
    diagnostics={r['identity']:r for r in json.loads((OUT/'diagnostics.json').read_text())}
    result=[]
    for r in old['records']:
        p=BASE/'latents'/(r['identity']+'.pt');q=OUT/'latents'/p.name
        assert sha(p)==r['sha256']
        assert sha(q)==next(a['sha256'] for a in new['records'] if a['identity']==r['identity'])
        result.append(dict(identity=r['identity'],role=r['role'],old=torch.load(p,weights_only=True),
            new=torch.load(q,weights_only=True),diagnostics=diagnostics[r['identity']]))
    return result


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    dest=OUT/'training';dest.mkdir(exist_ok=False)
    data=[r for r in load_bundles() if r['role']=='train'];assert len(data)==16
    torch.set_num_threads(4);model=load_model().cuda();model.first_stage_model.cpu();torch.cuda.empty_cache()
    torch.manual_seed(protocol['seed']);adapter=Compatibility().cuda()
    initial={k:v.detach().clone() for k,v in adapter.state_dict().items()};layers=install(model,adapter)
    parameters=list(adapter.parameters());records=[]
    for arm in protocol['training_arms']:
        adapter.load_state_dict(initial);adapter.mode=arm;adapter.enabled=True
        optimizer=torch.optim.AdamW(parameters,lr=protocol['lr']);torch.manual_seed(protocol['seed'])
        trace=[];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        for step in range(protocol['steps']):
            record=data[step%16];condition=protocol['conditions'][step//16 % len(protocol['conditions'])]
            batch={k:v.cuda() for k,v in record['old'].items()}
            refs=batch['references'].clone();refs[:1]=record['new']['variants'][condition].cuda()
            adapter.context=features(batch['lq'],refs,torch.tensor(record['diagnostics']['conditions'][condition],device='cuda'))
            set_cache(model,list(refs.split(1)))
            t=torch.tensor([[100,300,500,700][(step//16)%4]],device='cuda')
            noise=torch.randn_like(batch['target']);noisy=model.q_sample(batch['target'],t,noise)
            if step==0:
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    adapter.enabled=False;base=predict(model,noisy,batch['lq'],t)
                    adapter.enabled=True;neutral=predict(model,noisy,batch['lq'],t)
                assert torch.equal(base,neutral),'Neutral adapter changed baseline'
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=predict(model,noisy,batch['lq'],t)
                error=(prediction.float()-noise).square()
                rec=error.mean();roi=error[:,:,10:58,12:52].mean()
                losses={'reconstruction':rec,'face_roi':roi};coeff={'reconstruction':.25,'face_roi':.75}
                total=sum(coeff[k]*v for k,v in losses.items())
            diagnostic={}
            if step%16==0:
                vectors={}
                for name,value in losses.items():
                    grads=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
                    vectors[name]=torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(parameters,grads)])
                total_grad=sum(coeff[k]*g for k,g in vectors.items());total_norm=float(total_grad.norm())
                for name,value in losses.items():
                    diagnostic[name]=dict(raw=float(value.detach()),weight=coeff[name],weighted=float(value.detach())*coeff[name],
                        raw_gradient_norm=float(vectors[name].norm()),weighted_gradient_norm=float((vectors[name]*coeff[name]).norm()),
                        norm_ratio_to_total=float((vectors[name]*coeff[name]).norm())/max(total_norm,1e-12))
                diagnostic['note']='Norm ratios need not sum to one because loss gradients can cancel or align.'
            total.backward();gradient=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            optimizer.step();assert not any(p.grad is not None for p in model.parameters())
            trace.append(dict(step=step+1,identity=record['identity'],condition=condition,
                raw_losses={k:float(v.detach()) for k,v in losses.items()},weighted_total=float(total.detach()),
                gradient_norm=float(gradient),gradient_audit=diagnostic))
            if (step+1)%16==0:
                (dest/f'{arm}_trace.json').write_text(json.dumps(trace,indent=2));print(arm,step+1,round(float(total.detach()),6),flush=True)
        path=dest/f'{arm}.pt';torch.save(adapter.state_dict(),path)
        records.append(dict(arm=arm,steps=len(trace),sha256=sha(path),seconds=time.monotonic()-start,
            peak_mib=torch.cuda.max_memory_allocated()/2**20,neutral_exact=True,backbone_frozen=True))
        (dest/'receipt.json').write_text(json.dumps(dict(arms=records,parameters=sum(p.numel() for p in parameters),
            layers=layers,protocol_sha256=sha(OUT/'protocol.json'),core_sha256=sha(OUT.parents[1]/'scripts/target_region_compatibility_v1.py'),
            script_sha256=sha(__file__),other_losses='Perceptual, identity, visible and consistency losses not trained in this controlled pilot.'),indent=2))
    print('TRAINING COMPLETE',flush=True)


if __name__=='__main__':main()
