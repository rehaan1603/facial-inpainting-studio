"""Continue all six signed local training runs at a matched 6,000-step total budget."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from train_refiner import sample
from refiner import MaskRefiner,refinement_loss
ROOT=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    cfg=json.loads((ROOT/'configs/learned_refiner.json').read_text());extension=json.loads((ROOT/'configs/refiner_extension_v1.json').read_text());local=json.loads((ROOT/'configs/local.json').read_text())
    cache=Path(local['cache'])/'refiner_data';metadata=json.loads((cache/'metadata.json').read_text());assert digest(ROOT/cfg['manifest'])==metadata['signature']['manifest_sha256']
    train=np.load(cache/'train.npy',mmap_mode='r');val=np.load(cache/'tuning.npy',mmap_mode='r');torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    signature={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'scripts/train_refiner.py',ROOT/'scripts/refiner.py',ROOT/'configs/learned_refiner.json',ROOT/'configs/refiner_extension_v1.json',ROOT/cfg['manifest']]}
    signature['training_cache_sha256']=digest(cache/'train.npy');signature['tuning_cache_sha256']=digest(cache/'tuning.npy')
    def batch(indices,seed,data):
        xs=[];ms=[];ts=[]
        for j,index in enumerate(indices):
            x,m,t=sample(data[index],train[(index*37+seed)%len(train)],seed+j*997);xs.append(x.transpose(2,0,1));ms.append(m);ts.append(t)
        return tuple(torch.from_numpy(np.stack(a)).cuda() for a in [xs,ms,ts])
    tuning_batches=[batch(list(range(i,min(i+16,len(val)))),90000+i,val) for i in range(0,len(val),16)]
    out=ROOT/'outputs/refiner_extension_v1';out.mkdir(exist_ok=True)
    for seed in extension['seeds']:
        for variant,cost in extension['variants'].items():
            initial=ROOT/'outputs/learned_refiner'/f'{variant}_{seed}';folder=out/f'{variant}_{seed}';folder.mkdir(exist_ok=True);latest=folder/'latest.pt';final=folder/'training.json'
            initial_report=json.loads((initial/'training.json').read_text());assert initial_report['steps']==extension['initial_steps']
            for name,sha in initial_report['signature'].items():assert digest(ROOT/name)==sha
            run_signature={**signature,'initial_latest_sha256':digest(initial/'latest.pt'),'initial_best_sha256':digest(initial/'best.pt')}
            if final.exists():assert json.loads(final.read_text())['signature']==run_signature;continue
            # These optimizer/RNG checkpoints were created locally by our training script.
            state=torch.load(latest if latest.exists() else initial/'latest.pt',map_location='cpu',weights_only=False)
            assert state['signature']==(run_signature if latest.exists() else initial_report['signature'])
            model=MaskRefiner(cfg['width']).cuda();model.load_state_dict(state['model'],strict=True);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate']);optimizer.load_state_dict(state['optimizer']);scaler=torch.amp.GradScaler('cuda');scaler.load_state_dict(state['scaler'])
            rng=np.random.default_rng();rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
            start_step=state['step'];history=state['history'];best=state['best'];previous_seconds=state.get('extension_seconds',0.0)
            if not (folder/'best.pt').exists():
                best_state=torch.load(initial/'best.pt',map_location='cpu',weights_only=True);best_state['signature']=run_signature;torch.save(best_state,folder/'best.pt')
            start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            for step in range(start_step,extension['total_steps']):
                indices=rng.integers(0,len(train),cfg['batch_size']);x,m,t=batch(indices,seed*1000000+step*cfg['batch_size'],train)
                model.train();optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):loss=refinement_loss(model(x,m),t,cost)
                assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1);scaler.step(optimizer);scaler.update()
                if (step+1)%cfg['validation_interval']==0:
                    model.eval()
                    with torch.inference_mode():value=float(np.mean([refinement_loss(model(vx,vm),vt,cost).item() for vx,vm,vt in tuning_batches]))
                    history.append({'step':step+1,'train_loss':loss.detach().item(),'tuning_loss':value})
                    if value<best:
                        best=value;torch.save({'model':model.state_dict(),'width':cfg['width'],'seed':seed,'variant':variant,'step':step+1,'signature':run_signature},folder/'best.pt')
                    state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'step':step+1,'best':best,'history':history,'signature':run_signature,'extension_seconds':previous_seconds+time.perf_counter()-start}
                    tmp=folder/'latest.tmp';torch.save(state,tmp);tmp.replace(latest);print(f'{variant} seed={seed} step={step+1}/6000 tuning_loss={value:.6f}',flush=True)
            final.write_text(json.dumps({'seed':seed,'variant':variant,'total_steps':extension['total_steps'],'initial_steps':extension['initial_steps'],'history':history,'best_tuning_loss':best,'extension_seconds':state['extension_seconds'],'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'signature':run_signature,'limits':extension['claim_limit']},indent=2));del model,optimizer,scaler,state;torch.cuda.empty_cache()
    print('All six matched training extensions complete')
if __name__=='__main__':main()
