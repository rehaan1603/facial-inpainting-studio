"""Train matched generic and cost-sensitive controls; no test-set access."""
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFilter
import torch
from torch.nn import functional as F
from refiner import MaskRefiner,refinement_loss

ROOT=Path(__file__).resolve().parents[1]


def sample(target,texture,seed):
    rng=np.random.default_rng(seed);size=target.shape[0];mask=Image.new('L',(size,size));draw=ImageDraw.Draw(mask)
    for _ in range(int(rng.integers(1,4))):
        x,y=rng.integers(5,size-30,2);w,h=rng.integers(15,55,2)
        box=(int(x),int(y),int(min(size-3,x+w)),int(min(size-3,y+h)))
        if rng.random()<.5:draw.rectangle(box,fill=255)
        else:draw.ellipse(box,fill=255)
    true=np.array(mask)>0
    kind=int(rng.integers(0,3))
    if kind==0:occ=np.asarray(Image.fromarray(rng.integers(0,256,(8,8,3),dtype='uint8')).resize((size,size),Image.Resampling.BILINEAR)).copy()
    elif kind==1:occ=np.broadcast_to(rng.integers(0,256,(1,1,3),dtype='uint8'),target.shape)
    else:occ=np.asarray(Image.fromarray(texture).rotate(float(rng.uniform(-30,30)),resample=Image.Resampling.BILINEAR)).copy()
    observed=np.where(true[...,None],occ,target).astype('float32')/255
    radius=int(rng.integers(-6,7))
    changed=mask.filter(ImageFilter.MaxFilter(2*radius+1)) if radius>0 else mask.filter(ImageFilter.MinFilter(2*abs(radius)+1)) if radius<0 else mask
    supplied=np.array(changed)>0
    dx,dy=map(int,rng.integers(-6,7,2));shifted=np.zeros_like(supplied)
    ys,xs=np.where(supplied);ys=ys+dy;xs=xs+dx;keep=(ys>=0)&(ys<size)&(xs>=0)&(xs<size);shifted[ys[keep],xs[keep]]=True
    return observed,shifted.astype('float32')[None],true.astype('float32')[None]


def main():
    cfg_path=ROOT/'configs/learned_refiner.json';cfg=json.loads(cfg_path.read_text());local=json.loads((ROOT/'configs/local.json').read_text())
    cache=Path(local['cache'])/'refiner_data';train=np.load(cache/'train.npy',mmap_mode='r');val=np.load(cache/'tuning.npy',mmap_mode='r')
    metadata=json.loads((cache/'metadata.json').read_text());assert hashlib.sha256((ROOT/cfg['manifest']).read_bytes()).hexdigest()==metadata['signature']['manifest_sha256']
    out=ROOT/'outputs/learned_refiner';out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    signature={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['configs/learned_refiner.json','scripts/train_refiner.py','scripts/refiner.py',cfg['manifest']]}
    def batch(indices,seed,data):
        xs=[];ms=[];ts=[]
        for j,index in enumerate(indices):
            x,m,t=sample(data[index],train[(index*37+seed)%len(train)],seed+j*997)
            xs.append(x.transpose(2,0,1));ms.append(m);ts.append(t)
        return tuple(torch.from_numpy(np.stack(a)).cuda() for a in [xs,ms,ts])
    tuning_batches=[batch(list(range(i,min(i+16,len(val)))),90000+i,val) for i in range(0,len(val),16)]
    for seed in cfg['seeds']:
        for variant,cost in cfg['variants'].items():
            folder=out/f'{variant}_{seed}';folder.mkdir(exist_ok=True);final=folder/'training.json'
            if final.exists():assert json.loads(final.read_text())['signature']==signature;continue
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=MaskRefiner(cfg['width']).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate']);scaler=torch.amp.GradScaler('cuda')
            rng=np.random.default_rng(seed);history=[];start_step=0;best=float('inf');latest=folder/'latest.pt'
            if latest.exists():
                state=torch.load(latest,map_location='cpu',weights_only=False);assert state['signature']==signature
                model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler']);rng.bit_generator.state=state['numpy_rng'];start_step=state['step'];best=state['best'];history=state['history']
                torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
            start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            for step in range(start_step,cfg['steps']):
                indices=rng.integers(0,len(train),cfg['batch_size']);x,m,t=batch(indices,seed*1000000+step*cfg['batch_size'],train)
                model.train();optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):loss=refinement_loss(model(x,m),t,cost)
                assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1);scaler.step(optimizer);scaler.update()
                if (step+1)%cfg['validation_interval']==0 or step+1==cfg['steps']:
                    model.eval();scores=[]
                    with torch.inference_mode():
                        for vx,vm,vt in tuning_batches:scores.append(float(refinement_loss(model(vx,vm),vt,cost)))
                    value=float(np.mean(scores));history.append({'step':step+1,'train_loss':float(loss),'tuning_loss':value})
                    if value<best:
                        best=value;torch.save({'model':model.state_dict(),'width':cfg['width'],'seed':seed,'variant':variant,'step':step+1,'signature':signature},folder/'best.pt')
                    state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'step':step+1,'best':best,'history':history,'signature':signature}
                    tmp=folder/'latest.tmp';torch.save(state,tmp);tmp.replace(latest)
                    print(f'{variant} seed={seed} step={step+1}/{cfg["steps"]} tuning_loss={value:.4f}',flush=True)
            result={'seed':seed,'variant':variant,'negative_weight':cost,'steps':cfg['steps'],'history':history,'parameters':sum(p.numel() for p in model.parameters()),'seconds_this_session':time.perf_counter()-start,'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'signature':signature,'limits':'Standard generic and weighted segmentation controls; synthetic supervision, no novel-method claim.'}
            final.write_text(json.dumps(result,indent=2));del model,optimizer,scaler;torch.cuda.empty_cache()


if __name__=='__main__':main()
