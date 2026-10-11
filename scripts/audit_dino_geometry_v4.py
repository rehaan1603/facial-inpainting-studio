"""Frozen DINO feature feasibility, real and known-shift controls. No training."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import json,sys,time,hashlib
from pathlib import Path
from importlib.metadata import version
import numpy as np
from PIL import Image
import torch
from transformers import Dinov2Model
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from audit_spatial_reference_geometry_v1 import sha,read_image
from spatial_reference_geometry_v1 import sanitize
from sdxl_token_compatibility_probe_v1 import ensure_studio_idle
from dino_geometry_v4 import compare

def main():
    out=ROOT/'outputs/spatial_reference_correspondence_v4/geometry'
    if out.exists():raise ValueError('Preserve previous attempt')
    oldpath=ROOT/'outputs/spatial_reference_correspondence_v3/geometry/receipt.json';old=json.loads(oldpath.read_text())
    cachepath=ROOT/'outputs/masked_sdxl_compatibility_pilot_v2/cache/receipt.json';assert sha(cachepath)==old['cache_sha256']
    cache=json.loads(cachepath.read_text());ids=['6792','2250','2083','4164','674','476','1281','8748']
    assert cache['train_identities']==ids and [c['identity'] for c in cache['records']]==ids and not set(ids)&set(cache['excluded_identities'])
    for c in cache['records']:
        assert c['role']=='train' and c['target']['identity']==c['identity'] and len(c['references'])==4 and all(r['identity']==c['identity'] for r in c['references'])
    modelpath=ROOT/'outputs/dense_geometry_assets_v4/dinov2-small';assets=json.loads((modelpath/'receipt.json').read_text())
    for f in assets['files']:assert sha(modelpath/f['file'])==f['sha256']
    assert assets['revision']=='ed25f3a31f01632728cabb09d1542f84ab7b0056'
    ensure_studio_idle();assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=3_000_000_000
    sources=['research/DENSE_DINO_GEOMETRY_PREFLIGHT_V4.md','scripts/audit_dino_geometry_v4.py','scripts/dino_geometry_v4.py','scripts/dense_feature_geometry_v3.py','scripts/spatial_reference_geometry_v1.py','scripts/audit_spatial_reference_geometry_v1.py']
    receipt=dict(status='running',training=False,assets=assets,source_hashes={f:sha(ROOT/f) for f in sources},cache_sha256=sha(cachepath),versions={k:version(k) for k in ['torch','transformers','numpy','Pillow','opencv-python']},rows=[],wrong_person_rows=[],controls=[],features=[])
    out.mkdir(parents=True);start=time.monotonic()
    def save():(out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False))
    save()
    try:
        torch.set_num_threads(4);torch.manual_seed(20261011);torch.cuda.reset_peak_memory_stats()
        model=Dinov2Model.from_pretrained(str(modelpath),local_files_only=True,torch_dtype=torch.float16,trust_remote_code=False).eval().requires_grad_(False).to('cuda')
        receipt['frozen_parameters']=sum(p.numel() for p in model.parameters());receipt['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert receipt['trainable_parameters']==0
        config=json.loads((modelpath/'preprocessor_config.json').read_text());mean=np.array(config['image_mean'],np.float32);std=np.array(config['image_std'],np.float32)
        receipt['preprocessing']=dict(resize=518,resample='PIL bicubic',center_crop=False,mean=mean.tolist(),std=std.tolist())
        def extract(rgb,mask=None):
            if mask is not None:rgb=sanitize(rgb,mask)
            rgb=np.asarray(Image.fromarray(rgb).resize((518,518),Image.Resampling.BICUBIC),np.float32)/255
            pixels=torch.from_numpy(((rgb-mean)/std).transpose(2,0,1).copy())[None].to('cuda',dtype=torch.float16)
            with torch.inference_mode():feature=model(pixel_values=pixels).last_hidden_state[0,1:].float().cpu().numpy()
            assert feature.shape==(1369,384) and np.isfinite(feature).all()
            assert torch.cuda.max_memory_allocated()<=2_000_000_000,'Fixed memory cap exceeded'
            receipt['peak_allocated_bytes']=torch.cuda.max_memory_allocated();receipt['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
            return feature
        banks={};masks={}
        for index,c in enumerate(cache['records']):
            ensure_studio_idle();mask=np.zeros((512,512),bool);x0,y0,x1,y1=c['mask']['bbox_xyxy_exclusive'];mask[y0:y1,x0:x1]=True
            assert hashlib.sha256(mask.tobytes()).hexdigest()==c['mask']['binary_sha256']
            observed=sanitize(read_image(c['target']),mask);target=extract(observed,mask)
            if index==0:
                alternate=observed.copy();alternate[mask]=37
                assert np.array_equal(target,extract(alternate,mask))
                receipt['first_target_probe']='passed';receipt['hidden_pixel_invariance']='exact';save()
                print('First probe passed',receipt['peak_allocated_bytes'],'bytes',flush=True)
            refs=np.stack([extract(read_image(r)) for r in c['references']]);banks[c['identity']]=(target,refs);masks[c['identity']]=mask
            path=out/(c['identity']+'_features.npz');np.savez_compressed(path,target=target,references=refs);receipt['features'].append(dict(identity=c['identity'],file=path.name,sha256=sha(path)))
            receipt['controls'].append(dict(identity=c['identity'],shift=0,**compare(target,target,mask,mask)))
            for shift in [8,32]:
                rgb=np.full_like(observed,128);rgb[:,shift:]=observed[:,:512-shift];moved=np.ones_like(mask);moved[:,shift:]=mask[:,:512-shift]
                feature=extract(rgb);receipt['controls'].append(dict(identity=c['identity'],shift=shift,**compare(target,feature,mask,moved,shift)))
            save();print('Encoded and controlled',c['identity'],flush=True)
        del model;torch.cuda.empty_cache()
        for index,ident in enumerate(ids):
            target,refs=banks[ident];other=ids[(index+1)%len(ids)]
            rows=[dict(index=i,**compare(target,ref,masks[ident])) for i,ref in enumerate(refs)]
            wrong=[dict(index=i,**compare(target,ref,masks[ident])) for i,ref in enumerate(banks[other][1])]
            receipt['rows'].append(dict(identity=ident,references=rows,accepted_references=sum(r['eligible'] for r in rows)))
            receipt['wrong_person_rows'].append(dict(identity=ident,reference_identity=other,references=wrong))
            print(ident,sum(r['eligible'] for r in rows),'correct;',sum(r['eligible'] for r in wrong),'wrong pass',flush=True);save()
        receipt['usable_cases']=sum(r['accepted_references']>=2 for r in receipt['rows']);receipt['wrong_pair_acceptance']=sum(r['eligible'] for row in receipt['wrong_person_rows'] for r in row['references'])/32
        receipt['gate_passed']=receipt['usable_cases']==8 and receipt['wrong_pair_acceptance']<=.1;receipt['status']='complete'
    except Exception as exc:receipt.update(status='failed',error=repr(exc));raise
    finally:receipt['elapsed_seconds']=time.monotonic()-start;save()

if __name__=='__main__':main()
