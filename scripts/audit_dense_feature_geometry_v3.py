"""Bounded frozen CLIP extraction and matched geometry controls; no training."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import json,time,sys,hashlib
from pathlib import Path
from importlib.metadata import version
import numpy as np
from PIL import Image
import torch
from transformers import CLIPImageProcessor,CLIPVisionModelWithProjection
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from audit_spatial_reference_geometry_v1 import read_image,sha
from spatial_reference_geometry_v1 import sanitize
from dense_feature_geometry_v3 import compare
from sdxl_token_compatibility_probe_v1 import ensure_studio_idle

def main():
    out=ROOT/'outputs/spatial_reference_correspondence_v3/geometry'
    if out.exists():raise ValueError('Preserve previous attempt')
    cache_path=ROOT/'outputs/masked_sdxl_compatibility_pilot_v2/cache/receipt.json'
    old=json.loads((ROOT/'outputs/spatial_reference_correspondence_v1/geometry/receipt.json').read_text())
    assert sha(cache_path)==old['cache_receipt_sha256']
    cache=json.loads(cache_path.read_text());ids=old['training_ids']
    assert cache['train_identities']==ids and [c['identity'] for c in cache['records']]==ids
    assert not set(ids)&set(cache['excluded_identities'])
    for c in cache['records']:
        assert c['role']=='train' and c['target']['identity']==c['identity']
        assert len(c['references'])==4 and all(r['identity']==c['identity'] for r in c['references'])
    local=json.loads((ROOT/'configs/local.json').read_text())
    model_path=Path(local['cache'])/'eye_appearance_models_v1/models/image_encoder'
    assert sha(model_path/'model.safetensors')=='6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030'
    ensure_studio_idle()
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=4_000_000_000,'GPU unavailable or busy'
    paths=['research/DENSE_FEATURE_GEOMETRY_PREFLIGHT_V3.md','scripts/audit_dense_feature_geometry_v3.py','scripts/dense_feature_geometry_v3.py','scripts/spatial_reference_geometry_v1.py','scripts/audit_spatial_reference_geometry_v1.py']
    receipt=dict(status='running',training=False,source_hashes={f:sha(ROOT/f) for f in paths},cache_sha256=sha(cache_path),model_sha256=sha(model_path/'model.safetensors'),model_config_sha256=sha(model_path/'config.json'),versions={k:version(k) for k in ['torch','transformers','numpy','opencv-python']},rows=[],wrong_person_rows=[],extractions=[])
    out.mkdir(parents=True);start=time.monotonic()
    def save():(out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False))
    save()
    try:
        torch.set_num_threads(4);torch.manual_seed(20261011);torch.cuda.reset_peak_memory_stats()
        model=CLIPVisionModelWithProjection.from_pretrained(str(model_path),local_files_only=True,torch_dtype=torch.float16).eval().requires_grad_(False).to('cuda')
        receipt['frozen_parameters']=sum(p.numel() for p in model.parameters())
        receipt['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert receipt['trainable_parameters']==0
        processor=CLIPImageProcessor();receipt['processor']=processor.to_dict()
        def extract(rgb,mask=None):
            ensure_studio_idle()
            if mask is not None:rgb=sanitize(rgb,mask)
            pixels=processor(images=Image.fromarray(rgb),return_tensors='pt').pixel_values.to('cuda',dtype=torch.float16)
            with torch.inference_mode():features=model.vision_model(pixels,output_hidden_states=True).hidden_states[-2][0,1:].float().cpu().numpy()
            assert features.shape==(256,1280) and np.isfinite(features).all()
            peak=torch.cuda.max_memory_allocated()
            assert peak<=3_500_000_000,'Fixed extraction memory cap exceeded'
            receipt['peak_allocated_bytes']=peak
            receipt['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
            return features
        banks={};masks={}
        for index,c in enumerate(cache['records']):
            mask=np.zeros((512,512),bool);x0,y0,x1,y1=c['mask']['bbox_xyxy_exclusive'];mask[y0:y1,x0:x1]=True
            assert hashlib.sha256(mask.tobytes()).hexdigest()==c['mask']['binary_sha256']
            observed=sanitize(read_image(c['target']),mask)
            target=extract(observed,mask)
            if index==0:
                alternate=observed.copy();alternate[mask]=37
                assert np.array_equal(target,extract(alternate,mask)),'Hidden-pixel feature invariance failed'
                receipt['first_target_probe']='passed';receipt['hidden_pixel_invariance']='exact';save()
                print('First-image GPU/invariance probe passed; peak',receipt['peak_allocated_bytes'],flush=True)
            refs=np.stack([extract(read_image(r)) for r in c['references']])
            banks[c['identity']]=(target,refs);masks[c['identity']]=mask
            p=out/(c['identity']+'_features.npz');np.savez_compressed(p,target=target,references=refs)
            receipt['extractions'].append(dict(identity=c['identity'],file=p.name,sha256=sha(p)));save()
            print('Encoded',c['identity'],flush=True)
        del model;torch.cuda.empty_cache()
        for index,ident in enumerate(ids):
            target,refs=banks[ident];other=ids[(index+1)%len(ids)]
            rows=[dict(index=i,**compare(target,ref,masks[ident])) for i,ref in enumerate(refs)]
            wrong=[dict(index=i,**compare(target,ref,masks[ident])) for i,ref in enumerate(banks[other][1])]
            receipt['rows'].append(dict(identity=ident,references=rows,accepted_references=sum(r['eligible'] for r in rows)))
            receipt['wrong_person_rows'].append(dict(identity=ident,reference_identity=other,references=wrong))
            print(ident,sum(r['eligible'] for r in rows),'correct;',sum(r['eligible'] for r in wrong),'wrong references pass',flush=True);save()
        receipt['usable_cases']=sum(r['accepted_references']>=2 for r in receipt['rows'])
        receipt['wrong_pair_acceptance']=sum(r['eligible'] for row in receipt['wrong_person_rows'] for r in row['references'])/32
        receipt['gate_passed']=receipt['usable_cases']==8 and receipt['wrong_pair_acceptance']<=.1
        receipt['status']='complete'
    except Exception as exc:
        receipt.update(status='failed',error=repr(exc));raise
    finally:
        receipt['elapsed_seconds']=time.monotonic()-start;save()

if __name__=='__main__':main()
