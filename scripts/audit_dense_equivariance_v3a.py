"""Known geometry diagnostic with sanitized visible-only training inputs."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import sys,json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor,CLIPVisionModelWithProjection
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from audit_spatial_reference_geometry_v1 import sha,read_image
from spatial_reference_geometry_v1 import sanitize
from dense_feature_geometry_v3 import grid_selection,mutual_matches,fit_pairs
from sdxl_token_compatibility_probe_v1 import ensure_studio_idle

def analyze(a,b,mask,shift):
    moved=np.ones_like(mask);moved[:,shift:]=mask[:,:512-shift] if shift else mask
    coords,tk=grid_selection(mask);_,rk=grid_selection(moved if shift else mask)
    if shift:rk &= coords[:,0]>=shift+32
    pairs=mutual_matches(a[tk],b[rk]);xy=np.array([coords[tk][i] for i,j in pairs]);uv=np.array([coords[rk][j] for i,j in pairs])
    result=fit_pairs(xy,uv)
    if 'transform' in result:
        m=np.array(result['transform']);source=coords[rk]
        expected=source-np.array([shift,0]);actual=source@m[:,:2].T+m[:,2]
        result['known_transform_rms_pixels']=float(np.sqrt(((actual-expected)**2).sum(1).mean()))
    return result

def main():
    oldroot=ROOT/'outputs/spatial_reference_correspondence_v3/geometry'
    old=json.loads((oldroot/'receipt.json').read_text());assert old['status']=='complete'
    cache_path=ROOT/'outputs/masked_sdxl_compatibility_pilot_v2/cache/receipt.json';assert sha(cache_path)==old['cache_sha256']
    cache=json.loads(cache_path.read_text());assert [c['identity'] for c in cache['records']]==[x['identity'] for x in old['extractions']]
    out=ROOT/'outputs/spatial_reference_correspondence_v3a_retry';out.mkdir(exist_ok=False)
    receipt=dict(status='running',training=False,source_receipt_sha256=sha(oldroot/'receipt.json'),protocol_sha256=sha(ROOT/'research/DENSE_EQUIVARIANCE_PREFLIGHT_V3A.md'),script_sha256=sha(__file__),rows=[])
    def save():(out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False))
    save();start=time.monotonic()
    try:
        cases=[]
        for c,pin in zip(cache['records'],old['extractions']):
            assert sha(oldroot/pin['file'])==pin['sha256']
            features=np.load(oldroot/pin['file'])['target'];mask=np.zeros((512,512),bool);x0,y0,x1,y1=c['mask']['bbox_xyxy_exclusive'];mask[y0:y1,x0:x1]=True
            identity=analyze(features,features,mask,0);assert identity['eligible'] and identity['known_transform_rms_pixels']<1e-4
            receipt['rows'].append(dict(identity=c['identity'],shift=0,**identity));cases.append((c,features,mask))
        receipt['identity_controls_passed']=8;save();ensure_studio_idle()
        assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=4_000_000_000
        local=json.loads((ROOT/'configs/local.json').read_text());model_path=Path(local['cache'])/'eye_appearance_models_v1/models/image_encoder'
        assert sha(model_path/'model.safetensors')==old['model_sha256']
        torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
        model=CLIPVisionModelWithProjection.from_pretrained(str(model_path),local_files_only=True,torch_dtype=torch.float16).eval().requires_grad_(False).to('cuda')
        processor=CLIPImageProcessor();assert json.loads(json.dumps(processor.to_dict()))==old['processor']
        for c,original,mask in cases:
            ensure_studio_idle();observed=sanitize(read_image(c['target']),mask)
            for shift in [8,32]:
                rgb=np.full_like(observed,128);rgb[:,shift:]=observed[:,:512-shift]
                pixels=processor(images=Image.fromarray(rgb),return_tensors='pt').pixel_values.to('cuda',dtype=torch.float16)
                with torch.inference_mode():feature=model.vision_model(pixels,output_hidden_states=True).hidden_states[-2][0,1:].float().cpu().numpy()
                assert np.isfinite(feature).all() and torch.cuda.max_memory_allocated()<=3_500_000_000
                result=analyze(original,feature,mask,shift);receipt['rows'].append(dict(identity=c['identity'],shift=shift,**result));save()
                print(c['identity'],shift,result.get('eligible'),result.get('known_transform_rms_pixels'),flush=True)
        receipt.update(status='complete',peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except Exception as exc:receipt.update(status='failed',error=repr(exc));raise
    finally:receipt['elapsed_seconds']=time.monotonic()-start;save()

if __name__=='__main__':main()
