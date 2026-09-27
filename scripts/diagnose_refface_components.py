"""Post-result descriptive instrumentation; no parameter tuning or clean pixels.

Observe the author's component style gate and verify that hooks leave every
saved native prediction unchanged. These repeat forwards are not new examples.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, write_new
from src.preservation.refface_baseline import RefFaceBaseline, rgb_tensor, conservative_mask


def main():
    cache=Path(json.loads((ROOT/'configs/local.json').read_text())['cache'])
    base=ROOT/'outputs/refface_baseline_v1'
    signature=json.loads((base/'signature.json').read_text())
    assert signature['wrapper_sha256']==sha(ROOT/'src/preservation/refface_baseline.py')
    for name,digest in signature['weight_sha256'].items():assert sha(cache/'refface_weights_v1'/name)==digest
    cfg=json.loads((ROOT/'research/protocols/refface_baseline_v1.json').read_text())
    manifest=ROOT/'outputs/distortion_aware_v1/inference_manifest.json'
    assert sha(manifest)==signature['inference_manifest_sha256']
    cases={c['case_id']:c for c in json.loads(manifest.read_text())['cases']}
    generation=json.loads((base/'comparison.json').read_text())
    selected=[r for r in generation['rows'] if not r.get('reused_control') and r['status']=='complete']
    torch.set_num_threads(4);torch.manual_seed(17)
    model=RefFaceBaseline(cache/'refface_source_v1',cache/'refface_parser_source_v1',cache/'refface_weights_v1')
    captured={}
    def hook(module,args,output):captured['output']=output
    handle=model.generator.register_forward_hook(hook)
    rows=[]
    for r in selected:
        c=cases[r['case_id']];assert c['identity'] not in cfg['reserved_identities']
        with Image.open(c['observed']) as im:original=np.asarray(im.convert('RGB'))
        with Image.open(c['mask']) as im:missing=np.asarray(im.convert('L'))>=128
        with Image.open(c['references'][r['reference_index']]['path']) as im:reference=im.convert('RGB')
        safe=np.where(missing[...,None],np.uint8(128),original)
        obs=rgb_tensor(Image.fromarray(safe),256,'cuda');ref=rgb_tensor(reference,256,'cuda')
        mask=conservative_mask(missing).cuda();labels=model.reference_labels(reference)
        with torch.no_grad():raw=model.native(obs,mask,ref,labels)
        array=((raw[0].clamp(-1,1)+1)*127.5).permute(1,2,0).cpu().numpy().astype(np.uint8)
        native=Path(r['output']).with_name(Path(r['output']).stem+'_native.png')
        assert sha(native)==r['native_sha256']
        with Image.open(native) as im:assert np.array_equal(array,np.asarray(im.convert('RGB')))
        result=captured.pop('output');layers=[]
        for seg,before,after in zip(result[2],result[3],result[4]):
            m=F.interpolate(mask,size=seg.shape[-2:],mode='nearest').bool()
            components=[]
            for component in [2,3,4,5,12,13]:
                region=seg[:,component:component+1].bool()
                hole=region&m;visible=region&~m
                value=lambda x:float(x.abs().masked_select(hole).sum())
                components.append({'label':component,'predicted_pixels':int(region.sum()),
                    'predicted_missing_pixels':int(hole.sum()),'predicted_visible_pixels':int(visible.sum()),
                    'style_abs_sum_before_in_hole':value(before),'style_abs_sum_after_in_hole':value(after)})
            layers.append({'resolution':list(seg.shape[-2:]),'components':components})
        rows.append({'key':r['key'],'native_pixels_bit_exact':True,'layers':layers})
    handle.remove()
    eligible=[c for r in rows for layer in r['layers'] for c in layer['components']
              if c['predicted_missing_pixels'] and c['style_abs_sum_before_in_hole']>0]
    dropped=[c for c in eligible if c['style_abs_sum_after_in_hole']==0]
    receipt={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'post_result_diagnostic':True,
        'repeat_forwards':len(rows),'all_saved_native_pixels_bit_exact':True,
        'eligible_component_layer_events':len(eligible),'fully_suppressed_component_layer_events':len(dropped),
        'suppressed_with_visible_component_pixels':sum(c['predicted_visible_pixels']>0 for c in dropped),
        'rows':rows,'script_sha256':sha(__file__),'baseline_signature_sha256':sha(base/'signature.json'),
        'clean_target_or_gallery_pixels_accessed':False,'reserved_final_pixels_accessed':False,
        'interpretation':'This measures the author gate on its own predicted components; not true anatomical segmentation. It does not prove the gate caused poor reconstruction, nor that changing it will help. No weights, outputs or parameters changed.'}
    write_new(ROOT/'research/refface_component_diagnostic_v1.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
