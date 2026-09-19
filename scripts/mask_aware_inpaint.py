"""Experimental reference selection before the unchanged FaceID generator."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import sha,write_new
from src.reference_selection.reference_analyzer import ReferenceAnalyzer
from src.reference_selection.selection_baselines import POLICIES,select

def reconstruct_selected(image,mask,references,output,policy='mask_aware',seed=17,steps=30,scale=.8,
                         strength=.99,blend='poisson',model_resolution=512,progress_path=None,
                         runtime=None,analyzer=None,analysis=None,fallback='error'):
    output=Path(output)
    inputs={Path(p).resolve() for p in [image,mask,*references]}
    destinations=[output,output.with_suffix('.json'),output.with_name(output.stem+'_selection.json'),
                  *[output.with_name(output.stem+s) for s in ['_raw.png','_hard.png','_effective_mask.png','_input.png']]]
    if any(p.resolve() in inputs or p.exists() for p in destinations):raise ValueError('Use a new output path; inputs and existing results are protected')
    analysis=analysis if analysis is not None else (analyzer or ReferenceAnalyzer()).analyze(image,mask,references)
    if analysis['observed_sha256']!=sha(image) or analysis['mask_sha256']!=sha(mask):raise ValueError('Stale input analysis')
    if [str(Path(p).resolve()) for p in references]!=[r['path'] for r in analysis['references']]:raise ValueError('Stale reference list')
    for r in analysis['references']:
        if r['valid'] and sha(r['path'])!=r['sha256']:raise ValueError('Reference changed since analysis')
    chosen=select(analysis,policy,seed);record=dict(analysis,policy=policy,seed=seed,selected_paths=chosen)
    write_new(output.with_name(output.stem+'_selection.json'),record)
    if not chosen:
        if fallback!='lama':raise ValueError('No valid reference photos. Supply clear single-face photos or explicitly select the LaMa fallback.')
        import numpy as np
        from PIL import Image,ImageOps
        from inpaint import Inpainter
        with Image.open(image) as im:observed=np.asarray(ImageOps.exif_transpose(im).convert('RGB').resize((512,512),Image.Resampling.LANCZOS))
        with Image.open(mask) as im:binary=np.asarray(im.convert('L').resize((512,512),Image.Resampling.NEAREST))>=128
        start=time.perf_counter();pred=Inpainter('lama')(observed.astype('float32')/255,binary,seed)
        result=np.rint(pred*255).clip(0,255).astype('uint8');result[~binary]=observed[~binary]
        Image.fromarray(result).save(output)
        metadata={'backbone':'lama','explicit_fallback':True,'fallback_reason':'No valid references','inference_seconds_including_offload':time.perf_counter()-start,'result_sha256':sha(output),'input_sha256':sha(image),'mask_sha256':sha(mask)}
        write_new(output.with_suffix('.json'),dict(metadata,reference_selection=record))
        return metadata
    from reference_inpaint import reconstruct
    metadata=reconstruct(image,mask,chosen,output,seed=seed,steps=steps,scale=0 if policy=='zero_scale' else scale,
                         strength=strength,blend=blend,model_resolution=model_resolution,progress_path=progress_path,runtime=runtime)
    metadata['reference_selection']=record
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf-8')
    return metadata

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['image','mask','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--references',type=Path,nargs='+',required=True);p.add_argument('--policy',choices=POLICIES,default='mask_aware')
    p.add_argument('--seed',type=int,default=17);p.add_argument('--steps',type=int,default=30);p.add_argument('--scale',type=float,default=.8)
    p.add_argument('--strength',type=float,default=.99);p.add_argument('--blend',choices=['poisson','hard'],default='poisson')
    p.add_argument('--model-resolution',type=int,choices=[512,1024],default=512);p.add_argument('--progress',type=Path)
    p.add_argument('--fallback',choices=['error','lama'],default='error');a=p.parse_args()
    try:reconstruct_selected(a.image,a.mask,a.references,a.output,a.policy,a.seed,a.steps,a.scale,a.strength,a.blend,a.model_resolution,a.progress,fallback=a.fallback)
    except Exception as error:
        if a.progress:
            from atomic_records import write_json
            write_json(a.progress,{'error':str(error),'message':'Reference selection failed'},required=False)
        raise

if __name__=='__main__':main()
