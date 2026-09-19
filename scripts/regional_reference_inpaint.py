"""Experimental spatial routing; runtime photographs only, no target lookup."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageOps
from src.research_integrity import ROOT,sha,write_new
from src.reference_fusion.feature_bank import analyze_runtime
from src.reference_fusion.regional_fusion import POLICIES,build_masks

def reconstruct_routed(image,mask,references,output,policy='regional',seed=17,steps=30,scale=.8,
                       strength=.99,blend='poisson',runtime=None,progress_path=None,bank=None):
    output=Path(output);refs=[Path(p) for p in references]
    inputs={Path(p).resolve() for p in [image,mask,*refs]}
    destinations=[output,output.with_suffix('.json'),output.with_name(output.stem+'_routing.json'),output.with_name(output.stem+'_routing.npz'),
                  *[output.with_name(output.stem+s) for s in ['_input.png','_effective_mask.png','_raw.png','_hard.png']]]
    if any(p.resolve() in inputs or p.exists() for p in destinations):raise ValueError('Use a new output path; existing inputs/results are protected')
    analysis,geometry,analyzer=bank if bank is not None else analyze_runtime(image,mask,refs)
    if analysis['observed_sha256']!=sha(image) or analysis['mask_sha256']!=sha(mask):raise ValueError('Stale analysis')
    if [str(p.resolve()) for p in refs]!=[r['path'] for r in analysis['references']]:raise ValueError('Reference list changed')
    valid=[r for r in analysis['references'] if r['valid']]
    for r in valid:
        if sha(r['path'])!=r['sha256']:raise ValueError('Reference changed')
    with Image.open(mask) as im:binary=np.asarray(im.convert('L').resize((512,512),Image.Resampling.NEAREST))>=128
    # Geometry is measured on the input; the generator processes 512 square pixels.
    with Image.open(image) as im:width,height=ImageOps.exif_transpose(im).size
    geometry=dict(geometry)
    if geometry['bbox'] is not None:
        geometry['bbox']=(np.asarray(geometry['bbox'])*[512/width,512/height,512/width,512/height]).tolist()
        geometry['landmarks']=(np.asarray(geometry['landmarks'])*[512/width,512/height]).tolist()
    maps,details=build_masks(analysis,binary,geometry,policy)
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output.with_name(output.stem+'_routing.npz'),weights=maps)
    routing=dict(details,analysis=analysis,damaged_input_geometry=geometry,
                 map_sha256=sha(output.with_name(output.stem+'_routing.npz')),
                 note='Native per-reference attention averaging differs from original concatenated-token attention')
    write_new(output.with_name(output.stem+'_routing.json'),routing)
    from reference_inpaint import reconstruct,CACHE
    from diffusers import StableDiffusionXLInpaintPipeline,DDIMScheduler
    import torch
    runtime=runtime if runtime is not None else {}
    if 'base_pipeline' not in runtime:
        pipeline=StableDiffusionXLInpaintPipeline.from_pretrained(str(CACHE/'osor_models/sdxl'),torch_dtype=torch.float16,local_files_only=True,add_watermarker=False)
        pipeline.scheduler=DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.load_ip_adapter(str(CACHE/'reference_models'),subfolder='',weight_name='ip-adapter-faceid-portrait_sdxl.bin',image_encoder_folder=None,local_files_only=True)
        pipeline.enable_model_cpu_offload();runtime['base_pipeline']=pipeline
    from src.reference_fusion.fusion_adapter import RoutedPipeline
    pipeline=runtime['base_pipeline']
    delegated={'pipeline':pipeline if policy=='concat' else RoutedPipeline(pipeline,maps),'memory':'model','analyser':analyzer.detector}
    metadata=reconstruct(image,mask,[r['path'] for r in valid],output,seed=seed,steps=steps,scale=scale,strength=strength,blend=blend,progress_path=progress_path,runtime=delegated)
    metadata['regional_routing']=routing
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf-8')
    return metadata

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['image','mask','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--references',type=Path,nargs='+',required=True);p.add_argument('--policy',choices=POLICIES,default='regional')
    p.add_argument('--seed',type=int,default=17);p.add_argument('--steps',type=int,default=30);p.add_argument('--scale',type=float,default=.8)
    p.add_argument('--strength',type=float,default=.99);p.add_argument('--blend',choices=['poisson','hard'],default='poisson');p.add_argument('--progress',type=Path)
    a=p.parse_args()
    try:reconstruct_routed(a.image,a.mask,a.references,a.output,a.policy,a.seed,a.steps,a.scale,a.strength,a.blend,progress_path=a.progress)
    except Exception as error:
        if a.progress:
            from atomic_records import write_json
            write_json(a.progress,{'error':str(error),'message':'Regional routing failed'},required=False)
        raise

if __name__=='__main__':main()
