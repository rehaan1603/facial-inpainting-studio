# Studio-only resolution experiment derived from frozen reference_inpaint.py SHA256 6fa4fe9c7742c8a1ec69d1586cc98a07573b1b85e5bebf0ad4c899b75a782daf.
# Only adds 256 to accepted model resolutions; historical research generator remains unchanged.
"""Local multi-photo conditioning baseline: SDXL inpainting + FaceID Portrait.

References are supplied by the user as photos of one person. This does not search
for or identify a person in a database. No network access is needed for inference.
"""
import argparse,hashlib,json,os,time
from importlib.metadata import version
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import cv2
import numpy as np
from PIL import Image,ImageOps
import torch
from diffusers import StableDiffusionXLInpaintPipeline,DDIMScheduler
from insightface.app import FaceAnalysis
from reference_blending import harmonize_reference
from atomic_records import write_json
ROOT=Path(__file__).resolve().parents[1]
CACHE=Path(json.loads((ROOT/'configs/local.json').read_text())['cache'])

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_rgb(path):
    with Image.open(path) as im:return ImageOps.exif_transpose(im).convert('RGB').copy()

def reconstruct(image,mask,references,output,seed=17,steps=30,scale=.8,memory='model',progress_path=None,blend='poisson',strength=1.0,runtime=None,model_resolution=512):
    def progress(message,**extra):
        print(message,flush=True)
        if progress_path:
            # A progress-file lock must not discard an otherwise valid GPU run.
            write_json(progress_path,{'message':message,**extra},required=False)
    if not 1<=len(references)<=4:raise ValueError('Supply between one and four reference images.')
    if not 10<=steps<=50 or not 0<=scale<=1.5:raise ValueError('Unsupported sampling settings.')
    if not .5<=strength<=1:raise ValueError('Strength must be between 0.5 and 1.0.')
    if blend not in ['poisson','hard']:raise ValueError('Choose supported edge blending.')
    if model_resolution not in [256,512,1024]:raise ValueError('Model resolution must be 256, 512 or 1024.')
    source=load_rgb(image)
    with Image.open(mask) as im:supplied=im.convert('L')
    if source.size!=supplied.size:raise ValueError('Input and mask dimensions must match.')
    source=source.resize((512,512),Image.Resampling.LANCZOS);binary=np.asarray(supplied.resize((512,512),Image.Resampling.NEAREST))>=128
    if not binary.any():raise ValueError('Mark the area to reconstruct first.')
    mask_image=Image.fromarray(binary.astype('uint8')*255)
    output=Path(output)
    destinations=[output,output.with_suffix('.json'),*[output.with_name(output.stem+suffix) for suffix in ['_raw.png','_hard.png','_effective_mask.png','_input.png']]]
    if progress_path:destinations.append(Path(progress_path))
    inputs={Path(p).resolve() for p in [image,mask,*references]}
    if any(p.resolve() in inputs for p in destinations):raise ValueError('Output files must not overwrite any input.')
    if len({p.resolve() for p in destinations})!=len(destinations):raise ValueError('Output and progress paths must be distinct.')
    if len({sha(p) for p in references})!=len(references):raise ValueError('Use different reference photographs, not duplicate files.')
    torch.set_num_threads(4)
    progress('Checking faces in the reference photos…')
    face_dir=CACHE/'reference_models/insightface/models/buffalo_l'
    if not face_dir.is_dir():raise ValueError('Reference face model is not installed.')
    analyser=runtime.get('analyser') if runtime is not None else None
    if analyser is None:
        analyser=FaceAnalysis(name=str(face_dir),allowed_modules=['detection','recognition'],providers=['CPUExecutionProvider'])
        analyser.prepare(ctx_id=-1,det_size=(640,640))
        if runtime is not None:runtime['analyser']=analyser
    embeddings=[];reference_info=[]
    for index,path in enumerate(references):
        rgb=np.asarray(load_rgb(path));faces=analyser.get(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        if len(faces)!=1:raise ValueError(f'Reference {index+1} must contain one clear, visible face. Found {len(faces)}.')
        embedding=faces[0].normed_embedding
        if embedding is None or not np.isfinite(embedding).all():raise ValueError(f'Reference {index+1} could not be encoded.')
        embeddings.append(embedding);reference_info.append({'sha256':sha(path),'detection_score':float(faces[0].det_score)})
    del analyser
    reference_tensor=torch.from_numpy(np.stack(embeddings)).float()[None]
    # Each reference contributes its own learned tokens. Do not average unlike poses away.
    conditioning=torch.cat([torch.zeros_like(reference_tensor),reference_tensor]).to(dtype=torch.float16)
    progress('Loading the reference-guided inpainting model…')
    pipeline=runtime.get('pipeline') if runtime is not None else None
    if pipeline is None:
        pipeline=StableDiffusionXLInpaintPipeline.from_pretrained(str(CACHE/'osor_models/sdxl'),torch_dtype=torch.float16,local_files_only=True,add_watermarker=False)
        pipeline.scheduler=DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.load_ip_adapter(str(CACHE/'reference_models'),subfolder='',weight_name='ip-adapter-faceid-portrait_sdxl.bin',image_encoder_folder=None,local_files_only=True)
        if memory=='sequential':pipeline.enable_sequential_cpu_offload()
        else:pipeline.enable_model_cpu_offload()
        if runtime is not None:runtime.update(pipeline=pipeline,memory=memory)
    elif runtime['memory']!=memory:raise ValueError('A reused inference runtime must keep the same memory mode.')
    # Tile only the high-resolution VAE to keep decoding within laptop VRAM.
    if model_resolution==1024:pipeline.vae.enable_tiling()
    else:pipeline.vae.disable_tiling()
    pipeline.set_ip_adapter_scale(scale)
    projection=pipeline.unet.encoder_hid_proj.image_projection_layers[0]
    for processor in pipeline.unet.attn_processors.values():
        if hasattr(processor,'num_tokens'):processor.num_tokens=[projection.num_tokens]
    sampling_steps=int(steps*strength)
    def callback(pipe,step,timestep,kwargs):progress(f'Reconstructing with {len(references)} reference photos: step {step+1} of {sampling_steps}',step=step+1,total=sampling_steps);return kwargs
    generator=torch.Generator(device='cpu').manual_seed(seed);start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    model_source=source.resize((model_resolution,model_resolution),Image.Resampling.LANCZOS)
    model_mask=mask_image.resize((model_resolution,model_resolution),Image.Resampling.NEAREST)
    generated=pipeline(prompt='A realistic portrait photograph of a person, natural facial features, consistent lighting, detailed skin.',negative_prompt='painting, drawing, distorted face, deformed eyes, blurry, extra facial features, low quality',image=model_source,mask_image=model_mask,ip_adapter_image_embeds=[conditioning],height=model_resolution,width=model_resolution,num_inference_steps=steps,strength=strength,guidance_scale=5.0,generator=generator,callback_on_step_end=callback,output_type='np').images[0]
    if generated.shape!=(model_resolution,model_resolution,3) or not np.isfinite(generated).all():raise RuntimeError('The model returned invalid pixels.')
    generated=(generated.clip(0,1)*255).round().astype('uint8')
    generated=np.asarray(Image.fromarray(generated).resize((512,512),Image.Resampling.LANCZOS));observed=np.asarray(source)
    hard,_=harmonize_reference(generated,observed,binary,mode='hard')
    result,blend_info=harmonize_reference(generated,observed,binary,mode=blend)
    assert np.array_equal(result[~binary],observed[~binary])
    output.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(result).save(output)
    Image.fromarray(generated).save(output.with_name(output.stem+'_raw.png'));Image.fromarray(hard).save(output.with_name(output.stem+'_hard.png'))
    mask_image.save(output.with_name(output.stem+'_effective_mask.png'));source.save(output.with_name(output.stem+'_input.png'))
    metadata={'backbone':'sdxl_faceid_portrait','resolution':[512,512],'seed':seed,'steps':steps,'adapter_scale':scale,'references':reference_info,'reference_count':len(references),'reference_tokens_per_image':int(projection.num_tokens),'memory_mode':memory,'inference_seconds_including_offload':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'adapter_sha256':sha(CACHE/'reference_models/ip-adapter-faceid-portrait_sdxl.bin'),'outside_effective_mask':'Exact preservation at processed input resolution','scope':'Existing reference-guided baseline, not a novel method or verified recovery of the true hidden face.'}
    metadata.update(strength=strength,sampling_steps=sampling_steps,blending=blend_info,input_sha256=sha(image),mask_sha256=sha(mask),raw_output_sha256=sha(output.with_name(output.stem+'_raw.png')),result_sha256=sha(output))
    metadata.update(model_resolution=[model_resolution,model_resolution],vae_tiling=model_resolution==1024,environment={name:version(name) for name in ['torch','diffusers','transformers','tokenizers','regex','onnx','ml_dtypes','insightface']})
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2));progress('Reference-guided result ready',complete=True)
    return metadata

def main():
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--mask',type=Path,required=True);p.add_argument('--references',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=17);p.add_argument('--steps',type=int,default=30);p.add_argument('--scale',type=float,default=.8);p.add_argument('--memory',choices=['model','sequential'],default='model');p.add_argument('--progress',type=Path);p.add_argument('--blend',choices=['poisson','hard'],default='poisson');p.add_argument('--strength',type=float,default=1.0);p.add_argument('--model-resolution',type=int,choices=[256,512,1024],default=512);args=p.parse_args()
    if args.output.resolve() in [p.resolve() for p in [args.image,args.mask,*args.references]]:p.error('Output must not overwrite an input.')
    try:reconstruct(args.image,args.mask,args.references,args.output,args.seed,args.steps,args.scale,args.memory,args.progress,args.blend,args.strength,model_resolution=args.model_resolution)
    except Exception as error:
        if args.progress:write_json(args.progress,{'error':str(error),'message':'Reference-guided inpainting failed.'},required=False)
        raise
if __name__=='__main__':main()
