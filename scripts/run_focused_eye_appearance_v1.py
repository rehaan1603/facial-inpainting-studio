"""One frozen addition: reference CLIP appearance alongside unchanged FaceID."""
import gc, json, time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from insightface.utils import face_align
from insightface.app import FaceAnalysis
from reference_inpaint import ROOT, CACHE, sha, reconstruct, load_rgb, StableDiffusionXLInpaintPipeline, DDIMScheduler

OUT=ROOT/'outputs/focused_eye_appearance_v1'
ASSETS=CACHE/'eye_appearance_models_v1'
IDS=['263','6681','10149']

class AppearanceProxy:
    def __init__(self,pipe): self.pipe=pipe; self.appearance_scale=0.; self.appearance=None; self.calls=[]
    def __getattr__(self,name): return getattr(self.pipe,name)
    def set_ip_adapter_scale(self,scale): self.pipe.set_ip_adapter_scale([scale,self.appearance_scale])
    def __call__(self,*args,**kwargs):
        assert kwargs['guidance_scale']==5. and len(kwargs['ip_adapter_image_embeds'])==1
        layers=self.pipe.unet.encoder_hid_proj.image_projection_layers
        assert len(layers)==2 and type(layers[1]).__name__=='IPAdapterPlusImageProjection'
        tokens=[layers[0].num_tokens,int(layers[1].latents.shape[1])]
        for processor in self.pipe.unet.attn_processors.values():
            if hasattr(processor,'num_tokens'):processor.num_tokens=tokens
        identity=kwargs['ip_adapter_image_embeds'][0]
        assert tuple(identity.shape)==(2,4,512) and tuple(self.appearance.shape)==(2,4,257,1280)
        kwargs['ip_adapter_image_embeds']=[identity,self.appearance]
        self.calls.append(dict(identity_shape=list(identity.shape),appearance_shape=list(self.appearance.shape),tokens_per_reference=tokens,
                               reference_count=4,scales=[.8,self.appearance_scale],projection_classes=[type(x).__name__ for x in layers]))
        (OUT/'wiring.json').write_text(json.dumps(self.calls,indent=2))
        return self.pipe(*args,**kwargs)

def main():
    cfg=json.loads((ROOT/'outputs/masked_sdxl_resolution_v1/config.json').read_text())
    cfg.update(expected=3,selected_identities=IDS,locations=['eyes'],guidance=5.,scale=.8,appearance_scale=.2,blend='poisson',training=False,
               single_change='Add Plus Face CLIP appearance branch at fixed scale0.2; keep Portrait branch and all sampling settings',
               source_sha256=sha(ROOT/'scripts/reference_inpaint.py'),runner_sha256=sha(__file__),
               protocol_sha256=sha(ROOT/'research/FOCUSED_EYE_APPEARANCE_PREFLIGHT_V1.md'),scorer_sha256=sha(ROOT/'scripts/score_masked_sdxl_v1.py'))
    receipt=json.loads((ASSETS/'receipt.json').read_text()); assert receipt['revision']=='018e402774aeeddd60609b4ecdb7e298259dc729'
    for r in receipt['files']:assert sha(r['path'])==r['sha256']
    cfg['appearance_assets']=receipt
    cases={c['identity']:c for c in json.loads((ROOT/'outputs/masked_task_v1/cases.json').read_text()) if c['location']=='eyes'}
    assert sha(ROOT/'outputs/masked_task_v1/cases.json')==cfg['cases_sha256']
    for ident in IDS:
        for k in ['input','mask']:assert sha(cases[ident][k])==cases[ident][k+'_sha256']
        for r in cfg['references'][ident+'/clean']:assert sha(r['path'])==r['sha256']
    OUT.mkdir(exist_ok=False)
    cfg['frozen_at_utc']=datetime.now(timezone.utc).isoformat()
    cfg['environment']={name:version(name) for name in ['torch','diffusers','transformers','insightface','onnxruntime','safetensors']}
    (OUT/'config.json').write_text(json.dumps(cfg,indent=2))
    torch.set_num_threads(4);cv2.setNumThreads(1)
    analyser=FaceAnalysis(name=str(CACHE/'reference_models/insightface/models/buffalo_l'),allowed_modules=['detection','recognition'],providers=['CPUExecutionProvider'])
    analyser.prepare(ctx_id=-1,det_size=(640,640))
    # Precompute in CPU float32, release the 2.5GB encoder before diffusion loading.
    encoder=CLIPVisionModelWithProjection.from_pretrained(str(ASSETS/'models/image_encoder'),local_files_only=True,torch_dtype=torch.float32).float().cpu().eval()
    assert all(p.dtype==torch.float32 for p in encoder.parameters()), 'CPU encoder weights must all use float32'
    for p in encoder.parameters():p.requires_grad_(False)
    processor=CLIPImageProcessor()
    cfg['image_processor']=processor.to_dict()
    banks={};feature_receipts={}
    for ident in IDS:
        crops=[];detections=[]
        for j,r in enumerate(cfg['references'][ident+'/clean']):
            rgb=np.asarray(load_rgb(r['path'])); bgr=rgb[:,:,::-1].copy();faces=analyser.get(bgr)
            assert len(faces)==1,'Exactly one face is required in every reference'
            crop=face_align.norm_crop(bgr,landmark=faces[0].kps,image_size=224)[:,:,::-1].copy()
            p=OUT/f'{ident}_reference_{j}_aligned.png';Image.fromarray(crop).save(p);crops.append(Image.fromarray(crop))
            detections.append(dict(reference_sha256=r['sha256'],crop_sha256=sha(p),landmarks=faces[0].kps.tolist(),score=float(faces[0].det_score)))
        pixels=processor(images=crops,return_tensors='pt').pixel_values.float()
        with torch.inference_mode():
            # Only hidden states are used; the pooled visual projection is unnecessary.
            positive=encoder.vision_model(pixels,output_hidden_states=True).hidden_states[-2]
            negative=encoder.vision_model(torch.zeros_like(pixels),output_hidden_states=True).hidden_states[-2]
        bank=torch.stack([negative,positive]).to(dtype=torch.float16).contiguous(); assert tuple(bank.shape)==(2,4,257,1280) and torch.isfinite(bank).all()
        path=OUT/f'{ident}_appearance.pt';torch.save(bank,path);banks[ident]=bank
        feature_receipts[ident]=dict(path=str(path),sha256=sha(path),shape=list(bank.shape),references=detections,
                                    encoder_layer='penultimate hidden state',unconditional='encoder applied to zero pixel tensor')
        print('ENCODED',ident,flush=True)
    del encoder,positive,negative,pixels;gc.collect()
    cfg['appearance_features']=feature_receipts
    pipe=StableDiffusionXLInpaintPipeline.from_pretrained(str(CACHE/'osor_models/sdxl'),torch_dtype=torch.float16,local_files_only=True,add_watermarker=False)
    pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter([str(CACHE/'reference_models'),str(ASSETS/'sdxl_models')],subfolder=['',''],
                         weight_name=['ip-adapter-faceid-portrait_sdxl.bin','ip-adapter-plus-face_sdxl_vit-h.safetensors'],
                         image_encoder_folder=None,local_files_only=True)
    pipe.enable_model_cpu_offload()
    proxy=AppearanceProxy(pipe);runtime={'pipeline':proxy,'memory':'model','analyser':analyser}
    cfg['scheduler']=dict(pipe.scheduler.config);(OUT/'config.json').write_text(json.dumps(cfg,indent=2))
    def run(ident,p):
        proxy.appearance=banks[ident]; c=cases[ident]
        return reconstruct(Path(c['input']),Path(c['mask']),[Path(r['path']) for r in cfg['references'][ident+'/clean']],p,
                           seed=17,steps=30,scale=.8,strength=1.,memory='model',blend='poisson',runtime=runtime,model_resolution=1024)
    p=OUT/'replay/263_eyes_clean_17.png';run('263',p);checks={}
    for suffix,folder in [('.png','focused_reference_finish_v1'),('_raw.png','masked_sdxl_resolution_v1')]:
        new=p.with_name(p.stem+suffix);old=ROOT/'outputs'/folder/new.name
        checks[suffix]=dict(pixel_exact=np.array_equal(np.asarray(Image.open(new)),np.asarray(Image.open(old))),new_sha256=sha(new),control_sha256=sha(old))
    (OUT/'replay.json').write_text(json.dumps(checks,indent=2)); assert all(v['pixel_exact'] for v in checks.values()),'Neutral replay mismatch; stop before candidate inference'
    proxy.appearance_scale=.2;rows=[]
    for ident in IDS:
        p=OUT/f'{ident}_eyes_clean_17.png';start=time.monotonic();row=dict(identity=ident,location='eyes',condition='clean',seed=17,file=p.name,appearance_scale=.2)
        try:
            meta=run(ident,p);row.update(status='complete',sha256=sha(p),raw_sha256=meta['raw_output_sha256'],seconds=time.monotonic()-start,peak_allocated_bytes=meta['peak_allocated_bytes'])
        except Exception as exc:row.update(status='failed',error=repr(exc),seconds=time.monotonic()-start)
        rows.append(row);(OUT/'rows.json').write_text(json.dumps(rows,indent=2));print('APPEARANCE',ident,row['status'],flush=True)
    (OUT/'wiring.json').write_text(json.dumps(proxy.calls,indent=2))

if __name__=='__main__':main()
