"""Existing-dataset pilot only; all new-dataset/final identities remain excluded."""
import argparse
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter
from reference_intervention_core import ROOT,sha

OUT=ROOT/'outputs/target_compatibility_v1'
BASE=ROOT/'outputs/reference_intervention_pilot_v1'
CONDITIONS=['clean','blur','noise','jpeg','downsample','occlusion','lighting','wrong_patch']


def prepare():
    OUT.mkdir(exist_ok=False)
    manifest=json.loads((BASE/'manifest.json').read_text())
    protocol=dict(date='2026-10-02',task='Full-image blur/noise restoration, NOT erased-region inpainting',
        scope='Four already-observed development identities; feasibility screen, no final superiority claim',
        manifest_sha256=sha(BASE/'manifest.json'),cases=manifest['cases'],excluded=manifest['excluded_identities'],
        seed=20261002,steps=128,lr=.001,checkpoint='last fixed update only',
        training_arms=['no_target','full'],conditions=CONDITIONS,
        corruption='One reference corrupted; other three unchanged. Inference never receives corruption labels.',
        pose='Measured five-landmark geometric proxy; natural pose mismatch is an observational stratum, not simulated rotation.',
        primary='Corrupted-reference LPIPS delta vs matched frozen baseline and no-target equal-capacity training control',
        engineering_gate=dict(lpips_delta_max=-.005,facenet_delta_min=-.005,psnr_delta_min=-.1,minimum_identities_improved=3),
        statistical_gate='No significance/superiority on four identities; independent cohort only after engineering gate',
        losses='0.25 full latent epsilon MSE + 0.75 canonical face-ROI epsilon MSE. Both arms identical. Other requested losses remain separate feasibility work.',
        reference_selection='Four identical references for every paired arm; first reference receives each fixed corruption',
        regions='Fixed aligned-face rectangles, not anatomical segmentation; pose is a feature, not geometric warping',
        unimplemented=['image-space perceptual/identity gradient audit','true missing-area matched training','independent confirmation','final evaluation'],
        code={name:sha(ROOT/'scripts'/name) for name in ['target_region_compatibility_v1.py',Path(__file__).name]})
    rows=[]
    for index,case in enumerate(manifest['cases']):
        assert case['identity'] not in manifest['excluded_identities']
        folder=OUT/'images'/case['identity'];folder.mkdir(parents=True)
        refs=[]
        for i,source in enumerate(case['references']):
            assert sha(source['image_path'])==source['source_sha256']
            with Image.open(source['image_path']) as im:ref=im.convert('RGB').resize((512,512),Image.Resampling.BICUBIC)
            ref.save(folder/f'reference_{i}.png');refs.append(ref)
        rng=np.random.default_rng(20261002+index)
        first=refs[0];pixels=np.asarray(first).astype(float)
        buffer=io.BytesIO();first.save(buffer,format='JPEG',quality=15);buffer.seek(0)
        variants=dict(clean=first,blur=first.filter(ImageFilter.GaussianBlur(4)),
            noise=Image.fromarray(np.clip(pixels+rng.normal(0,25,pixels.shape),0,255).astype('uint8')),
            jpeg=Image.open(buffer).convert('RGB'),downsample=first.resize((32,32),Image.Resampling.BICUBIC).resize((512,512),Image.Resampling.BICUBIC),
            lighting=Image.fromarray(np.clip(pixels*np.array([1.25,.85,.65])+12,0,255).astype('uint8')),
            wrong_patch=Image.open(BASE/'images'/case['identity']/'corrupted_reference.png').convert('RGB'))
        occluded=first.copy();occluded.paste((127,127,127),(128,128,384,384));variants['occlusion']=occluded
        for condition,im in variants.items():im.save(folder/f'{condition}.png')
        rows.append(dict(identity=case['identity'],role=case['role'],files={p.name:sha(p) for p in folder.glob('*.png')},
                         lq=str(BASE/'images'/case['identity']/'input.png'),lq_sha256=sha(BASE/'images'/case['identity']/'input.png')))
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (OUT/'images.json').write_text(json.dumps(rows,indent=2))
    print('Prepared 20 existing identities; no reserved or new-dataset pixels',flush=True)


def diagnostics():
    import cv2
    from insightface.app import FaceAnalysis
    cv2.setNumThreads(1)
    cache=Path(json.loads((ROOT/'configs/local.json').read_text())['cache'])
    detector=FaceAnalysis(name=str(cache/'reference_models/insightface/models/buffalo_l'),
        allowed_modules=['detection'],providers=['CPUExecutionProvider'])
    detector.prepare(ctx_id=-1,det_size=(640,640),det_thresh=.5)
    def describe(path):
        rgb=np.asarray(Image.open(path).convert('RGB')).copy()
        faces=detector.get(rgb[:,:,::-1].copy())
        pose=None;confidence=0.
        if len(faces)==1:
            k=faces[0].kps;d=max(float(np.linalg.norm(k[1]-k[0])),1.)
            centre=(k[0]+k[1])/2;mouth=(k[3]+k[4])/2
            pose=[float((k[2,0]-centre[0])/d),float((k[2,1]-centre[1])/max(mouth[1]-centre[1],1)),float(np.arctan2(*(k[1]-k[0])[::-1]))]
            confidence=float(faces[0].det_score)
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(float)/255
        sharp=float(np.log1p(np.var(cv2.Laplacian(gray,cv2.CV_64F))*10000)/10)
        high=float(np.mean(np.abs(gray-cv2.GaussianBlur(gray,(3,3),0))))
        return dict(pose_proxy=pose,confidence=confidence,face_count=len(faces),sharpness=sharp,high_frequency=high,brightness=float(gray.mean()))
    records=[]
    for row in json.loads((OUT/'images.json').read_text()):
        folder=OUT/'images'/row['identity'];assert sha(row['lq'])==row['lq_sha256']
        target=describe(row['lq']);cache_features={}
        for name,expected in row['files'].items():
            assert sha(folder/name)==expected;cache_features[name]=describe(folder/name)
        conditions={}
        for condition in CONDITIONS:
            values=[]
            for name in [condition+'.png',*[f'reference_{i}.png' for i in range(1,4)]]:
                ref=cache_features[name]
                gap=float(np.linalg.norm(np.array(target['pose_proxy'])-ref['pose_proxy'])) if target['pose_proxy'] is not None and ref['pose_proxy'] is not None else 1.
                values.append([min(gap,2)/2,ref['sharpness'],ref['high_frequency'],abs(ref['brightness']-target['brightness']),min(ref['confidence'],target['confidence']),1.])
            conditions[condition]=values
        records.append(dict(identity=row['identity'],conditions=conditions,target=target,references=cache_features))
        print('DIAGNOSTICS',row['identity'],flush=True)
    (OUT/'diagnostics.json').write_text(json.dumps(records,indent=2))


def encode():
    import torch
    from reference_intervention_core import load_model
    model=load_model();model.first_stage_model.cuda();torch.set_num_threads(4)
    folder=OUT/'latents';folder.mkdir(exist_ok=False);receipt=[]
    old=json.loads((BASE/'cache_receipt.json').read_text())
    for row in old['records']:
        source=BASE/'latents'/(row['identity']+'.pt');assert sha(source)==row['sha256']
        original=torch.load(source,weights_only=True)
        variants={'clean':original['references'][:1], 'wrong_patch':original['corrupt_reference']}
        for condition in CONDITIONS:
            if condition in variants:continue
            p=OUT/'images'/row['identity']/(condition+'.png')
            rgb=np.asarray(Image.open(p).convert('RGB')).copy()
            x=torch.from_numpy(rgb).permute(2,0,1)[None].cuda().float()/127.5-1
            with torch.no_grad():variants[condition]=model.encode_first_stage(x).cpu()
        path=folder/(row['identity']+'.pt');torch.save(dict(variants=variants),path)
        receipt.append(dict(identity=row['identity'],role=row['role'],sha256=sha(path)))
        print('ENCODED',row['identity'],flush=True)
    (OUT/'cache.json').write_text(json.dumps(dict(records=receipt,protocol_sha256=sha(OUT/'protocol.json'),
        source_cache_sha256=sha(BASE/'cache_receipt.json'),script_sha256=sha(__file__)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','diagnostics','encode']);globals()[p.parse_args().stage]()
