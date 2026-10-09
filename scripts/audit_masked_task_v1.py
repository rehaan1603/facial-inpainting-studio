"""Fixed masked-task feasibility experiment; no training or production changes."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from reference_intervention_core import ROOT, sha

OUT = ROOT/'outputs/masked_task_v1'
BASE = ROOT/'outputs/reference_intervention_pilot_v1'
OLD = ROOT/'outputs/target_compatibility_v1'
CHECKPOINT = ROOT/'outputs/identity_timestep_v5/timestep_scaled/adapter.pt'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def read(path):
    return np.asarray(Image.open(path).convert('RGB')).copy()


def neutralize(rgb, mask):
    assert mask.dtype == bool and mask.any() and not mask.all()
    result = rgb.copy()
    result[mask] = 127
    return result


def prepare():
    from benchmark_v2 import semantic_mask
    import cv2
    from insightface.app import FaceAnalysis
    local = json.loads((ROOT/'configs/local.json').read_text())
    detector = FaceAnalysis(name=str(Path(local['cache'])/'reference_models/insightface/models/buffalo_l'),
                           allowed_modules=['detection'], providers=['CPUExecutionProvider'])
    detector.prepare(ctx_id=-1, det_size=(640,640), det_thresh=.5)
    def describe(rgb):
        faces = detector.get(rgb[:,:,::-1].copy()); pose = None; confidence = 0.
        if len(faces) == 1:
            k = faces[0].kps; d = max(float(np.linalg.norm(k[1]-k[0])),1.)
            centre = (k[0]+k[1])/2; mouth = (k[3]+k[4])/2
            pose = np.array([(k[2,0]-centre[0])/d,(k[2,1]-centre[1])/max(mouth[1]-centre[1],1),np.arctan2(*(k[1]-k[0])[::-1])])
            confidence = float(faces[0].det_score)
        gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(float)/255
        return dict(pose=pose,confidence=confidence,sharp=float(np.log1p(np.var(cv2.Laplacian(gray,cv2.CV_64F))*10000)/10),
                    high=float(np.mean(np.abs(gray-cv2.GaussianBlur(gray,(3,3),0)))),brightness=float(gray.mean()))
    manifest = json.loads((BASE/'manifest.json').read_text())
    cases = [c for c in manifest['cases'] if c['role']=='adapter_holdout']
    assert len(cases)==4 and CHECKPOINT.exists()
    tr = json.loads((CHECKPOINT.parent/'training.json').read_text())
    assert sha(CHECKPOINT)==tr['arms'][0]['checkpoint_sha256']
    OUT.mkdir(exist_ok=False)
    config = dict(task='Genuinely erased facial regions; new diagnostic, not comparable to restoration scores',
        identities=[c['identity'] for c in cases],seeds=[17,29,43],conditions=['clean','wrong_patch'],
        arms=['baseline','timestep_scaled'],locations=['eyes','mouth'],steps=50,guidance=1.5,
        precision='bfloat16',resolution=512,reference_encoding='separate existing verified VQ latents',
        mask_rule='Existing benchmark_v2.semantic_mask at512, eyes pad8/mouth pad12; same rule applied to current adapter holdout targets. These are newly instantiated masks, not old evaluated case files.',
        diagnostics='Recomputed exclusively from erased input and supplied references; legacy coverage slot remains1 to avoid simultaneous feature-definition change.',
        checkpoint=str(CHECKPOINT),checkpoint_sha256=sha(CHECKPOINT),manifest_sha256=sha(BASE/'manifest.json'),
        mask_code_sha256=sha(ROOT/'scripts/benchmark_v2.py'),script_sha256=sha(__file__),expected_outputs=96,
        training=False,final_test_used=False,limitations=['Four reused development identities','Annotation-derived supplied masks','No upstream pretraining exclusion claim'])
    write(OUT/'config.json',config)
    rows=[]
    for c in cases:
        ident=c['identity']; target=read(BASE/'images'/ident/'target.png')
        assert ident not in manifest['excluded_identities']
        assert sha(c['target']['image_path'])==c['target']['source_sha256']
        for location in config['locations']:
            mask=semantic_mask(Path(local['hq']),int(c['target']['hq_id']),location,512)
            assert mask is not None and mask.any() and not mask.all()
            folder=OUT/'inputs'/f'{ident}_{location}';folder.mkdir(parents=True)
            observed=neutralize(target,mask)
            altered=target.copy();altered[mask]=np.random.default_rng(20261008).integers(0,256,(int(mask.sum()),3),dtype=np.uint8)
            assert np.array_equal(observed,neutralize(altered,mask))
            Image.fromarray(observed).save(folder/'input.png');Image.fromarray(mask.astype('uint8')*255).save(folder/'mask.png')
            td=describe(observed);diags={}
            for condition in config['conditions']:
                vals=[]
                for name in [condition+'.png','reference_1.png','reference_2.png','reference_3.png']:
                    rd=describe(read(OLD/'images'/ident/name))
                    gap=float(np.linalg.norm(td['pose']-rd['pose'])) if td['pose'] is not None and rd['pose'] is not None else 1.
                    vals.append([min(gap,2)/2,rd['sharp'],rd['high'],abs(rd['brightness']-td['brightness']),min(rd['confidence'],td['confidence']),1.])
                diags[condition]=vals
            rows.append(dict(identity=ident,location=location,hq_id=c['target']['hq_id'],input=str(folder/'input.png'),mask=str(folder/'mask.png'),
                input_sha256=sha(folder/'input.png'),mask_sha256=sha(folder/'mask.png'),target=str(BASE/'images'/ident/'target.png'),
                target_sha256=sha(BASE/'images'/ident/'target.png'),hole_pixels=int(mask.sum()),hidden_pixel_invariance=True,diagnostics=diags))
    write(OUT/'cases.json',rows)
    write(OUT/'preparation_receipt.json',dict(config_sha256=sha(OUT/'config.json'),cases_sha256=sha(OUT/'cases.json')))
    print('PREPARED',len(rows),'cases; input hidden-pixel invariance passed',flush=True)


def generate(limit):
    from reference_intervention_core import load_model
    from target_region_compatibility_v1 import Compatibility,features,install
    from train_target_compatibility_v1 import load_bundles
    cfg=json.loads((OUT/'config.json').read_text());cases=json.loads((OUT/'cases.json').read_text())
    prepared=json.loads((OUT/'preparation_receipt.json').read_text())
    assert sha(OUT/'config.json')==prepared['config_sha256'] and sha(OUT/'cases.json')==prepared['cases_sha256']
    assert sha(CHECKPOINT)==cfg['checkpoint_sha256']
    dest=OUT/'evaluation';dest.mkdir(exist_ok=True)
    provenance=dest/'provenance.json'
    current=dict(script_sha256=sha(__file__),config_sha256=sha(OUT/'config.json'),cases_sha256=sha(OUT/'cases.json'))
    if provenance.exists():assert json.loads(provenance.read_text())==current
    else:write(provenance,current)
    rows=json.loads((dest/'rows.json').read_text()) if (dest/'rows.json').exists() else []
    for r in rows:
        assert sha(dest/r['file'])==r['sha256'] and sha(dest/r['raw'])==r['raw_sha256']
    done={(r['identity'],r['location'],r['arm'],r['condition'],r['seed']) for r in rows}
    torch.set_num_threads(4);model=load_model().cuda();adapter=Compatibility().cuda();install(model,adapter)
    adapter.load_state_dict(torch.load(CHECKPOINT,weights_only=True));adapter.mode='full'
    bundles={r['identity']:r for r in load_bundles()}
    from ldm.models.diffusion.ddim import DDIMSampler
    sampler=DDIMSampler(model,print_tqdm=False,schedule='uniform_trailing')
    def sample(lq,refs,seed):
        cond=dict(lq_image=lq,ref_image=torch.cat(list(refs.split(1)),-1))
        null=dict(lq_image=lq,ref_image=torch.zeros_like(cond['ref_image']))
        torch.manual_seed(seed);noise=torch.randn(1,8,64,64,device='cuda')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            latent,_=sampler.sample(S=50,unconditional_guidance_scale=1.5,conditioning=cond,unconditional_conditioning=null,
                                    shape=[8,64,64],x_T=noise,batch_size=1,verbose=False)
            rgb=((model.decode_first_stage(latent)+1)/2).clamp(0,1)
        assert torch.isfinite(rgb).all()
        return (rgb[0].permute(1,2,0).float().cpu().numpy()*255).round().astype('uint8')
    # Fresh restoration replay verifies the unchanged sampler before any masked case.
    receipt=dest/'replay.json'
    if not receipt.exists():
        b=bundles[cases[0]['identity']];adapter.enabled=False
        a=sample(b['old']['lq'].cuda(),b['old']['references'].cuda(),17)
        path=dest/'restoration_replay.png';Image.fromarray(a).save(path)
        legacy=json.loads((OLD/'evaluation/rows.json').read_text())
        expected=next(r for r in legacy if r['identity']==cases[0]['identity'] and r['arm']=='baseline' and r['condition']=='clean')
        ok=sha(path)==expected['sha256'];write(receipt,dict(byte_exact=ok,expected=expected['sha256'],actual=sha(path)))
        assert ok,'Unchanged baseline failed fresh replay'
    else: assert json.loads(receipt.read_text())['byte_exact']
    # Seed/condition outer loop completes an across-identity clean smoke screen first.
    count=0
    for seed in cfg['seeds']:
      for condition in cfg['conditions']:
       for c in cases:
        assert sha(c['input'])==c['input_sha256'] and sha(c['mask'])==c['mask_sha256']
        observed=read(c['input']);mask=np.asarray(Image.open(c['mask']))>127
        x=torch.from_numpy(observed.copy()).permute(2,0,1)[None].cuda().float()/127.5-1
        with torch.inference_mode():lq=model.encode_first_stage(x)
        b=bundles[c['identity']];refs=b['old']['references'].clone().cuda();refs[:1]=b['new']['variants'][condition].cuda()
        adapter.context=features(lq,refs,torch.tensor(c['diagnostics'][condition],device='cuda'))
        for arm in cfg['arms']:
            key=(c['identity'],c['location'],arm,condition,seed)
            if key in done:continue
            if limit and count>=limit:return
            adapter.enabled=arm!='baseline';start=time.monotonic();torch.cuda.reset_peak_memory_stats()
            raw=sample(lq,refs,seed);result=np.where(mask[...,None],raw,observed)
            assert np.array_equal(result[~mask],observed[~mask])
            name=f'{c["identity"]}_{c["location"]}_{arm}_{condition}_{seed}'
            p=dest/(name+'.png');q=dest/(name+'_raw.png');assert not p.exists() and not q.exists()
            Image.fromarray(result).save(p);Image.fromarray(raw).save(q)
            row=dict(identity=c['identity'],location=c['location'],arm=arm,condition=condition,seed=seed,file=p.name,raw=q.name,
                     sha256=sha(p),raw_sha256=sha(q),seconds=time.monotonic()-start,peak_mib=torch.cuda.max_memory_allocated()/2**20)
            rows.append(row);write(dest/'rows.json',rows);count+=1
            print('GENERATED',len(rows),name,round(row['seconds'],2),flush=True)
    assert len(rows)==cfg['expected_outputs']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','generate']);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    prepare() if a.stage=='prepare' else generate(a.limit)
