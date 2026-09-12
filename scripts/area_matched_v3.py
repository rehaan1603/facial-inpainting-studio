"""Create translated masks with exact matched missing/FP/FN counts across locations."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt,gaussian_filter

ROOT=Path(__file__).resolve().parents[1]


def canonical(size,count,error_count,seed,aspect=1.5):
    yy,xx=np.mgrid[:size,:size];cx=cy=size//2
    score=((xx-cx)/aspect)**2+(yy-cy)**2
    # Stable ordering makes the exact-pixel shape deterministic even for tied distances.
    true=np.zeros((size,size),bool);true.flat[np.argsort(score,axis=None,kind='stable')[:count]]=True
    rng=np.random.default_rng(seed)
    noise=gaussian_filter(rng.normal(size=(size,size)),sigma=5)
    noise=noise/max(float(noise.std()),1e-8)*2
    inside=distance_transform_edt(true)+noise
    outside=distance_transform_edt(~true)+noise
    def select(values,region,n):
        ix=np.flatnonzero(region);chosen=ix[np.argsort(values.flat[ix],kind='stable')[:n]]
        result=np.zeros_like(true);result.flat[chosen]=True;return result
    removed=select(inside,true,error_count);added=select(outside,~true,error_count)
    return true,{'accurate':true.copy(),'under':true&~removed,'over':true|added,'mixed':(true&~removed)|added}


def translate(mask,dx,dy):
    yy,xx=np.where(mask);yy=yy+dy;xx=xx+dx
    if (yy<0).any() or (yy>=mask.shape[0]).any() or (xx<0).any() or (xx>=mask.shape[1]).any():return None
    result=np.zeros_like(mask);result[yy,xx]=True;return result


def corrupt(target,true,seed,center=None):
    rng=np.random.default_rng(seed);texture=rng.integers(0,256,(12,12,3),dtype=np.uint8)
    occ=np.array(Image.fromarray(texture).resize((target.shape[1],target.shape[0]),Image.Resampling.BILINEAR)).astype('float32')/255
    if center is not None:
        occ=np.roll(occ,(center[1]-target.shape[0]//2,center[0]-target.shape[1]//2),axis=(0,1))
    return np.where(true[...,None],occ,target).astype('float32')


def semantic_center(hq,idx,parts,size):
    union=np.zeros((size,size),bool)
    for part in parts:
        path=hq/'CelebAMask-HQ-mask-anno'/str(idx//2000)/f'{idx:05d}_{part}.png'
        if path.exists():
            with Image.open(path) as im:union|=np.array(im.convert('L').resize((size,size),Image.Resampling.NEAREST))>0
    if not union.any():return None
    yy,xx=np.where(union);return (int(np.round(xx.mean())),int(np.round(yy.mean())))


def main():
    config_path=ROOT/'configs/area_matched_v3.json';cfg=json.loads(config_path.read_text())
    local=json.loads((ROOT/'configs/local.json').read_text());hq=Path(local['hq']);size=cfg['resolution']
    original=json.loads((ROOT/'outputs/benchmark_v2/cases.json').read_text())
    identities=list({c['hq_id']:c for c in original}.values())
    sources={int(r['hq_id']):r for r in csv.DictReader((ROOT/'data/manifests/celebahq_clean.csv').open())}
    out=ROOT/'outputs/area_matched_v3_margin12';out.mkdir(parents=True,exist_ok=True)
    records=[];rejected=[];preview=[]
    for c in identities:
        idx=c['hq_id'];row=sources[idx];assert row['split']=='val'
        assert hashlib.sha256(Path(row['image_path']).read_bytes()).hexdigest()==c['source_sha256']
        centers={'eye_center':semantic_center(hq,idx,['l_eye','r_eye'],size),'mouth_center':semantic_center(hq,idx,['mouth','u_lip','l_lip'],size),'upper_image':tuple(cfg['upper_image_center_xy'])}
        for count in cfg['missing_pixels']:
            error_count=round(count*cfg['error_fraction_of_missing']);seed=cfg['seed']+idx*10000+count
            true,masks=canonical(size,count,error_count,seed,cfg['ellipse_aspect_ratio'])
            translated={};failure=None
            for location,center in centers.items():
                if center is None:failure=f'missing center: {location}';break
                dx,dy=center[0]-size//2,center[1]-size//2
                items={'true':translate(true,dx,dy),**{k:translate(v,dx,dy) for k,v in masks.items()}}
                if any(v is None for v in items.values()):failure=f'out of bounds: {location}';break
                margin=cfg['required_margin_pixels']
                for mask in items.values():
                    yy,xx=np.where(mask)
                    if min(yy.min(),xx.min(),size-1-yy.max(),size-1-xx.max())<margin:
                        failure=f'insufficient correction margin: {location}';break
                if failure:break
                translated[location]=items
            if failure:
                rejected.append({'hq_id':idx,'identity':c['identity'],'partition':c['partition'],'missing_pixels':count,'reason':failure});continue
            for location,items in translated.items():
                folder=out/'masks'/f'{idx}_{count}_{location}';folder.mkdir(parents=True,exist_ok=True)
                hashes={}
                for name,mask in items.items():
                    path=folder/f'{name}.png';Image.fromarray(mask.astype('uint8')*255).save(path)
                    hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
                entries={}
                for condition in cfg['conditions']:
                    supplied=items[condition];fp=int((supplied&~items['true']).sum());fn=int((~supplied&items['true']).sum())
                    assert int(items['true'].sum())==count
                    assert fp==(error_count if condition in ['over','mixed'] else 0)
                    assert fn==(error_count if condition in ['under','mixed'] else 0)
                    entries[condition]={'fp_pixels':fp,'fn_pixels':fn}
                records.append({'hq_id':idx,'identity':c['identity'],'partition':c['partition'],'location':location,'center_xy':centers[location],'missing_pixels':count,'error_pixels':error_count,'seed':seed,'source_sha256':c['source_sha256'],'mask_directory':str(folder.relative_to(ROOT)).replace('\\','/'),'mask_sha256':hashes,'conditions':entries})
            if len(preview)<3 and c['partition']=='assessment' and count==3932:
                with Image.open(row['image_path']) as im:target=np.array(im.convert('RGB').resize((size,size),Image.Resampling.LANCZOS)).astype('float32')/255
                ims=[target]+[corrupt(target,translated[loc]['true'],seed,centers[loc]) for loc in cfg['locations']]
                preview.append(ims)
    (out/'cases.json').write_text(json.dumps(records,indent=2))
    retained={p:len({r['identity'] for r in records if r['partition']==p}) for p in ['tuning','assessment']}
    report={'config':cfg,'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),'source_cases_sha256':hashlib.sha256((ROOT/'outputs/benchmark_v2/cases.json').read_bytes()).hexdigest(),'generated_cases':len(records),'identity_area_groups':len(records)//3,'retained_identities':retained,'rejected_groups':rejected,'by_area':{str(n):sum(r['missing_pixels']==n for r in records) for n in cfg['missing_pixels']},'texture_policy':'Translate the canonical texture with the mask; occluder RGB values match across locations within each identity/area group.','limits':['Dataset construction report; model evaluation is recorded separately.','Synthetic textures only.','Center placement is not proof of complete semantic coverage.','Group rejection can bias retained cases; preserve rejection ledger.','Same development identities; not a fresh held-out evaluation.']}
    (ROOT/'research/area_matched_v3.json').write_text(json.dumps(report,indent=2))
    canvas=Image.new('RGB',(size*4,280*len(preview)),'white');draw=ImageDraw.Draw(canvas)
    for i,ims in enumerate(preview):
        for j,im in enumerate(ims):
            canvas.paste(Image.fromarray((im*255).round().astype('uint8')),(size*j,280*i+24))
            draw.text((size*j+3,280*i+3),['Target','Eye center','Mouth center','Upper-image control'][j],fill='black')
    canvas.save(out/'preview.png')
    print(json.dumps({k:v for k,v in report.items() if k not in ['config','rejected_groups']},indent=2));print('Rejected groups:',len(rejected))


if __name__=='__main__':main()
