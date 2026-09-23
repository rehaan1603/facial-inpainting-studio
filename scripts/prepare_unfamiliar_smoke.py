"""Fresh development confirmation; reserved-final pixels are never read."""
import csv,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha,rank,write_new
from src.degradation.distortion_pipeline import degrade
from src.degradation.face_region_masks import damage_mask
from src.preservation.confidence import evidence_map
from distortion_aware_study import KINDS,RESERVED


def main():
    from src.reference_selection.reference_analyzer import ReferenceAnalyzer
    out=ROOT/'outputs/unfamiliar_smoke_v1';out.mkdir(exist_ok=False)
    old=json.loads((ROOT/'research/protocols/unseen_identity_protocol_v1.json').read_text())
    excluded=set().union(*(set(v['identity_labels']) for v in old['exclusion_ledger'].values()))|{c['identity'] for c in old['cases']}|{c['identity'] for c in old['attempts']}
    extension=json.loads((ROOT/'outputs/distortion_extension_v1/selection.json').read_text())
    excluded |= {str(a['identity']) for a in extension['attempts']} | {str(a['identity']) for a in extension['selected']}
    assert RESERVED<=excluded
    source=ROOT/'data/manifests/celebahq_reviewed_v2.csv';rows=list(csv.DictReader(source.open()))
    groups=defaultdict(list)
    for row in rows:
        if row['split']=='val' and row['identity'] not in excluded:groups[row['identity']].append(row)
    fingerprints=ROOT/'outputs/near_duplicate_audit/fingerprints.jsonl'
    fp={str(r['hq_id']):r for r in map(json.loads,fingerprints.read_text().splitlines()) if r['dataset']=='celebahq'}
    protected=[r for r in fp.values() if str(r['identity']) in excluded]
    protected_sha={r['source_sha256'] for r in protected};protected_rgb={r['decoded_rgb_sha256'] for r in protected};protected_phash=[int(r['phash'],16) for r in protected]
    detector=ReferenceAnalyzer().detector;selected=[];attempts=[];retained=[]
    for identity in sorted([i for i,g in groups.items() if len(g)>=8],key=lambda i:rank(i,'unfamiliar_smoke_v1')):
        assert identity not in RESERVED and identity not in excluded
        kept=[];attempt={'identity':identity,'photos':[]}
        for row in sorted(groups[identity],key=lambda r:rank(r['hq_id'],'unfamiliar_smoke_photo')):
            f=fp[row['hq_id']];ph=int(f['phash'],16);reasons=[]
            if f['source_sha256'] in protected_sha or f['decoded_rgb_sha256'] in protected_rgb or any((ph^p).bit_count()<=6 for p in protected_phash):reasons.append('protected_duplicate')
            if any(f['source_sha256']==r['source_sha256'] or f['decoded_rgb_sha256']==r['decoded_rgb_sha256'] or (ph^int(r['phash'],16)).bit_count()<=6 for r in retained+kept):reasons.append('role_duplicate')
            record=dict(row,decoded_rgb_sha256=f['decoded_rgb_sha256'],phash=f['phash'])
            if not reasons:
                assert sha(row['image_path'])==row['source_sha256']
                with Image.open(row['image_path']) as im:image=np.asarray(ImageOps.exif_transpose(im).convert('RGB').resize((512,512),Image.Resampling.LANCZOS))
                faces=detector.get(image[:,:,::-1].copy())
                if len(faces)!=1:reasons.append('non_unique_face')
                else:record['construction_geometry']={'bbox':faces[0].bbox.tolist(),'landmarks':faces[0].kps.tolist()}
            attempt['photos'].append({'hq_id':row['hq_id'],'reasons':reasons})
            if not reasons:kept.append(record)
            if len(kept)==8:break
        attempt['selected']=len(kept)==8;attempts.append(attempt)
        if len(kept)==8:selected.append({'identity':identity,'images':kept});retained.extend(kept)
        print('EXTENSION GROUP',identity,len(kept),'selected',len(selected),flush=True)
        if len(selected)==2:break
    if len(selected)!=2:
        write_new(out/'selection_failure.json',{'selected':selected,'attempts':attempts});raise ValueError('Insufficient eligible development groups')
    cfg={'date':'2026-09-22','identities':[g['identity'] for g in selected],'purpose':'unfamiliar local development functional smoke, not final or pretraining-disjoint validation','selection':'fixed hash order, exclude all earlier protocol and extension attempted/selected identities, exact and pHash<=6 duplicate filtering, one detected face, eight photos per identity','conditions':['gaussian_blur','mixed'],'seed':17,'steps':50,'guidance':1.5,'final_test_used':False}
    write_new(ROOT/'research/protocols/unfamiliar_smoke_v1.json',cfg)
    write_new(out/'selection.json',{'selected':selected,'attempts':attempts,'excluded_identities':sorted(excluded,key=int),'final_pixels_opened':False})
    inference=[];evaluation=[]
    for group in selected:
        identity=group['identity'];folder=out/'roles'/identity;folder.mkdir(parents=True);roles=[]
        for i,row in enumerate(group['images']):
            with Image.open(row['image_path']) as im:image=ImageOps.exif_transpose(im).convert('RGB').resize((512,512),Image.Resampling.LANCZOS)
            path=folder/f'role_{i}.png';image.save(path);roles.append({'path':str(path),'sha256':sha(path)})
        clean=np.asarray(Image.open(roles[0]['path']).convert('RGB'));geometry=group['images'][0]['construction_geometry']
        mask=damage_mask(clean.shape[:2],'central_face','medium',17,geometry['bbox'],geometry['landmarks'])
        for kind in cfg['conditions']:
            observed,_,metadata=degrade(clean,kind,'medium',17,mask=mask,region='central_face')
            cid=f'{identity}_{kind}';dest=out/'cases'/cid;dest.mkdir(parents=True)
            Image.fromarray(observed).save(dest/'observed.png');Image.fromarray(mask.astype('uint8')*255).save(dest/'mask.png')
            c={'case_id':cid,'identity':identity,'kind':kind,'references':roles[1:5],'distortion':metadata}
            for name in ['observed','mask']:c[name]=str(dest/(name+'.png'));c[name+'_sha256']=sha(c[name])
            inference.append(c);evaluation.append({'case_id':cid,'identity':identity,'target':roles[0],'gallery':roles[5:]})
    write_new(out/'inference_manifest.json',{'cases':inference});write_new(out/'evaluation_manifest.json',{'cases':evaluation})
    print('Four cases on two newly observed development identities prepared; no final pixels opened',flush=True)

if __name__=='__main__':main()
