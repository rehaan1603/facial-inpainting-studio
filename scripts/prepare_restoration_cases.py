"""Materialize only the frozen validation roles; never open final-test pixels."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageOps
from src.research_integrity import ROOT,sha,write_new
from src.degradation import degrade
from src.degradation.metadata import save_case

def main():
    protocol_path=ROOT/'research/protocols/unseen_identity_protocol_v1.json';protocol=json.loads(protocol_path.read_text())
    out=ROOT/'outputs/generalization_v1/cases';out.mkdir(exist_ok=False)
    cases=[]
    for identity in protocol['cases']:
        if identity['split']!='validation':continue
        folder=out/identity['identity'];folder.mkdir();roles={}
        for r in identity['images']:
            if sha(r['image_path'])!=r['source_sha256']:raise ValueError('Source changed')
            with Image.open(r['image_path']) as im:image=ImageOps.exif_transpose(im).convert('RGB').resize((512,512),Image.Resampling.LANCZOS)
            dest=folder/(r['role']+'.png');image.save(dest);roles[r['role']]={'path':str(dest),'sha256':sha(dest)}
        target=next(r for r in identity['images'] if r['role']=='target');geometry=target['construction_detection']
        clean=np.asarray(Image.open(roles['target']['path']).convert('RGB'))
        for condition in protocol['experiment']['conditions']:
            region,kind,severity=condition.rsplit('_',2)
            observed,mask,metadata=degrade(clean,kind,severity,17,region=region,bbox=geometry['bbox'],landmarks=geometry['landmarks'])
            dest=folder/condition;meta=save_case(dest,observed,mask,metadata)
            cases.append({'case_id':identity['identity']+'_'+condition,'identity':identity['identity'],'split':'validation','condition':condition,
                          'observed':str(dest/'observed.png'),'mask':str(dest/'mask.png'),'distortion':meta,
                          'references':[roles[f'reference_{i}'] for i in range(1,5)],'evaluation_only_target':roles['target'],
                          'evaluation_only_gallery':[roles[f'gallery_{i}'] for i in range(1,4)]})
    write_new(out/'manifest.json',{'protocol_sha256':sha(protocol_path),'script_sha256':sha(__file__),'cases':cases})
    print('Prepared',len(cases),'cases; final identities remain unused')

if __name__=='__main__':main()
