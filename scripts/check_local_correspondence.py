"""Extract only observed-target/reference local features; no clean target access."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.local_correspondence.features import LocalEncoder,extract,BOXES
from src.reference_selection.reference_analyzer import ReferenceAnalyzer,load_rgb
from src.research_integrity import ROOT,sha,write_new
from distortion_aware_study import BASE,RESERVED


def main():
    out=ROOT/'outputs/local_correspondence_v1';out.mkdir(exist_ok=False)
    encoder=LocalEncoder();detector=ReferenceAnalyzer().detector;records=[];cache={}
    signature={'encoder_sha256':encoder.weight_sha256,'frozen_parameters':encoder.parameter_count,'trainable_parameters':0,
               'sources':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/local_correspondence/features.py']},
               'inference_manifest_sha256':sha(BASE/'inference_manifest.json'),'clean_target_access':False}
    write_new(out/'signature.json',signature)
    for c in json.loads((BASE/'inference_manifest.json').read_text())['cases']:
        assert c['identity'] not in RESERVED
        record={'case_id':c['case_id'],'identity':c['identity'],'kind':c['kind'],'status':'failed'}
        try:
            observed=load_rgb(c['observed']);mask=load_rgb(c['mask'])[:,:,0]>=128
            assert sha(c['observed'])==c['observed_sha256'] and sha(c['mask'])==c['mask_sha256']
            faces=detector.get(observed[:,:,::-1].copy());record['observed_face_count']=len(faces)
            if len(faces)!=1:raise ValueError('No unique damaged-input face; no clean geometry fallback')
            target=extract(observed,faces[0],encoder,mask)
            references=[]
            for ref in c['references']:
                assert sha(ref['path'])==ref['sha256']
                if ref['sha256'] not in cache:
                    image=load_rgb(ref['path']);detected=detector.get(image[:,:,::-1].copy())
                    if len(detected)!=1:raise ValueError('Reference detection failed')
                    cache[ref['sha256']]=extract(image,detected[0],encoder)
                references.append(cache[ref['sha256']])
            folder=out/c['case_id'];folder.mkdir()
            np.savez_compressed(folder/'features.npz',observed=target['features'],references=np.stack([r['features'] for r in references]),
                                observed_global=target['global_identity'],reference_global=np.stack([r['global_identity'] for r in references]),matrix=target['matrix'])
            Image.fromarray(target['aligned']).save(folder/'observed_aligned.png')
            for i,r in enumerate(references):Image.fromarray(r['aligned']).save(folder/f'reference_{i+1}_aligned.png')
            detail=lambda r:{k:v for k,v in r.items() if k not in ['features','global_identity','aligned','matrix']}
            record.update(status='complete',features=str(folder/'features.npz'),features_sha256=sha(folder/'features.npz'),
                          observed=detail(target),references=[detail(r) for r in references],regions=list(BOXES))
        except Exception as error:record['error']=f'{type(error).__name__}: {error}'
        records.append(record);print('LOCAL FEATURES',len(records),'/24',record['status'],flush=True)
    write_new(out/'manifest.json',{'signature_sha256':sha(out/'signature.json'),'rows':records,'final_test_used':False})


if __name__=='__main__':main()
