"""Versioned development-only denoising grid and explicit evidence preservation."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, write_new
from src.preservation.confidence import evidence_map, preserve_observation

BASE = ROOT/'outputs/distortion_aware_v1'
PROTOCOL = ROOT/'research/protocols/distortion_aware_v1.json'
KINDS = ['removal','gaussian_blur','noise','jpeg','downsample','mixed']
RESERVED = {'7789','8726','5774','6632','7170','6225','5170','8561'}


def rgb(path):
    with Image.open(path) as im:return np.asarray(im.convert('RGB')).copy()


def prepare():
    from src.degradation.distortion_pipeline import degrade
    BASE.mkdir(exist_ok=False)
    prior_path=ROOT/'outputs/generalization_v1/cases/manifest.json'
    prior=json.loads(prior_path.read_text())['cases']
    cfg={'version':1,'date':'2026-09-20','stage':'Small development screening; seed-29 confirmation required before promotion',
         'identities':['1306','2790','1043','787'],'previously_observed_identities':True,
         'kinds':KINDS,'strengths':[.5,.75,.99],'scales':[.8,1.2],'seeds':[17],
         'steps':30,'guidance_scale':5.,'blend':'poisson','resolution':512,'severity':'medium',
         'mask':'Same central-face mask within each identity for every distortion',
         'preservation_confidence':.5,'statistical_unit':'identity',
         'primary_contrasts':['strength .50 vs .99 at scale .8, by damage kind','strength .75 vs .99 at scale .8, by damage kind','preserve vs standard at .99/.8, by degraded kind'],
         'multiplicity':'Holm across 17 primary FaceNet contrasts (12 strengths and 5 preservation); other metrics descriptive safeguards',
         'promotion_gate':'No promotion from this n=4 single-seed screen. Require independent development confirmation, identity and LPIPS/hole-error agreement, exact visible preservation, failure review and stable seed behavior.',
         'final_test_used':False,'generator_sha256':sha(ROOT/'scripts/reference_inpaint.py'),
         'prior_manifest_sha256':sha(prior_path),'known_limitations':['Supplied distortion class and confidence, not blind restoration','Fixed confidence is a heuristic','Unknown pretraining exposure','Same step budget but lower strength executes fewer denoising steps','No matched-active-step causal ablation yet']}
    write_new(PROTOCOL,cfg)
    inference=[];evaluation=[]
    for identity in cfg['identities']:
        assert identity not in RESERVED
        c=next(c for c in prior if c['identity']==identity and c['condition']=='central_face_mixed_medium')
        assert c['split']=='validation'
        target=rgb(c['evaluation_only_target']['path']);mask=rgb(c['mask'])[:,:,0]>=128
        assert sha(c['evaluation_only_target']['path'])==c['evaluation_only_target']['sha256']
        for kind in KINDS:
            observed,_,metadata=degrade(target,kind,'medium',17,mask=mask,region='central_face')
            folder=BASE/'cases'/identity/kind;folder.mkdir(parents=True)
            Image.fromarray(observed).save(folder/'observed.png');Image.fromarray(mask.astype('uint8')*255).save(folder/'mask.png')
            confidence=evidence_map(mask,kind=='removal',.5)
            np.save(folder/'confidence.npy',confidence)
            case={'case_id':identity+'_'+kind,'identity':identity,'split':'development_screen','kind':kind,
                  'observed':str(folder/'observed.png'),'mask':str(folder/'mask.png'),
                  'confidence':str(folder/'confidence.npy'),'confidence_sha256':sha(folder/'confidence.npy'),
                  'observed_sha256':sha(folder/'observed.png'),'mask_sha256':sha(folder/'mask.png'),'references':c['references'],
                  'distortion':metadata}
            inference.append(case);evaluation.append({'case_id':case['case_id'],'identity':identity,'target':c['evaluation_only_target'],'gallery':c['evaluation_only_gallery']})
    write_new(BASE/'inference_manifest.json',{'protocol_sha256':sha(PROTOCOL),'cases':inference})
    write_new(BASE/'evaluation_manifest.json',{'cases':evaluation,'clean_data_scope':'Scoring only, not imported by generation'})
    print('Prepared 24 cases; 144 requested GPU settings; final identities unused',flush=True)


def run():
    from reference_inpaint import reconstruct
    cfg=json.loads(PROTOCOL.read_text());manifest=json.loads((BASE/'inference_manifest.json').read_text())
    assert manifest['protocol_sha256']==sha(PROTOCOL)
    sources=[Path(__file__),ROOT/'scripts/reference_inpaint.py',ROOT/'scripts/reference_blending.py',ROOT/'src/preservation/confidence.py']
    signature={'protocol_sha256':sha(PROTOCOL),'manifest_sha256':sha(BASE/'inference_manifest.json'),'sources':{str(p.relative_to(ROOT)):sha(p) for p in sources}}
    lock=BASE/'generation_signature.json'
    if lock.exists():assert json.loads(lock.read_text())==signature,'Frozen source changed'
    else:write_new(lock,signature)
    runtime={};rows=[]
    for c in manifest['cases']:
        assert c['identity'] not in RESERVED and c['split']=='development_screen'
        for name in ['observed','mask','confidence']:assert sha(c[name])==c[name+'_sha256']
        for r in c['references']:assert sha(r['path'])==r['sha256']
        observed=rgb(c['observed']);mask=rgb(c['mask'])[:,:,0]>=128;confidence=np.load(c['confidence'])
        rows.append({'key':c['case_id']+'_observed','case_id':c['case_id'],'identity':c['identity'],'kind':c['kind'],'mode':'observed','status':'complete','output':c['observed'],'output_sha256':sha(c['observed'])})
        for seed in cfg['seeds']:
            for strength in cfg['strengths']:
                for scale in cfg['scales']:
                    key=f"{c['case_id']}_s{strength}_a{scale}_seed{seed}"
                    folder=BASE/'generations'/key;record_path=folder/'record.json';output=folder/'result.png'
                    if record_path.exists():
                        record=json.loads(record_path.read_text())
                        if record['status']=='complete':assert sha(record['output'])==record['output_sha256']
                    else:
                        record={'key':key,'case_id':c['case_id'],'identity':c['identity'],'kind':c['kind'],'mode':'standard','seed':seed,'strength':strength,'scale':scale}
                        try:
                            if folder.exists():raise ValueError('Interrupted partial output: preserve and audit before retry')
                            reconstruct(c['observed'],c['mask'],[r['path'] for r in c['references']],output,seed=seed,steps=cfg['steps'],scale=scale,strength=strength,blend=cfg['blend'],runtime=runtime)
                            record.update(status='complete',output=str(output),output_sha256=sha(output),metadata_sha256=sha(output.with_suffix('.json')))
                        except Exception as error:record.update(status='failed',error=f'{type(error).__name__}: {error}')
                        write_new(record_path,record)
                    rows.append(record)
                    if record['status']=='complete':
                        preserved=folder/'preserved.png'
                        if not preserved.exists():Image.fromarray(preserve_observation(observed,rgb(output),mask,confidence)).save(preserved)
                        rows.append(dict(record,key=key+'_preserve',mode='preserve',output=str(preserved),output_sha256=sha(preserved),parent_output_sha256=record['output_sha256'],confidence_sha256=c['confidence_sha256']))
                    else:rows.append(dict(record,key=key+'_preserve',mode='preserve'))
                    print('GRID',len([r for r in rows if r['mode']=='standard']),'/144',key,record['status'],flush=True)
    write_new(BASE/'comparison.json',{'signature_sha256':sha(lock),'rows':rows,'final_test_used':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);args=p.parse_args()
    prepare() if args.action=='prepare' else run()
