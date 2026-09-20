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

BASE = ROOT/'outputs/distortion_extension_v1'
PROTOCOL = ROOT/'research/protocols/distortion_extension_v1.json'
KINDS = ['removal','gaussian_blur','noise','jpeg','downsample','mixed']
RESERVED = {'7789','8726','5774','6632','7170','6225','5170','8561'}


def rgb(path):
    with Image.open(path) as im:return np.asarray(im.convert('RGB')).copy()


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
        assert c['identity'] not in RESERVED and c['split']=='development_confirmation'
        for name in ['observed','mask','confidence']:assert sha(c[name])==c[name+'_sha256']
        for r in c['references']:assert sha(r['path'])==r['sha256']
        observed=rgb(c['observed']);mask=rgb(c['mask'])[:,:,0]>=128;confidence=np.load(c['confidence'])
        rows.append({'key':c['case_id']+'_observed','case_id':c['case_id'],'identity':c['identity'],'kind':c['kind'],'severity':c['severity'],'mode':'observed','status':'complete','output':c['observed'],'output_sha256':sha(c['observed'])})
        for seed in cfg['seeds']:
            for strength in cfg['strengths']:
                for scale in cfg['scales']:
                    key=f"{c['case_id']}_s{strength}_a{scale}_seed{seed}"
                    folder=BASE/'generations'/key;record_path=folder/'record.json';output=folder/'result.png'
                    if record_path.exists():
                        record=json.loads(record_path.read_text())
                        if record['status']=='complete':assert sha(record['output'])==record['output_sha256']
                    else:
                        record={'key':key,'case_id':c['case_id'],'identity':c['identity'],'kind':c['kind'],'severity':c['severity'],'mode':'standard','seed':seed,'strength':strength,'scale':scale}
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
                    print('GRID',len([r for r in rows if r['mode']=='standard']),'/96',key,record['status'],flush=True)
    write_new(BASE/'comparison.json',{'signature_sha256':sha(lock),'rows':rows,'final_test_used':False})


if __name__=='__main__':
    import time
    deadline=time.monotonic()+7200
    while not (ROOT/'outputs/local_latent_v1/comparison.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Previous GPU study incomplete')
        time.sleep(5)
    run()
