"""Matched global/local reference conditioning; frozen historical outputs preserved."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha,write_new
from src.local_correspondence.latent_fusion import LocalLatentPipeline
from distortion_aware_study import BASE,RESERVED

OUT=ROOT/'outputs/local_latent_v1'


def main():
    from src.reference_fusion.gpu_masks import make_gpu_runtime
    from src.reference_fusion.feature_bank import analyze_runtime
    from regional_reference_inpaint import reconstruct_routed
    from reference_inpaint import reconstruct
    cfgpath=ROOT/'research/protocols/local_latent_v1.json';cfg=json.loads(cfgpath.read_text())
    manifest=json.loads((OUT/'feature_manifest.json').read_text())
    for path,digest in manifest['signature']['sources'].items():assert sha(ROOT/path)==digest
    assert manifest['signature']['protocol_sha256']==sha(cfgpath)
    sources=[Path(__file__),ROOT/'src/local_correspondence/latent_fusion.py',ROOT/'scripts/reference_inpaint.py',ROOT/'scripts/regional_reference_inpaint.py',ROOT/'src/reference_fusion/gpu_masks.py']
    signature={'sources':{str(p.relative_to(ROOT)):sha(p) for p in sources},'feature_manifest_sha256':sha(OUT/'feature_manifest.json'),'protocol_sha256':sha(cfgpath),
               'primary_comparisons':'damage minus each of the other five policies; paired condition/seed means within identity; Holm over five FaceNet tests'}
    lock=OUT/'generation_signature.json'
    if lock.exists():assert json.loads(lock.read_text())==signature
    else:write_new(lock,signature)
    cases={c['case_id']:c for c in json.loads((BASE/'inference_manifest.json').read_text())['cases']}
    old=json.loads((ROOT/'outputs/regional_routing_v2/comparison.json').read_text())['rows']
    runtime=make_gpu_runtime();base=runtime['base_pipeline'];analyzer=None;rows=[]
    for f in manifest['rows']:
        c=cases[f['case_id']];assert c['identity'] not in RESERVED
        assert f['status']=='complete' and sha(f['bank'])==f['bank_sha256']
        features=np.load(f['bank']);refs=[r['path'] for r in c['references']]
        assert sha(c['observed'])==c['observed_sha256'] and sha(c['mask'])==c['mask_sha256']
        for r in c['references']:assert sha(r['path'])==r['sha256']
        bank=analyze_runtime(c['observed'],c['mask'],refs,analyzer);analyzer=bank[2]
        for seed in cfg['seeds']:
            for policy in cfg['policies']:
                key=f"{c['case_id']}_seed{seed}_{policy}";folder=OUT/'generations'/key;dest=folder/'record.json';output=folder/'result.png'
                if dest.exists():
                    record=json.loads(dest.read_text())
                    if record['status']=='complete':assert sha(record['output'])==record['output_sha256']
                else:
                    record={'key':key,'case_id':c['case_id'],'identity':c['identity'],'kind':c['kind'],'seed':seed,'policy':policy,'status':'failed'}
                    try:
                        if folder.exists():raise ValueError('Partial output exists; audit before retry')
                        reused=False
                        if policy in ['global','regional_global']:
                            prior=next(r for r in old if r['identity']==c['identity'] and r['condition']=='central_face_mixed_medium' and r['seed']==seed and r['policy']==('concat' if policy=='global' else 'regional'))
                            metadata=json.loads(Path(prior['output']).with_suffix('.json').read_text())
                            if metadata['input_sha256']==c['observed_sha256'] and metadata['mask_sha256']==c['mask_sha256'] and [r['sha256'] for r in metadata['references']]==[r['sha256'] for r in c['references']] and metadata['seed']==seed and metadata['strength']==cfg['strength'] and metadata['adapter_scale']==cfg['scale'] and metadata['steps']==cfg['steps']:
                                assert sha(prior['output'])==prior['output_sha256']
                                record.update(status='complete',output=prior['output'],output_sha256=prior['output_sha256'],reused_from='regional_routing_v2');reused=True
                        if not reused:
                            if policy=='regional_global':
                                metadata=reconstruct_routed(c['observed'],c['mask'],refs,output,'regional',seed,cfg['steps'],cfg['scale'],cfg['strength'],cfg['blend'],runtime=runtime,bank=bank)
                            else:
                                proxy=base if policy=='global' else LocalLatentPipeline(base,features[policy],features['gate'])
                                metadata=reconstruct(c['observed'],c['mask'],refs,output,seed=seed,steps=cfg['steps'],scale=cfg['scale'],strength=cfg['strength'],blend=cfg['blend'],runtime={'pipeline':proxy,'memory':'model','analyser':analyzer.detector})
                                metadata['local_feature_conditioning']={'policy':policy,'feature_bank_sha256':f['bank_sha256'],'gain':cfg['latent_injection_gain'],'reference_pixel_paste':False,'trainable_parameters':0}
                                output.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
                            record.update(status='complete',output=str(output),output_sha256=sha(output),metadata_sha256=sha(output.with_suffix('.json')))
                    except Exception as error:record['error']=f'{type(error).__name__}: {error}'
                    write_new(dest,record)
                rows.append(record);print('LOCAL GENERATION',len(rows),'/48',key,record['status'],record.get('error',''),flush=True)
    write_new(OUT/'comparison.json',{'signature_sha256':sha(lock),'rows':rows,'final_test_used':False})


if __name__=='__main__':main()
