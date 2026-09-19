"""Frozen validation comparison; resumable records, no silent retries/overwrites."""
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha,write_new
from src.reference_selection.reference_analyzer import ReferenceAnalyzer
from src.reference_selection.selection_baselines import select
from mask_aware_inpaint import reconstruct_selected

def main():
    base=ROOT/'outputs/generalization_v1';protocol_path=ROOT/'research/protocols/unseen_identity_protocol_v1.json'
    protocol=json.loads(protocol_path.read_text());config=protocol['experiment']
    manifest_path=base/'cases/manifest.json';manifest=json.loads(manifest_path.read_text())
    if sha(protocol_path)!=manifest['protocol_sha256']:raise ValueError('Protocol changed')
    if sha(ROOT/'scripts/reference_inpaint.py')!=config['generator_sha256']:raise ValueError('Frozen generator changed')
    sources=[*sorted((ROOT/'src').rglob('*.py')),ROOT/'scripts/mask_aware_inpaint.py',Path(__file__),ROOT/'configs/mask_aware_selection_v1.json']
    signature={'sources':{str(p.relative_to(ROOT)):sha(p) for p in sources},'manifest_sha256':sha(manifest_path),'protocol_sha256':sha(protocol_path)}
    lock=base/'comparison_signature.json'
    if lock.exists():
        if json.loads(lock.read_text())!=signature:raise ValueError('Implementation changed; version experiment instead of resuming')
    else:write_new(lock,signature)
    runtime={};analyzer=ReferenceAnalyzer();runtime['analyser']=analyzer.detector
    logical=[];cache={}
    for case in manifest['cases']:
        if case['split']!='validation':raise ValueError('Final data cannot enter development comparison')
        image,mask=Path(case['observed']),Path(case['mask'])
        if sha(image)!=case['distortion']['observed_sha256'] or sha(mask)!=case['distortion']['mask_sha256']:raise ValueError('Case changed')
        refs=[Path(r['path']) for r in case['references']]
        for p,r in zip(refs,case['references']):
            if sha(p)!=r['sha256']:raise ValueError('Reference changed')
        analysis_path=base/'analysis'/(case['case_id']+'.json')
        if analysis_path.exists():analysis=json.loads(analysis_path.read_text())
        else:analysis=analyzer.analyze(image,mask,refs);write_new(analysis_path,analysis)
        for seed in config['seeds']:
            for policy in config['policies']:
                chosen=select(analysis,policy,seed)
                request={'image_sha256':sha(image),'mask_sha256':sha(mask),'references_sha256':[sha(p) for p in chosen],
                         'seed':seed,'steps':config['steps'],'scale':0 if policy=='zero_scale' else config['scale'],
                         'strength':config['strength'],'blend':config['blend'],'resolution':config['resolution']}
                key=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
                record_path=base/'generations'/key/'record.json';output=record_path.parent/'result.png'
                if key not in cache:
                    if record_path.exists():
                        record=json.loads(record_path.read_text())
                        if record['request']!=request:raise ValueError('Cached request differs')
                        if record['status']=='complete' and sha(record['output'])!=record['output_sha256']:raise ValueError('Cached output changed')
                    else:
                        record={'request':request,'case_id':case['case_id'],'first_policy':policy}
                        try:
                            metadata=reconstruct_selected(image,mask,refs,output,policy,seed,config['steps'],config['scale'],config['strength'],config['blend'],runtime=runtime,analysis=analysis)
                            record.update(status='complete',output=str(output),output_sha256=sha(output),metadata_sha256=sha(output.with_suffix('.json')),seconds=metadata['inference_seconds_including_offload'])
                        except Exception as error:record.update(status='failed',error=f'{type(error).__name__}: {error}')
                        write_new(record_path,record)
                    cache[key]=record
                record=cache[key]
                logical.append({'case_id':case['case_id'],'identity':case['identity'],'condition':case['condition'],'seed':seed,'policy':policy,'selected_reference_hashes':request['references_sha256'],'generation_key':key,'status':record['status'],'error':record.get('error')})
                print('COMPARISON',len(logical),'/ 168',case['case_id'],seed,policy,record['status'],'unique',len(cache),flush=True)
    write_new(base/'comparison.json',{'signature_sha256':sha(lock),'logical_rows':logical,'unique_generations':len(cache),'final_test_used':False})

if __name__=='__main__':main()
