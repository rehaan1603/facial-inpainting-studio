"""Frozen second-cycle development study; existing v1 results never overwritten."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha,write_new
from src.reference_fusion.feature_bank import analyze_runtime
from regional_reference_inpaint import reconstruct_routed

def main():
    import diffusers
    base=ROOT/'outputs/regional_routing_v2';base.mkdir(exist_ok=True)
    protocol_path=ROOT/'research/protocols/regional_routing_protocol_v2.json';cfg=json.loads(protocol_path.read_text())
    cases_path=ROOT/cfg['cases_manifest'];cases=json.loads(cases_path.read_text())['cases']
    oldbase=ROOT/'outputs/generalization_v1';oldcomp=json.loads((oldbase/'comparison.json').read_text())
    sources=[*sorted((ROOT/'src').rglob('*.py')),Path(__file__),ROOT/'scripts/regional_reference_inpaint.py',ROOT/'scripts/reference_inpaint.py']
    package=Path(diffusers.__file__).parent
    signature={'protocol_sha256':sha(protocol_path),'cases_sha256':sha(cases_path),'old_comparison_sha256':sha(oldbase/'comparison.json'),
               'sources':{str(p.relative_to(ROOT)):sha(p) for p in sources},
               'diffusers_version':diffusers.__version__,'diffusers_sources':{str(p.relative_to(package)):sha(p) for p in [package/'models/attention_processor.py',package/'models/embeddings.py',package/'image_processor.py']}}
    lock=base/'signature.json'
    if lock.exists():
        if json.loads(lock.read_text())!=signature:raise ValueError('Frozen routing experiment changed; use a new version')
    else:write_new(lock,signature)
    from src.reference_fusion.gpu_masks import make_gpu_runtime
    runtime=make_gpu_runtime();analyzer=None;rows=[]
    for case in cases:
        if case['split']!='validation':raise ValueError('Reserved final identities are forbidden')
        if sha(case['observed'])!=case['distortion']['observed_sha256'] or sha(case['mask'])!=case['distortion']['mask_sha256']:raise ValueError('Case changed')
        for r in case['references']:
            if sha(r['path'])!=r['sha256']:raise ValueError('Reference changed')
        refs=[r['path'] for r in case['references']]
        bank=analyze_runtime(case['observed'],case['mask'],refs,analyzer);analyzer=bank[2]
        for seed in cfg['seeds']:
            for policy in cfg['policies']:
                key=f"{case['case_id']}_seed{seed}_{policy}";record_path=base/'generations'/key/'record.json'
                if record_path.exists():
                    record=json.loads(record_path.read_text())
                    if record['status']=='complete' and sha(record['output'])!=record['output_sha256']:raise ValueError('Output changed')
                else:
                    record={'case_id':case['case_id'],'identity':case['identity'],'condition':case['condition'],'seed':seed,'policy':policy}
                    try:
                        if policy=='concat':
                            logical=next(r for r in oldcomp['logical_rows'] if r['case_id']==case['case_id'] and r['seed']==seed and r['policy']=='all')
                            previous=json.loads((oldbase/'generations'/logical['generation_key']/'record.json').read_text());request=previous['request']
                            expected={'image_sha256':sha(case['observed']),'mask_sha256':sha(case['mask']),'references_sha256':[sha(r) for r in refs],
                                      'seed':seed,'steps':cfg['steps'],'scale':cfg['scale'],'strength':cfg['strength'],'blend':cfg['blend'],'resolution':cfg['resolution']}
                            if previous['status']!='complete' or request!=expected or sha(previous['output'])!=previous['output_sha256']:raise ValueError('Prior concat control mismatch')
                            record.update(status='complete',output=previous['output'],output_sha256=previous['output_sha256'],reused_generation_key=logical['generation_key'])
                        else:
                            output=record_path.parent/'result.png'
                            metadata=reconstruct_routed(case['observed'],case['mask'],refs,output,policy,seed,cfg['steps'],cfg['scale'],cfg['strength'],cfg['blend'],runtime=runtime,bank=bank)
                            record.update(status='complete',output=str(output),output_sha256=sha(output),metadata_sha256=sha(output.with_suffix('.json')),seconds=metadata['inference_seconds_including_offload'])
                    except Exception as error:record.update(status='failed',error=f'{type(error).__name__}: {error}')
                    write_new(record_path,record)
                rows.append(record);print('ROUTING',len(rows),'/144',key,record['status'],record.get('error',''),flush=True)
    write_new(base/'comparison.json',{'signature_sha256':sha(lock),'rows':rows,'final_test_used':False})

if __name__=='__main__':main()
