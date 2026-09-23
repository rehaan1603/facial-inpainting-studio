"""Execute frozen restoration on newly selected development inputs only."""
import json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha,write_new
from scripts.distortion_aware_study import RESERVED

def main():
    base=ROOT/'outputs/unfamiliar_smoke_v1'
    deadline=time.monotonic()+1800
    while not (base/'inference_manifest.json').exists() or not (ROOT/'outputs/reference_risk_diagnostic_v1/comparison.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Preparation or preceding GPU diagnostic incomplete')
        time.sleep(3)
    cases=json.loads((base/'inference_manifest.json').read_text())['cases']
    assert len(cases)==4 and not ({c['identity'] for c in cases}&RESERVED)
    write_new(base/'generation_signature.json',{'runner_sha256':sha(__file__),'generator_sha256':sha(ROOT/'scripts/refldm_restore.py'),
        'manifest_sha256':sha(base/'inference_manifest.json'),'protocol_sha256':sha(ROOT/'research/protocols/unfamiliar_smoke_v1.json')})
    rows=[]
    for c in cases:
        common={k:c[k] for k in ['case_id','identity','kind']};dest=base/'results'/c['case_id'];dest.mkdir(parents=True,exist_ok=False)
        rows.append(dict(common,key=c['case_id']+'_observed',mode='observed',status='complete',output=c['observed'],output_sha256=c['observed_sha256']))
        try:
            for path,digest in [(c['observed'],c['observed_sha256']),(c['mask'],c['mask_sha256'])]+[(r['path'],r['sha256']) for r in c['references']]:assert sha(path)==digest
            command=[sys.executable,str(ROOT/'scripts/refldm_restore.py'),'--image',c['observed'],'--mask',c['mask'],'--references',*[r['path'] for r in c['references']],'--output',str(dest/'result.png')]
            with (dest/'inference.log').open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=log,check=True,timeout=600)
            for name,file in [('native','result_raw.png'),('composited','result.png')]:
                rows.append(dict(common,key=c['case_id']+'_'+name,mode=name,status='complete',output=str(dest/file),output_sha256=sha(dest/file)))
        except Exception as e:
            for name in ['native','composited']:rows.append(dict(common,key=c['case_id']+'_'+name,mode=name,status='failed',error=f'{type(e).__name__}: {e}'))
        write_new(dest/'rows.json',rows[-3:]);print(c['case_id'],rows[-1]['status'],flush=True)
    write_new(base/'comparison.json',{'rows':rows,'final_test_used':False})
if __name__=='__main__':main()
