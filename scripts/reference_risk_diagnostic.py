"""Reference sensitivity and matched-coverage controls; no target access during generation."""
import json,sys,subprocess,math
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha,write_new

def gate(observed,candidate,mask,risk,fraction=.25):
    result=candidate.copy();indices=np.flatnonzero(mask)
    if not len(indices):raise ValueError('Empty mask')
    if risk.shape!=mask.shape or not np.isfinite(risk).all():raise ValueError('Invalid risk map')
    chosen=indices[np.argsort(-risk.flat[indices],kind='stable')[:math.ceil(fraction*len(indices))]]
    result.reshape(-1,3)[chosen]=observed.reshape(-1,3)[chosen]
    result[~mask]=observed[~mask]
    return result,chosen

def main():
    protocol=ROOT/'research/protocols/reference_risk_diagnostic_v1.json';cfg=json.loads(protocol.read_text())
    data=json.loads((ROOT/'outputs/distortion_aware_v1/inference_manifest.json').read_text())
    cases=[next(c for c in data['cases'] if c['case_id']==name) for name in cfg['cases']]
    assert {c['identity'] for c in cases}=={'1306','2790','1043','787'}
    out=ROOT/'outputs/reference_risk_diagnostic_v1';out.mkdir(exist_ok=False)
    write_new(out/'signature.json',{'protocol_sha256':sha(protocol),'runner_sha256':sha(__file__),
        'generator_sha256':sha(ROOT/'scripts/refldm_restore.py'),'input_manifest_sha256':sha(ROOT/'outputs/distortion_aware_v1/inference_manifest.json')})
    baseline=json.loads((ROOT/'outputs/refldm_development_v1/comparison.json').read_text())['rows'];rows=[]
    for c in cases:
        folder=out/c['case_id'];folder.mkdir();common={k:c[k] for k in ['case_id','identity','kind']}
        try:
            for path,digest in [(c['observed'],c['observed_sha256']),(c['mask'],c['mask_sha256'])]+[(r['path'],r['sha256']) for r in c['references']]:
                assert sha(path)==digest
            base=next(r for r in baseline if r['case_id']==c['case_id'] and r['mode']=='native')
            assert sha(base['output'])==base['output_sha256']
            raw=[np.asarray(Image.open(base['output']).convert('RGB'))]
            for omit in range(4):
                path=folder/f'omit_{omit}.png'
                command=[sys.executable,str(ROOT/'scripts/refldm_restore.py'),'--image',c['observed'],'--mask',c['mask'],'--references',
                    *[r['path'] for i,r in enumerate(c['references']) if i!=omit],'--output',str(path)]
                with (folder/f'omit_{omit}.log').open('w') as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=log,check=True,timeout=600)
                raw.append(np.asarray(Image.open(folder/f'omit_{omit}_raw.png').convert('RGB')))
                print(c['case_id'],'omit',omit,'complete',flush=True)
            observed=np.asarray(Image.open(c['observed']).convert('RGB'));mask=np.asarray(Image.open(c['mask']).convert('L'))>=128
            risk=np.stack(raw).astype(np.float64).std(axis=0).mean(axis=2)/255
            edit=np.abs(raw[0].astype(float)-observed.astype(float)).mean(axis=2)/255
            np.save(folder/'disagreement.npy',risk);np.save(folder/'edit_magnitude.npy',edit)
            arrays={'all_reference':np.where(mask[...,None],raw[0],observed)}
            for name,feature in [('disagreement_gate',risk),('edit_gate',edit)]:
                arrays[name],chosen=gate(observed,raw[0],mask,feature)
                np.save(folder/(name+'_preserved_indices.npy'),chosen)
            for name,array in arrays.items():
                path=folder/(name+'.png');Image.fromarray(array).save(path)
                rows.append(dict(common,key=c['case_id']+'_'+name,mode=name,status='complete',output=str(path),output_sha256=sha(path)))
        except Exception as e:
            for name in ['all_reference','disagreement_gate','edit_gate']:rows.append(dict(common,key=c['case_id']+'_'+name,mode=name,status='failed',error=f'{type(e).__name__}: {e}'))
        write_new(folder/'rows.json',rows[-3:])
    write_new(out/'comparison.json',{'rows':rows,'final_test_used':False})
if __name__=='__main__':main()
