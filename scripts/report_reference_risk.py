"""Descriptive risk/coverage diagnostic; never treats pixels as independent subjects."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha,write_new
from report_distortion_aware import METRICS,paired
BASE=ROOT/'outputs/reference_risk_diagnostic_v1'

def auc(score,label):
    positive=int(label.sum());negative=len(label)-positive
    if not positive or not negative:return None
    order=np.argsort(score,kind='stable');ordered=score[order];ranks=np.empty(len(score),float)
    starts=np.r_[0,np.flatnonzero(np.diff(ordered))+1];ends=np.r_[starts[1:],len(score)]
    for start,end in zip(starts,ends):ranks[order[start:end]]=(start+1+end)/2
    return float((ranks[label].sum()-positive*(positive+1)/2)/(positive*negative))

def main():
    rows=json.loads((BASE/'evaluation.json').read_text())['rows'];assert len(rows)==12
    source=json.loads((ROOT/'outputs/distortion_aware_v1/inference_manifest.json').read_text())['cases']
    targets={c['case_id']:c for c in json.loads((ROOT/'outputs/distortion_aware_v1/evaluation_manifest.json').read_text())['cases']}
    read=lambda p:np.asarray(Image.open(p).convert('RGB')).astype(float)/255
    diagnostics=[]
    for case_id in ['1306_mixed','2790_mixed','1043_mixed','787_mixed']:
        c=next(c for c in source if c['case_id']==case_id);folder=BASE/case_id
        raw=ROOT/'outputs/refldm_development_v1'/case_id/'native.png';e=targets[case_id]['target']
        assert sha(e['path'])==e['sha256'] and sha(c['observed'])==c['observed_sha256']
        if not (folder/'disagreement.npy').exists():
            diagnostics.append({'case_id':case_id,'status':'failed'});continue
        target=read(e['path']);observed=read(c['observed']);candidate=read(raw);mask=read(c['mask'])[:,:,0]>=.5
        harm=(np.abs(candidate-target).mean(2)-np.abs(observed-target).mean(2))[mask]>0
        risk=np.load(folder/'disagreement.npy')[mask];edit=np.load(folder/'edit_magnitude.npy')[mask]
        diagnostics.append({'case_id':case_id,'status':'complete','masked_pixels':len(harm),'harm_fraction':float(harm.mean()),
            'disagreement_auc':auc(risk,harm),'edit_magnitude_auc':auc(edit,harm),'mean_reference_disagreement':float(risk.mean())})
    arms={name:[r for r in rows if r['mode']==name] for name in ['all_reference','disagreement_gate','edit_gate']}
    summaries={}
    for name,group in arms.items():
        summaries[name]={}
        for metric in METRICS:
            vals=[r['evaluation'].get('metrics',{}).get(metric) for r in group];vals=[v for v in vals if v is not None]
            summaries[name][metric]={'mean':float(np.mean(vals)) if vals else None,'valid':len(vals)}
    comparisons={name:{m:paired(arms[name],arms['all_reference'],m) for m in ['facenet_cosine','hole_mae','lpips']}
        for name in ['disagreement_gate','edit_gate']}
    result={'date':'2026-09-22','status':'exploratory, uncalibrated','rows':len(rows),'complete':sum(r['evaluation']['status']=='complete' for r in rows),
        'diagnostics':diagnostics,'summaries':summaries,'exploratory_contrasts':comparisons,'evaluation_sha256':sha(BASE/'evaluation.json'),'final_test_used':False}
    write_new(ROOT/'research/reference_risk_results_v1.json',result)
    lines=['# Reference-disagreement diagnostic — 22 September 2026','',
        'Four already-observed mixed-damage cases. Four leave-one-reference-out runs per case (16 generations), same seed. All-reference restoration and two matched-coverage gates produce 12 scored images. Both gates preserve the highest-ranked 25% of masked pixels, using reference disagreement or edit magnitude. No clean target/gallery enters generation or gating. This is neither calibrated uncertainty nor unfamiliar-person validation.','',
        '| Case | Harm fraction | Disagreement AUC | Edit-magnitude AUC |','|---|---:|---:|---:|']
    for d in diagnostics:lines.append('| '+d['case_id']+' | '+' | '.join(str(d.get(k)) for k in ['harm_fraction','disagreement_auc','edit_magnitude_auc'])+' |')
    lines+=['','AUC describes within-image prediction of pixel-error increase. Pixels are not independent statistical units; there is no pixel-level significance claim.','',
        '| Arm | FaceNet | Hole MAE | LPIPS |','|---|---:|---:|---:|']
    for name,s in summaries.items():lines.append('| '+name+' | '+' | '.join(str(s[m]['mean']) for m in ['facenet_cosine','hole_mae','lpips'])+' |')
    lines+=['','All metrics, coverage and exploratory identity-unit contrasts are in `reference_risk_results_v1.json`. No calibrated gate or website default is promoted. A gate may retain visible damage. Fresh identity-separated calibration/validation and simple-preservation controls remain necessary before a method claim.']
    (ROOT/'research/REFERENCE_RISK_RESULTS_V1.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
