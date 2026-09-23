"""Report the frozen reference-proxy experiment and its simple-control ablations."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha,write_new
from report_distortion_aware import METRICS,paired
BASE=ROOT/'outputs/proxy_calibration_v1'

def main():
    data=json.loads((BASE/'evaluation.json').read_text());rows=data['rows'];assert len(rows)==20
    cfg=json.loads((ROOT/'research/protocols/proxy_calibration_v1.json').read_text())
    arms={a:[r for r in rows if r['mode']==a] for a in cfg['arms']};summaries={}
    for name,group in arms.items():
        assert len(group)==4;summaries[name]={'rows':4,'scored':sum(r['evaluation']['status']=='complete' for r in group)}
        for r in group:
            if r['status']=='complete':assert sha(r['output'])==r['output_sha256']
        for metric in METRICS:
            vals=[r['evaluation'].get('metrics',{}).get(metric) for r in group];vals=[v for v in vals if v is not None]
            summaries[name][metric]={'mean':float(np.mean(vals)) if vals else None,'valid':len(vals)}
    contrasts={f'proxy_spatial_minus_{control}/{metric}':paired(arms['proxy_spatial'],arms[control],metric)
        for control in ['fixed_half','proxy_global'] for metric in ['facenet_cosine','hole_mae']}
    previous=0
    for n,k in enumerate(sorted(contrasts,key=lambda k:contrasts[k]['p_exact'] if contrasts[k]['p_exact'] is not None else 1)):
        c=contrasts[k];previous=max(previous,min(1,(4-n)*(c['p_exact'] if c['p_exact'] is not None else 1)));c['p_holm']=previous
    result={'date':'2026-09-23','identities':4,'previously_observed':True,'rows':20,'complete':sum(r['evaluation']['status']=='complete' for r in rows),
        'summaries':summaries,'primary_contrasts':contrasts,'evaluation_sha256':sha(BASE/'evaluation.json'),
        'proxy_generations':len(list(BASE.glob('*/*_receipt.json'))),'final_test_used':False}
    write_new(ROOT/'research/proxy_calibration_results_v1.json',result)
    write_new(ROOT/'research/proxy_calibration_evidence_v1.json',{'signature':json.loads((BASE/'signature.json').read_text()),
        'evaluation_signature':json.loads((BASE/'evaluation_signature.json').read_text()),'rows':[{k:v for k,v in r.items() if k!='output'} for r in rows]})
    lines=['# Reference-proxy calibration — 23 September 2026','',
        'Frozen exploratory screen on four previously observed mixed-damage development cases. First supplied reference is synthetically damaged five ways and restored using only the remaining three references. Its known original supervises a nine-coefficient spatial blending rule and a global blend control; no restoration-backbone weights are trained. No actual target truth or withheld gallery enters generation or calibration. Proxy images, fitted coefficients and output photos remain local.','',
        'The predictor sees local input/output statistics, not a target degradation label. Proxy corruption severity is fixed medium, matching the development regime; this is not demonstrated blind real-world generalization. Reusing mask coordinates across aligned photos may misalign semantic regions. Mixing weights are not probabilities of correctness.','',
        '| Arm | FaceNet | Gallery FaceNet | Hole MAE | LPIPS |','|---|---:|---:|---:|---:|']
    for a,s in summaries.items():lines.append('| '+a+' | '+' | '.join(str(s[m]['mean']) for m in ['facenet_cosine','facenet_gallery_cosine','hole_mae','lpips'])+' |')
    lines+=['','| Primary contrast | Identities | Delta | 95% identity bootstrap CI | Holm p |','|---|---:|---:|---|---:|']
    for k,c in contrasts.items():lines.append(f"| {k} | {c['identities']} | {c['mean_delta']} | {c['ci95']} | {c['p_holm']} |")
    lines+=['','The statistical unit is identity. Four primary tests were specified before generation. All other metrics and failure coverage are retained in JSON. The historical 0.8180 aggregate is not comparable to this four-mixed-case subset. Success requires benefit beyond both fixed/global blending without material fidelity or perceptual regression; a favorable single metric is insufficient. Novelty and unfamiliar-person generalization are not established by this screen.']
    (ROOT/'research/PROXY_CALIBRATION_RESULTS_V1.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    sheet=Image.new('RGB',(6*192,4*216),'white');draw=ImageDraw.Draw(sheet)
    targets={c['case_id']:c for c in json.loads((ROOT/'outputs/distortion_aware_v1/evaluation_manifest.json').read_text())['cases']}
    for y,cid in enumerate(cfg['cases']):
        paths=[(a,next(r.get('output') for r in arms[a] if r['case_id']==cid)) for a in cfg['arms']]
        paths.append(('clean evaluation',targets[cid]['target']['path']))
        for x,(a,p) in enumerate(paths):
            draw.text((x*192,y*216),cid+' '+a,fill='black')
            if p:sheet.paste(Image.open(p).convert('RGB').resize((192,192)),(x*192,y*216+24))
    sheet.save(BASE/'review.jpg');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
