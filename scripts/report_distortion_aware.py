"""All-arm summaries, identity-unit paired inference, and local contact sheets."""
import itertools,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha,write_new
from distortion_aware_study import BASE,PROTOCOL,KINDS

METRICS=['facenet_cosine','facenet_gallery_cosine','arcface_conditioning_cosine','arcface_conditioning_gallery_cosine','lpips','ssim_rgb','psnr_rgb','hole_psnr','hole_mae','visible_mae','niqe','brisque']


def paired(a,b,metric='facenet_cosine'):
    left={r['identity']:r['evaluation'].get('metrics',{}).get(metric) for r in a}
    right={r['identity']:r['evaluation'].get('metrics',{}).get(metric) for r in b}
    deltas={i:left[i]-right[i] for i in sorted(left.keys()&right.keys()) if left[i] is not None and right[i] is not None}
    values=np.array(list(deltas.values()))
    if not len(values):return {'identities':0,'mean_delta':None,'ci95':None,'p_exact':None,'identity_deltas':{}}
    mean=float(values.mean());rng=np.random.default_rng(20260920)
    exact=[abs(np.mean(values*np.array(s))) for s in itertools.product([-1,1],repeat=len(values))]
    return {'identities':len(values),'mean_delta':mean,'ci95':np.quantile(rng.choice(values,(10000,len(values)),replace=True).mean(1),[.025,.975]).tolist(),
            'p_exact':float(np.mean(np.array(exact)>=abs(mean)-1e-12)),'identity_deltas':deltas}


def main():
    data=json.loads((BASE/'evaluation.json').read_text());rows=data['rows'];cfg=json.loads(PROTOCOL.read_text())
    assert len(rows)==312 and len({r['key'] for r in rows})==312
    def select(kind,mode='standard',strength=.99,scale=.8):
        return [r for r in rows if r['kind']==kind and r['mode']==mode and (mode=='observed' or (r['strength']==strength and r['scale']==scale))]
    summaries={};contrasts={}
    for kind in KINDS:
        summaries[kind]={}
        arms=[('observed',None,None)]+[(m,s,a) for m in ['standard','preserve'] for s in cfg['strengths'] for a in cfg['scales']]
        for mode,strength,scale in arms:
            arm=mode if mode=='observed' else f'{mode}_s{strength}_a{scale}'
            group=select(kind,mode,strength,scale);entry={'rows':len(group),'failed':sum(r['evaluation']['status']!='complete' for r in group)}
            for metric in METRICS:
                vals=[r['evaluation'].get('metrics',{}).get(metric) for r in group];vals=[v for v in vals if v is not None]
                entry[metric]={'mean':float(np.mean(vals)) if vals else None,'valid':len(vals)}
            summaries[kind][arm]=entry
        for strength in [.5,.75]:
            contrasts[f'{kind}_strength_{strength}_vs_.99']=paired(select(kind,strength=strength),select(kind))
        if kind!='removal':contrasts[f'{kind}_preserve_vs_standard']=paired(select(kind,'preserve'),select(kind))
    previous=0
    for index,key in enumerate(sorted(contrasts,key=lambda k:contrasts[k]['p_exact'] if contrasts[k]['p_exact'] is not None else 1)):
        value=contrasts[key]['p_exact'];previous=max(previous,min(1,(len(contrasts)-index)*(value if value is not None else 1)));contrasts[key]['p_holm']=previous
    safeguards={kind:{'low_vs_high':{m:paired(select(kind,strength=.5),select(kind),m) for m in METRICS},
                      'low_vs_observed':{m:paired(select(kind,strength=.5),select(kind,'observed'),m) for m in METRICS},
                      'preserve_vs_observed':{m:paired(select(kind,'preserve'),select(kind,'observed'),m) for m in METRICS}} for kind in KINDS}
    result={'summaries':summaries,'primary_contrasts':contrasts,'descriptive_safeguards':safeguards,
            'rows':len(rows),'generation_count':144,'identity_count':4,'seeds':[17],'final_test_used':False,
            'evaluation_sha256':sha(BASE/'evaluation.json'),'promotion':'Not eligible from screening alone'}
    write_new(BASE/'summary.json',result);write_new(ROOT/'research/distortion_aware_results_v1.json',result)
    safe=[{k:v for k,v in r.items() if k!='output'} for r in rows]
    write_new(ROOT/'research/distortion_aware_evidence_v1.json',{'signature':json.loads((BASE/'generation_signature.json').read_text()),'evaluation_signature':json.loads((BASE/'evaluation_signature.json').read_text()),'rows':safe})
    fmt=lambda value:'NA' if value is None else f'{value:.4f}'
    lines=['# Distortion-aware generation — development screening v1','','144 GPU generations, 144 preservation outputs and 24 unchanged-input controls; 312 scored rows. Four previously observed development identities, one fixed seed, six damage kinds, one central-face mask per identity and medium severity. This is a controlled screening study, not final validation.','',
           'Strengths 0.50, 0.75 and 0.99; adapter scales 0.8 and 1.2; 30-step schedule, guidance 5.0, fixed references and Poisson composition. Lower strength also uses fewer active steps; this is not yet an equal-active-step causal test. Preservation mixes 50% observed evidence within degraded regions, zero within fully removed regions, and preserves all known pixels exactly. The supplied class/confidence is not estimated from clean truth.','',
           '| Damage | Arm (scale 0.8) | FaceNet target ↑ | Gallery ↑ | LPIPS ↓ | Hole MAE ↓ | NIQE ↓ | BRISQUE ↓ |','|---|---|---:|---:|---:|---:|---:|---:|']
    for kind in KINDS:
        for arm in ['observed','standard_s0.5_a0.8','standard_s0.75_a0.8','standard_s0.99_a0.8','preserve_s0.99_a0.8']:
            e=summaries[kind][arm];lines.append('| '+kind+' | '+arm+' | '+' | '.join(fmt(e[m]['mean']) for m in ['facenet_cosine','facenet_gallery_cosine','lpips','hole_mae','niqe','brisque'])+' |')
    lines+=['','Full scale-1.2, preservation, ArcFace, SSIM, PSNR and coverage results are retained in `distortion_aware_results_v1.json`; no unfavorable arm is omitted from the evidence ledger.','',
            '| Primary FaceNet contrast | Identity n | Mean difference | 95% identity bootstrap interval | Exact p | Holm p |','|---|---:|---:|---|---:|---:|']
    for name,c in contrasts.items():lines.append(f"| {name} | {c['identities']} | {fmt(c['mean_delta'])} | {c['ci95']} | {fmt(c['p_exact'])} | {fmt(c['p_holm'])} |")
    lines+=['','## Interpretation boundary','','The 0.8180 historical all-reference mean was measured across a different three-condition/two-seed mixture. Comparing a new six-distortion average to that number does not establish an improvement. Use paired within-case settings here and a matched historical-protocol confirmation before making that claim.','',
            'Four identities provide weak uncertainty estimates; bootstrap intervals are unstable and the minimum two-sided exact p is 0.125. Statistical success cannot be claimed from this screen. Check unchanged-input controls: beating aggressive generation while underperforming the input is not successful restoration. ArcFace is the conditioning encoder and is diagnostic. No final identities, new adapter training or candidate reranking were used.','',
            'Local feature correspondence and novelty findings are reported separately. Seed confirmation, fresh development identities, severity/mask/reference variation and external baselines remain necessary.']
    (ROOT/'research/DISTORTION_AWARE_GENERATION_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    cases={c['case_id']:c for c in json.loads((BASE/'inference_manifest.json').read_text())['cases']}
    evalcases={c['case_id']:c for c in json.loads((BASE/'evaluation_manifest.json').read_text())['cases']}
    out=BASE/'contact_sheets';out.mkdir(exist_ok=True)
    for cid,c in cases.items():
        images=[('Clean scoring target',evalcases[cid]['target']['path']),('Observed',c['observed'])]
        for mode,s,a in [('standard',.5,.8),('standard',.75,.8),('standard',.99,.8),('standard',.99,1.2),('preserve',.5,.8),('preserve',.99,.8)]:
            row=next(r for r in rows if r['case_id']==cid and r['mode']==mode and r['strength']==s and r['scale']==a)
            images.append((f'{mode} s={s} a={a}',row.get('output')))
        sheet=Image.new('RGB',(1024,560),'white');draw=ImageDraw.Draw(sheet)
        for i,(label,path) in enumerate(images):
            x,y=i%4*256,i//4*280;draw.text((x+4,y+4),label,fill='black')
            if path:
                with Image.open(path) as im:sheet.paste(im.convert('RGB').resize((256,256)),(x,y+24))
            else:draw.text((x+4,y+60),'FAILED',fill='red')
        sheet.save(out/(cid+'.png'))
    print(json.dumps({'rows':len(rows),'failures':sum(r['evaluation']['status']!='complete' for r in rows),'primary_contrasts':len(contrasts)},indent=2))


if __name__=='__main__':main()
