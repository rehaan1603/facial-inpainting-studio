"""Matched external baseline report; identities, not images, are statistical units."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha,write_new
from report_distortion_aware import METRICS,paired
BASE=ROOT/'outputs/refldm_development_v1'

def main():
    rows=json.loads((BASE/'evaluation.json').read_text())['rows']
    assert len(rows)==48 and len({r['key'] for r in rows})==48
    controls=json.loads((ROOT/'outputs/distortion_aware_v1/evaluation.json').read_text())['rows']
    arms={a:[r for r in rows if r['mode']==a] for a in ['native','composited']}
    arms['observed']=[r for r in controls if r['mode']=='observed']
    for name,mode,strength in [('sdxl_low','standard',.5),('sdxl_high','standard',.99),('preserve_high','preserve',.99)]:
        arms[name]=[r for r in controls if r['mode']==mode and r.get('strength')==strength and r.get('scale')==.8]
    for group in arms.values():
        assert len(group)==24
        for r in group:
            if r['status']=='complete':assert sha(r['output'])==r['output_sha256']
    summaries={};grouped={}
    for name,group in arms.items():
        grouped[name]=[]
        for identity in sorted({r['identity'] for r in group}):
            subset=[r for r in group if r['identity']==identity and r['kind']!='removal'];assert len(subset)==5
            values={}
            for metric in METRICS:
                vals=[r['evaluation'].get('metrics',{}).get(metric) for r in subset]
                values[metric]=float(np.mean(vals)) if all(v is not None for v in vals) else None
            grouped[name].append({'identity':identity,'evaluation':{'metrics':values}})
        for kind in sorted({r['kind'] for r in group})+['all_partial']:
            subset=[r for r in group if (r['kind']!='removal' if kind=='all_partial' else r['kind']==kind)]
            entry={'rows':len(subset),'failed':sum(r['evaluation']['status']!='complete' for r in subset)}
            for metric in METRICS:
                vals=[r['evaluation'].get('metrics',{}).get(metric) for r in subset];valid=[v for v in vals if v is not None]
                entry[metric]={'mean':float(np.mean(valid)) if valid else None,'valid':len(valid)}
            summaries[name+'/'+kind]=entry
    contrasts={f'{a}_minus_{b}/{m}':paired(grouped[a],grouped[b],m)
        for a in ['native','composited'] for b in ['sdxl_low','sdxl_high'] for m in ['facenet_cosine','hole_mae']}
    previous=0
    for index,key in enumerate(sorted(contrasts,key=lambda k:contrasts[k]['p_exact'] if contrasts[k]['p_exact'] is not None else 1)):
        c=contrasts[key];previous=max(previous,min(1,(8-index)*(c['p_exact'] if c['p_exact'] is not None else 1)));c['p_holm']=previous
    safeguards={a:{m:paired(grouped[a],grouped['observed'],m) for m in METRICS} for a in ['native','composited']}
    result={'date':'2026-09-22','rows':48,'complete':sum(r['evaluation']['status']=='complete' for r in rows),
        'identity_units':4,'summaries':summaries,'primary_contrasts':contrasts,'versus_unchanged':safeguards,
        'evaluation_sha256':sha(BASE/'evaluation.json'),'final_test_used':False,'pretraining_identity_overlap':'unknown'}
    write_new(BASE/'summary.json',result);write_new(ROOT/'research/refldm_results_v1.json',result)
    lines=['# ReF-LDM matched development comparison — 22 September 2026','',
        'External author baseline, not a novel project method. Four already-observed development identities; six damage types; 50 DDIM steps, seed 17, CFG 1.5. Native whole-image outputs and exact outside-mask compositions are evaluated separately. No clean target/gallery conditions generation. Pretraining overlap is unknown. Eight final identities remain untouched.','',
        'Primary analysis averages five partially degraded conditions within each identity. Removal is exploratory and out of domain. Eight predeclared tests use exact sign flips, identity bootstrap intervals and Holm correction. Four identities are underpowered. Native outputs may change visible pixels; their actual changes are recorded separately without altering frozen metric formulas.','',
        '| Arm (20 partial cases) | FaceNet | Hole MAE | LPIPS | NIQE |','|---|---:|---:|---:|---:|']
    for name in arms:
        s=summaries[name+'/all_partial'];lines.append('| '+name+' | '+' | '.join(str(s[m]['mean']) for m in ['facenet_cosine','hole_mae','lpips','niqe'])+' |')
    lines+=['','## Primary comparisons','','| Contrast | Identities | Delta | CI95 | Holm p |','|---|---:|---:|---|---:|']
    for k,c in contrasts.items():lines.append(f"| {k} | {c['identities']} | {c['mean_delta']} | {c['ci95']} | {c['p_holm']} |")
    lines+=['','All metrics and per-condition coverage are in `refldm_results_v1.json`. Historical 0.8180 covers a different condition mixture and is not a matched comparator. No website promotion or publication-ready superiority follows from this small screen.']
    (ROOT/'research/REFLDM_RESULTS_V1.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    cases=json.loads((ROOT/'outputs/distortion_aware_v1/inference_manifest.json').read_text())['cases']
    evaluation={c['case_id']:c for c in json.loads((ROOT/'outputs/distortion_aware_v1/evaluation_manifest.json').read_text())['cases']}
    for identity in sorted({c['identity'] for c in cases}):
        sheet=Image.new('RGB',(6*192,6*218),'white');draw=ImageDraw.Draw(sheet)
        for y,c in enumerate(c for c in cases if c['identity']==identity):
            paths=[('observed',c['observed']),('clean (evaluation)',evaluation[c['case_id']]['target']['path'])]
            for arm in ['sdxl_high','preserve_high','native','composited']:
                r=next(r for r in arms[arm] if r['case_id']==c['case_id']);paths.append((arm,r.get('output')))
            for x,(label,path) in enumerate(paths):
                draw.text((x*192,y*218),c['kind']+' / '+label,fill='black')
                if path:sheet.paste(Image.open(path).convert('RGB').resize((192,192)),(x*192,y*218+24))
        sheet.save(BASE/(identity+'_review.jpg'))
    print(json.dumps({name:summaries[name+'/all_partial'] for name in arms},indent=2))

if __name__=='__main__':main()
