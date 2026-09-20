"""Fresh-development confirmation: four identity units, four primary tests."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha,write_new
from report_distortion_aware import METRICS,paired
BASE=ROOT/'outputs/distortion_extension_v1'

def main():
    rows=json.loads((BASE/'evaluation.json').read_text())['rows']
    assert len(rows)==240 and len({r['key'] for r in rows})==240
    for r in rows:
        if r['status']=='complete':assert sha(r['output'])==r['output_sha256']
    def arm(r):return 'observed' if r['mode']=='observed' else f"{r['mode']}_{r['strength']}"
    summaries={}
    for kind in sorted({r['kind'] for r in rows}):
        for severity in ['medium','severe']:
            for name in ['observed','standard_0.5','standard_0.99','preserve_0.5','preserve_0.99']:
                group=[r for r in rows if r['kind']==kind and r['severity']==severity and arm(r)==name]
                entry={'rows':len(group),'metrics':{}}
                for m in METRICS:
                    vals=[r['evaluation'].get('metrics',{}).get(m) for r in group];vals=[v for v in vals if v is not None]
                    entry['metrics'][m]={'mean':float(np.mean(vals)) if vals else None,'valid':len(vals)}
                summaries[f'{kind}/{severity}/{name}']=entry
    # Require complete metric pairs across all ten degraded cases per identity.
    grouped={}
    for name in ['observed','standard_0.5','standard_0.99','preserve_0.5','preserve_0.99']:
        grouped[name]=[]
        for identity in sorted({r['identity'] for r in rows}):
            group=[r for r in rows if r['identity']==identity and r['kind']!='removal' and arm(r)==name]
            assert len(group)==10
            values={}
            for m in METRICS:
                vals=[r['evaluation'].get('metrics',{}).get(m) for r in group]
                values[m]=float(np.mean(vals)) if all(v is not None for v in vals) else None
            grouped[name].append({'identity':identity,'evaluation':{'metrics':values}})
    contrasts={}
    for name in ['standard_0.5','preserve_0.99']:
        for m in ['facenet_cosine','hole_mae']:
            contrasts[f'{name}_minus_standard_0.99/{m}']=paired(grouped[name],grouped['standard_0.99'],m)
    previous=0
    for index,k in enumerate(sorted(contrasts,key=lambda k:contrasts[k]['p_exact'] if contrasts[k]['p_exact'] is not None else 1)):
        c=contrasts[k];previous=max(previous,min(1,(len(contrasts)-index)*(c['p_exact'] if c['p_exact'] is not None else 1)));c['p_holm']=previous
    safeguards={name:{m:paired(grouped[name],grouped['observed'],m) for m in METRICS} for name in grouped if name!='observed'}
    result={'rows':len(rows),'summaries':summaries,'primary_contrasts':contrasts,'versus_unchanged_input':safeguards,'identity_units':4,'final_test_used':False,'evaluation_sha256':sha(BASE/'evaluation.json'),'failed_rows':[r['key'] for r in rows if r['evaluation']['status']!='complete']}
    write_new(BASE/'summary.json',result);write_new(ROOT/'research/distortion_extension_results_v1.json',result)
    write_new(ROOT/'research/distortion_extension_evidence_v1.json',{'signature':json.loads((BASE/'generation_signature.json').read_text()),'evaluation_signature':json.loads((BASE/'evaluation_signature.json').read_text()),'rows':[{k:v for k,v in r.items() if k!='output'} for r in rows]})
    lines=['# Fresh development confirmation: distortion and preservation','','Four additional development identities (386, 2087, 2289, 1485), six distortion types, medium central-face and severe half-face masks, seed 29. First reference recompressed at JPEG quality 20. These identities are now observed development data, never fresh final evaluation. 96 scheduled generations plus preservation and unchanged-input controls = 240 scheduled rows. Consult failed_rows and per-metric coverage; scheduled rows must not be described as successful generations.','', 'Severity and mask shape vary together, so cross-stratum effects are not attributable solely to severity. Reference recompression is not an isolated reference-quality ablation. Each comparison holds inputs, references and seed fixed. The primary unit is identity, averaging ten partially degraded cases; removal is reported separately. Four Holm-corrected tests were specified before generation.','','| Contrast | n | Delta | 95% identity bootstrap interval | Exact p | Holm p |','|---|---:|---:|---|---:|---:|']
    for k,c in contrasts.items():lines.append(f"| {k} | {c['identities']} | {c['mean_delta']} | {c['ci95']} | {c['p_exact']} | {c['p_holm']} |")
    lines+=['','All 60 condition/arm summaries, all metrics, coverage and unchanged-input safeguards are in distortion_extension_results_v1.json. Positive FaceNet differences and negative MAE differences favor the first arm. Four identities remain underpowered; this is confirmation of a development trend, not final evidence of superiority. Historical 0.8180 uses a different condition mixture and is not a comparable endpoint.']
    (ROOT/'research/DISTORTION_EXTENSION_RESULTS_V1.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(contrasts,indent=2))
if __name__=='__main__':main()
