"""Keep original failures visible while reporting a separately recorded runtime retry."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha,write_new
from report_distortion_aware import METRICS

def main():
    base=ROOT/'outputs/unfamiliar_smoke_v1';recovery=ROOT/'outputs/unfamiliar_recovery_v1'
    original=json.loads((base/'evaluation.json').read_text())['rows'];retried=json.loads((recovery/'evaluation.json').read_text())['rows']
    replacements={r['key']:r for r in retried};rows=[replacements.get(r['key'],r) for r in original]
    assert len(original)==12 and len(retried)==2 and len(rows)==12
    for r in rows:
        if r['status']=='complete':assert sha(r['output'])==r['output_sha256']
    summaries={}
    for mode in ['observed','native','composited']:
        group=[r for r in rows if r['mode']==mode];entry={'cases':len(group)}
        for m in METRICS:
            values=[r['evaluation'].get('metrics',{}).get(m) for r in group];values=[v for v in values if v is not None]
            entry[m]={'mean':float(np.mean(values)) if values else None,'valid':len(values)}
        summaries[mode]=entry
    receipt={'date':'2026-09-23','identities':['1590','1529'],'initial_generations_succeeded':3,'initial_generations_scheduled':4,
        'initial_scored_rows':10,'initial_scheduled_rows':12,'runtime_retry_succeeded':1,'post_retry_scored_rows':sum(r['evaluation']['status']=='complete' for r in rows),
        'summaries':summaries,'original_evaluation_sha256':sha(base/'evaluation.json'),'recovery_evaluation_sha256':sha(recovery/'evaluation.json'),
        'final_test_used':False,'pretraining_overlap':'unknown','novelty_claim':False}
    write_new(ROOT/'research/unfamiliar_smoke_results_v1.json',receipt)
    write_new(ROOT/'research/unfamiliar_smoke_evidence_v1.json',{'original_rows':[{k:v for k,v in r.items() if k!='output'} for r in original],
        'separate_recovery_rows':[{k:v for k,v in r.items() if k!='output'} for r in retried]})
    lines=['# Unfamiliar development functionality check — 23 September 2026','',
        'Two identities absent from all previous local selected/attempted groups: 1590 and 1529. Hash-ranked selection, duplicate filtering, eight photographs per identity: one target, four references and three withheld gallery images. All reserved-final identities excluded before pixel access. These identities are now observed development data. Pretraining overlap is unknown.','',
        'Two conditions per identity: medium central-face Gaussian blur and mixed damage. Four reference-restoration runs were scheduled. Three completed initially; 1590_mixed terminated with native process exit code 3221227274 and no Python traceback. The cause is unresolved. An unchanged-settings retry succeeded; the original failure is retained. Initial scoring is 10/12 rows; after separately recorded recovery it is 12/12. This includes unchanged controls and native/composed variants, not twelve independent subjects.','',
        '| Arm, after runtime recovery | FaceNet | Hole MAE | LPIPS |','|---|---:|---:|---:|']
    for mode,s in summaries.items():lines.append('| '+mode+' | '+' | '.join(str(s[m]['mean']) for m in ['facenet_cosine','hole_mae','lpips'])+' |')
    lines+=['','No statistical superiority or unknown-person reliability claim follows from two identities. This verifies operation beyond the original examples, with a retained runtime failure. Full metric coverage and provenance are in the JSON receipts.']
    (ROOT/'research/UNFAMILIAR_SMOKE_RESULTS_V1.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    inputs=json.loads((base/'inference_manifest.json').read_text())['cases'];targets={c['case_id']:c for c in json.loads((base/'evaluation_manifest.json').read_text())['cases']}
    sheet=Image.new('RGB',(4*256,4*280),'white');draw=ImageDraw.Draw(sheet)
    for y,c in enumerate(inputs):
        paths=[('observed',c['observed']),('clean evaluation',targets[c['case_id']]['target']['path'])]
        for mode in ['native','composited']:paths.append((mode,next(r['output'] for r in rows if r['case_id']==c['case_id'] and r['mode']==mode)))
        for x,(label,path) in enumerate(paths):
            sheet.paste(Image.open(path).convert('RGB').resize((256,256)),(x*256,y*280+24));draw.text((x*256,y*280),c['case_id']+' '+label,fill='black')
    sheet.save(base/'review_after_recovery.jpg');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
