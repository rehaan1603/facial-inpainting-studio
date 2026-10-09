"""Verify frozen inputs, output coverage and reused metrics before reporting."""
import json
from pathlib import Path
from extended_evaluation_metrics import ROOT,sha

def main():
    out=ROOT/'outputs/focused_eye_guidance_replication_v2'
    cfg=json.loads((out/'config.json').read_text());summary=json.loads((out/'summary.json').read_text())
    assert len(summary['rows'])==18 and summary['all_completed']
    assert sha(ROOT/'outputs/masked_task_v1/cases.json')==cfg['cases_sha256']
    for key,file in [('runner_sha256','run_focused_eye_guidance_replication_v2.py'),('generator_sha256','reference_inpaint.py'),
                     ('proxy_sha256','run_focused_eye_guidance_v1.py'),('scorer_sha256','score_masked_sdxl_v1.py')]:
        assert sha(ROOT/'scripts'/file)==cfg[key]
    assert sha(ROOT/'research/FOCUSED_EYE_GUIDANCE_REPLICATION_PREFLIGHT_V2.md')==cfg['protocol_sha256']
    old={}
    for arm,folder in [('control','focused_reference_finish_v1'),('candidate','focused_eye_guidance_v1')]:
        for r in json.loads((ROOT/'outputs'/folder/'summary.json').read_text())['rows']:
            if r['identity'] in cfg['selected_identities'] and r['location']=='eyes':old[arm,r['identity']]=r
    cases={c['identity']:c for c in json.loads((ROOT/'outputs/masked_task_v1/cases.json').read_text()) if c['location']=='eyes'}
    for ident in cfg['selected_identities']:
        c=cases[ident]
        for k in ['input','mask']:assert sha(c[k])==c[k+'_sha256']
        refs=cfg['references'][ident+'/clean'];assert len(refs)==4 and len({r['sha256'] for r in refs})==4
        for r in refs:assert sha(r['path'])==r['sha256'] and r['sha256']!=c['target_sha256']
    differences=[]
    for r in summary['rows']:
        assert sha(out/r['file'])==r['sha256']
        assert sha(out/r['file'].replace('.png','_raw.png'))==r['raw_sha256']
        assert r['metrics']['visible_changed']==0
        assert r['metrics']['facenet_cosine'] is not None and r['metrics']['arcface_conditioning_cosine'] is not None
        if r['seed']==17:
            control=old[r['condition'],r['identity']];assert r['sha256']==control['sha256'] and r['raw_sha256']==control['raw_sha256']
            for key,value in r['metrics'].items():
                if isinstance(value,(int,float)) and isinstance(control['metrics'].get(key),(int,float)):
                    differences.append(abs(value-control['metrics'][key]))
    assert max(differences)==0.,'Reused metrics must reproduce exactly'
    receipt=dict(rows=18,new_generations=12,reused_images=6,matched_pairs=9,identities=3,seeds=cfg['seeds'],
                 facenet_valid=18,arcface_valid=18,visible_changed_pixels=0,maximum_reused_metric_difference=max(differences),
                 summary_sha256=sha(out/'summary.json'),config_sha256=sha(out/'config.json'),
                 verification_source_sha256=sha(__file__),passed=True,checkpoints_changed=False)
    (ROOT/'research/focused_eye_guidance_replication_v2_verification.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
