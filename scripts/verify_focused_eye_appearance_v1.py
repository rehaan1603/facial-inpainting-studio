"""Verify model/feature provenance, wiring replays, metrics coverage and preservation."""
import json
from pathlib import Path
from extended_evaluation_metrics import ROOT,sha

def main():
    receipts={}
    for folder in ['focused_eye_appearance_v1','focused_eye_factorial_v1']:
        out=ROOT/'outputs'/folder;cfg=json.loads((out/'config.json').read_text());summary=json.loads((out/'summary.json').read_text())
        assert summary['all_completed'] and len(summary['rows'])==3
        assert sha(ROOT/'scripts/reference_inpaint.py')==cfg['source_sha256']
        runner='run_focused_eye_appearance_v1.py' if folder=='focused_eye_appearance_v1' else 'run_focused_eye_factorial_v1.py'
        assert sha(ROOT/'scripts'/runner)==cfg['runner_sha256']
        assert sha(ROOT/'scripts/score_masked_sdxl_v1.py')==cfg['scorer_sha256']
        if folder.endswith('factorial_v1'):
            assert sha(ROOT/'scripts/run_focused_eye_appearance_v1.py')==cfg['appearance_wrapper_sha256']
            assert sha(ROOT/'scripts/run_focused_eye_guidance_v1.py')==cfg['guidance_wrapper_sha256']
        for asset in cfg['appearance_assets']['files']:assert sha(asset['path'])==asset['sha256']
        for feature in cfg['appearance_features'].values():assert sha(feature['path'])==feature['sha256'] and feature['shape']==[2,4,257,1280]
        assert sha(ROOT/'outputs/masked_task_v1/cases.json')==cfg['cases_sha256']
        for refs in cfg['references'].values():
            for r in refs:assert sha(r['path'])==r['sha256']
        replay=json.loads((out/'replay.json').read_text());assert all(v['pixel_exact'] and v['new_sha256']==v['control_sha256'] for v in replay.values())
        wiring=json.loads((out/'wiring.json').read_text());assert len(wiring)==4
        assert wiring[0]['scales']==[.8,0.]
        assert all(r['scales']==[.8,.2] and r['tokens_per_reference']==[16,16] for r in wiring[1:])
        for r in summary['rows']:
            assert sha(out/r['file'])==r['sha256']
            assert sha(out/r['file'].replace('.png','_raw.png'))==r['raw_sha256']
            assert r['metrics']['visible_changed']==0
            assert r['metrics']['facenet_cosine'] is not None and r['metrics']['arcface_conditioning_cosine'] is not None
        receipts[folder]=dict(outputs=3,facenet_valid=3,arcface_valid=3,visible_changed_pixels=0,neutral_replay_byte_exact=True,
                              summary_sha256=sha(out/'summary.json'),config_sha256=sha(out/'config.json'),
                              peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in summary['rows']),passed=True)
    (ROOT/'research/focused_eye_appearance_verification_v1.json').write_text(json.dumps(dict(experiments=receipts,verification_source_sha256=sha(__file__),training=False,checkpoints_changed=False),indent=2))
    print(json.dumps(receipts,indent=2))

if __name__=='__main__':main()
