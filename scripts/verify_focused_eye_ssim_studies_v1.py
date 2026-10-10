"""Provenance/preservation verification; export scores without photo pixels/features."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def verify_replay_pair(replay, control, saved):
    """Reopen physical replay files instead of trusting their saved receipt."""
    for suffix, check in saved.items():
        new = replay.with_name(replay.stem + suffix)
        old = control.with_name(control.stem + suffix)
        assert sha(new) == check['new_sha256']
        assert sha(old) == check['control_sha256']
        assert np.array_equal(np.asarray(Image.open(new)), np.asarray(Image.open(old)))


def main():
    assets=[]
    # Compare both installed filename aliases against the pinned download hashes.
    for r in json.loads((ROOT/'research/osor_downloads.json').read_text()):
        if r['repository']!='diffusers/stable-diffusion-xl-1.0-inpainting-0.1':continue
        original=Path(r['path']);paths=[original]
        if '.fp16.' in original.name:
            alias=original.with_name(original.name.replace('.fp16.','.'))
            if alias.exists():paths.append(alias)
        for path in paths:
            actual=sha(path);assert actual==r['sha256'],str(path)
            assets.append(dict(filename=path.relative_to(original.parents[len(Path(r['filename']).parts)-1]).as_posix(),
                sha256=actual,bytes=path.stat().st_size,revision=r['revision']))
    cases={c['identity']:c for c in json.loads((ROOT/'outputs/masked_task_v1/cases.json').read_text()) if c['location']=='eyes'}
    studies={}
    for folder,protocol,runner in [
        ('focused_eye_freeu_v1','FOCUSED_EYE_FREEU_AUDIT_AND_PREFLIGHT_V1.md','run_focused_eye_freeu_v1.py'),
        ('focused_eye_local_appearance_v1','FOCUSED_EYE_LOCAL_APPEARANCE_PREFLIGHT_V1.md','run_focused_eye_local_appearance_v1.py'),
        ('focused_eye_steps_v1','FOCUSED_EYE_STEPS_PREFLIGHT_V1.md','run_focused_eye_steps_v1.py')]:
        out=ROOT/'outputs'/folder;cfg=json.loads((out/'config.json').read_text());summary=json.loads((out/'summary.json').read_text())
        assert summary['all_attempted'] and summary['all_completed']
        assert sha(ROOT/'outputs/masked_task_v1/cases.json')==cfg['cases_sha256']
        assert sha(ROOT/'scripts'/runner)==cfg['runner_sha256']
        assert sha(ROOT/'research'/protocol)==cfg['protocol_sha256']
        generator_key='source_sha256' if 'local_appearance' in folder else 'generator_sha256'
        assert sha(ROOT/'scripts/reference_inpaint.py')==cfg[generator_key]
        assert sha(ROOT/'scripts/score_masked_sdxl_v1.py')==cfg['scorer_sha256']
        if folder.endswith('freeu_v1'):
            for path,digest in cfg['implementation'].items():assert sha(path)==digest
            replay=json.loads((out/'replay.json').read_text());checks=list(replay.values())
            replay_calls=1
        elif 'local_appearance' in folder:
            assert sha(ROOT/'scripts/local_appearance_processor_v1.py')==cfg['helper_sha256']
            assert sha(cfg['attention_implementation_path'])==cfg['attention_implementation_sha256']
            for r in cfg['appearance_features'].values():assert sha(r['path'])==r['sha256']
            replay=json.loads((out/'replays.json').read_text());checks=[v for group in replay.values() for v in group.values()]
            replay_calls=2
        else:
            assert sha(ROOT/'scripts/run_focused_eye_freeu_v1.py')==cfg['control_copy_helper_sha256']
            replay=json.loads((out/'replay.json').read_text());checks=list(replay.values());replay_calls=1
        assert all(v['pixel_exact'] and v['new_sha256']==v['control_sha256'] for v in checks)
        if folder.endswith('freeu_v1'):
            verify_replay_pair(out/'replay/263_eyes_control3_17.png', out/'263_eyes_control3_17.png', replay)
        elif 'local_appearance' in folder:
            for name, arm in [('full_mask', 'global'), ('appearance_off', 'control3')]:
                verify_replay_pair(out/'replay'/f'263_{name}.png', out/f'263_eyes_{arm}_17.png', replay[name])
        else:
            verify_replay_pair(out/'replay/263_control3.png', out/'263_eyes_control3_17.png', replay)
        exported=[];differences=[]
        for r in summary['rows']:
            p=out/r['file'];assert sha(p)==r['sha256'] and sha(p.with_name(p.stem+'_raw.png'))==r['raw_sha256']
            c=cases[r['identity']]
            for key in ['input','mask']:assert sha(c[key])==c[key+'_sha256']
            for ref in cfg['references'][r['identity']+'/clean']:
                assert sha(ref['path'])==ref['sha256'] and ref['sha256']!=c['target_sha256']
            rgb=np.asarray(Image.open(p).convert('RGB'));observed=np.asarray(Image.open(c['input']).convert('RGB'))
            mask=np.asarray(Image.open(c['mask']))>127
            assert np.array_equal(rgb[~mask],observed[~mask]) and r['metrics']['visible_changed']==0
            if r.get('reused'):
                old_path=Path(r['source'])
                old_rows=json.loads((old_path.parent/'summary.json').read_text())['rows']
                old=next(x for x in old_rows if x['file']==old_path.name)
                assert r['metrics']==old['metrics'] and r['regions']==old['regions']
                differences.extend(abs(v-old['metrics'][k]) for k,v in r['metrics'].items() if isinstance(v,(int,float)))
            allowed=['identity','location','condition','seed','status','sha256','raw_sha256','reused','seconds',
                     'peak_allocated_bytes','metrics','regions','target_detections','output_detections','error']
            exported.append({k:r[k] for k in allowed if k in r})
        fresh=[r for r in summary['rows'] if not r.get('reused')]
        receipt=dict(records=len(summary['rows']),new_candidates=len(fresh),reused_controls=len(summary['rows'])-len(fresh),
            replay_gpu_calls=replay_calls,replays_exact=True,replay_files_reopened=True,visible_changed_pixels=0,
            facenet_valid=sum(r['metrics']['facenet_cosine'] is not None for r in summary['rows']),
            arcface_valid=sum(r['metrics']['arcface_conditioning_cosine'] is not None for r in summary['rows']),
            maximum_reused_metric_difference=max(differences,default=0),
            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in fresh),
            candidate_seconds=[r['seconds'] for r in fresh],summary_sha256=sha(out/'summary.json'),config_sha256=sha(out/'config.json'))
        studies[folder]=receipt
        report=dict(date='2026-10-10',rows=exported,verification=receipt,config_sha256=sha(out/'config.json'),
            protocol_sha256=cfg['protocol_sha256'],runner_sha256=cfg['runner_sha256'],generator_sha256=cfg[generator_key],
            evaluation_sha256=cfg['scorer_sha256'],replay_checks=replay,parameters=dict(guidance=3.,steps=cfg['steps'],scale=.8,
            appearance_scale=.2 if 'local_appearance' in folder else 0.,freeu=cfg.get('freeu'),resolution=1024,export_resolution=512,training=False))
        if (out/'visual_review.json').exists():report['visual_review']=json.loads((out/'visual_review.json').read_text())
        (ROOT/'research'/(folder+'_metrics.json')).write_text(json.dumps(report,indent=2,allow_nan=False))
    final=dict(date='2026-10-10',studies=studies,model_assets=assets,passed=True,training=False,
        checkpoints_changed=False,production_changed=False,verifier_sha256=sha(__file__))
    (ROOT/'research/focused_eye_ssim_studies_v1_verification.json').write_text(json.dumps(final,indent=2,allow_nan=False))
    print(json.dumps(dict(studies=studies,pinned_model_files_verified=len(assets),passed=True),indent=2))


if __name__=='__main__':main()
