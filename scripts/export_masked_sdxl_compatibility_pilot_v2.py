"""Validate fixed pilot evidence and export numbers/hashes without photographs.

Keep tensor checkpoints, embeddings, paths to source photographs and contact
sheets local. This exporter cannot promote a model or start another experiment.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/masked_sdxl_compatibility_pilot_v2'
EVAL=ROOT/'outputs/masked_sdxl_compatibility_pilot_eval_v2'
IDS=['6792','2250','2083','4164','674','476','1281','8748']

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf8'))

def check(record): assert sha(record['path'])==record['sha256'],record['path']

def public_pin(record):
    path=Path(record['path']).resolve()
    return dict(file=str(path.relative_to(ROOT)).replace('\\','/') if path.is_relative_to(ROOT) else path.name,
                sha256=record['sha256'])

def main():
    cache=read(BASE/'cache/receipt.json');training=read(BASE/'training/receipt.json');cfg=read(BASE/'training/config.json')
    trace=read(BASE/'training/trace.json');selection=read(BASE/'reference_selection.json')
    assert cache['status']=='complete' and cache['train_identities']==IDS
    assert training['status']=='complete' and training['completed_updates']==training['finite_gradient_updates']==len(trace)==64
    assert training['trainable_parameters']==cfg['trainable_parameters']==385 and cfg['training_ids']==IDS
    assert training['frozen_weights_have_gradients'] is False and training['backend_flags_restored'] is True
    assert sha(BASE/'training/config.json')==training['config_sha256'] and sha(BASE/'training/trace.json')==training['trace_sha256']
    assert cfg['protocol_sha256']==training['protocol_sha256']==sha(cfg['protocol_path'])
    assert sha(cfg['cache_receipt']['path'])==cfg['cache_receipt']['sha256']
    for item in cfg['source_pins']:check(item)
    for item in training['checkpoints'].values():check(item)
    for i,row in enumerate(trace):
        assert row['update']==i+1 and row['identity']==IDS[i%8] and row['timestep']==[100,300,500,700][i//16]
        assert row['noise_seed']==20261010+i+1
        assert all(math.isfinite(row[k]) for k in ['loss','gradient_norm','full_epsilon_mse','hole_epsilon_mse','step_seconds']) and row['gradient_norm']>0
        assert abs(row['loss']-(.25*row['full_epsilon_mse']+.75*row['hole_epsilon_mse']))<1e-6
    report=dict(scope='Real masked SDXL development pilot; all failures retained, no accuracy percentage or publication claim',
        training=dict(status=training['status'],identities=IDS,updates=64,updates_per_person=8,resolution=512,trainable_parameters=385,
            frozen_parameters=3366130756,neutral_max_absolute_error=training['neutral_max_absolute_error'],
            frozen_weights_have_gradients=False,backend_flags_restored=True,parameter_update=training['parameter_update'],
            peak_allocated_bytes=training['peak_allocated_bytes'],peak_reserved_bytes=training['peak_reserved_bytes'],
            elapsed_seconds=training['elapsed_seconds'],trace=trace,
            config_sha256=sha(BASE/'training/config.json'),receipt_sha256=sha(BASE/'training/receipt.json'),
            checkpoints={k:public_pin(v) for k,v in training['checkpoints'].items()},
            source_pins=[public_pin(p) for p in cfg['source_pins']],versions=cfg['versions']),
        cache=dict(status=cache['status'],counts=cache['counts'],bytes=cache['cache_tensor_bytes'],
            elapsed_seconds=cache['elapsed_seconds'],receipt_sha256=sha(BASE/'cache/receipt.json')),
        reference_revision=dict(eligible_ids=selection['eligible_ids'],excluded_ids=selection['excluded_ids'],
            selection_sha256=sha(BASE/'reference_selection.json'),manifest_sha256=sha(BASE/'manifest.json'),
            reason='Same-person encoder eligibility and inherited protected/duplicate exclusions before optimization; test data unchanged',
            manifest_records=[dict(identity=c['identity'],role=c['role'],
                target={k:c['target'][k] for k in ['hq_id','original_file','identity','split','source_sha256']},
                references=[{k:r[k] for k in ['hq_id','original_file','identity','split','source_sha256']} for r in c['references']])
                for c in read(BASE/'manifest.json')['cases']]),
        previous_input_cache=dict(status=read(ROOT/'outputs/masked_sdxl_compatibility_pilot_v1/cache/receipt.json')['status'],
            failure=read(ROOT/'outputs/masked_sdxl_compatibility_pilot_v1/cache/receipt.json')['failure'],training_updates=0))
    zero_dir=ROOT/'outputs/masked_sdxl_zero_enabled_replay_v1'
    if (zero_dir/'receipt.json').is_file():
        zero=read(zero_dir/'receipt.json')
        assert zero['status']=='passed' and zero['scorer_enabled'] is True and zero['trained_checkpoint_used'] is False
        assert zero['protocol_sha256']==sha(ROOT/'research/MASKED_SDXL_ZERO_STATE_REPLAY_PREFLIGHT_V1.md')
        assert zero['script_sha256']==sha(ROOT/'scripts/check_masked_sdxl_zero_enabled_replay_v1.py')
        for suffix,proof in zero['checks'].items():
            assert proof['byte_exact'] and proof['pixel_exact'] and proof['new_sha256']==proof['control_sha256']
            assert sha(zero_dir/('263_initial_enabled'+suffix))==proof['new_sha256']
        report['enabled_zero_state_replay']=dict(status=zero['status'],checks=zero['checks'],elapsed_seconds=zero['elapsed_seconds'],
            receipt_sha256=sha(zero_dir/'receipt.json'),protocol_sha256=zero['protocol_sha256'],script_sha256=zero['script_sha256'])
    if (EVAL/'paired_results.json').is_file():
        summary=read(EVAL/'summary.json');paired=read(EVAL/'paired_results.json');generation=read(EVAL/'generation_receipt.json')
        assert paired['summary_sha256']==sha(EVAL/'summary.json') and summary['config_sha256']==sha(EVAL/'config.json')
        assert len(summary['rows'])==9
        for row in summary['rows']:
            if row['status']=='complete':
                assert sha(EVAL/row['file'])==row['sha256'] and sha((EVAL/row['file']).with_name(Path(row['file']).stem+'_raw.png'))==row['raw_sha256']
        report['evaluation']=dict(generation_status=generation['status'],candidate_attempts=generation['candidate_attempts'],
            numerical_gate=paired['numerical_gate'],visual_review_passed=paired['visual_review_passed'],
            failed_gates=paired['failed_gates'],comparisons=paired['comparisons'],
            neutral_replay_passed=paired['neutral_replay_passed'],matched_region_supports=paired['matched_region_supports'],
            coverage=paired['coverage'],summary_sha256=sha(EVAL/'summary.json'),paired_results_sha256=sha(EVAL/'paired_results.json'),
            records=[{k:row[k] for k in ['identity','condition','location','seed','status','sha256','raw_sha256','metrics','regions','target_detections','output_detections'] if k in row} for row in summary['rows']],
            website_promotion=False,replication_completed=False)
    out=ROOT/'research/masked_sdxl_compatibility_pilot_v2_evidence.json'
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(status='verified',training_updates=64,export=str(out),evaluation_included='evaluation' in report)),flush=True)

if __name__=='__main__':main()
