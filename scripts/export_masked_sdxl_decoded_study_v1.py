"""Verify completed decoded-loss evidence and export only scalar metadata/hashes."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/masked_sdxl_decoded_continuation_v1'
EVAL=ROOT/'outputs/masked_sdxl_decoded_continuation_eval_v1'
IDS=['6792','2250','2083','4164','674','476','1281','8748']


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def check(p):assert sha(p['path'])==p['sha256'],p['path']


def public_pin(p):
    path=Path(p['path']).resolve()
    return dict(file=str(path.relative_to(ROOT)).replace('\\','/') if path.is_relative_to(ROOT) else path.name,sha256=p['sha256'])


def main():
    probe_path=ROOT/'outputs/masked_sdxl_decoded_loss_probe_v1/receipt.json';probe=read(probe_path)
    assert probe['status']=='passed' and probe['parameter_update'] and probe['backend_flags_restored']
    assert probe['frozen_unet_gradients'] is False and probe['frozen_vae_gradients'] is False
    for p in probe['source_pins']:check(p)
    check(probe['starting_checkpoint'])
    document=dict(scope='Matched decoded missing-region loss development study; no accuracy percentage or generalization claim',
        probe={k:v for k,v in probe.items() if k not in ['source_pins','starting_checkpoint','cache_receipt','prior_training_receipt']},
        training={})
    document['probe']['source_pins']=[public_pin(x) for x in probe['source_pins']]
    document['probe']['checkpoint']=public_pin(probe['starting_checkpoint'])
    document['probe']['receipt_sha256']=sha(probe_path)
    configs={}
    for arm in ['epsilon','rgb']:
        folder=BASE/arm/'training';cfg=read(folder/'config.json');receipt=read(folder/'receipt.json');trace=read(folder/'trace.json')
        assert receipt['status']=='complete' and receipt['completed_updates']==receipt['finite_gradient_updates']==len(trace)==16
        assert cfg['training_ids']==IDS and cfg['trainable_parameters']==385 and receipt['modified_processors']==70
        assert cfg['protocol_sha256']==receipt['protocol_sha256']==sha(cfg['protocol_path'])
        assert receipt['config_sha256']==sha(folder/'config.json') and receipt['trace_sha256']==sha(folder/'trace.json')
        assert receipt['frozen_weights_have_gradients'] is False and receipt['backend_flags_restored'] and receipt['parameter_update']
        assert receipt['evaluation_photographs_decoded']==receipt['evaluation_annotations_decoded']==0
        assert cfg['optimizer']['reset'] and cfg['rgb_weight']==(.1 if arm=='rgb' else 0.)
        assert receipt['checkpoints']['initial']['sha256']==cfg['starting_checkpoint']['sha256']=='0f6ad433ce03a998c4fe4375e309fe79a85bc0d945448774fb603edbeace84ba'
        for p in cfg['source_pins']:check(p)
        for p in receipt['checkpoints'].values():check(p)
        check(cfg['cache_receipt']);check(cfg['probe_receipt']);check(cfg['prior_training_receipt'])
        for i,row in enumerate(trace):
            assert row['update']==i+1 and row['identity']==IDS[i%8] and row['timestep']==[100,300][i//8] and row['noise_seed']==20261011+i
            assert all(math.isfinite(row[k]) for k in ['loss','epsilon_objective','gradient_norm','full_epsilon_mse','hole_epsilon_mse','step_seconds']) and row['gradient_norm']>0
            assert abs(row['epsilon_objective']-(.25*row['full_epsilon_mse']+.75*row['hole_epsilon_mse']))<1e-6
            rgb=row['masked_rgb_l1']
            assert (rgb is None if arm=='epsilon' else math.isfinite(rgb))
            assert abs(row['loss']-(row['epsilon_objective']+cfg['rgb_weight']*(rgb or 0)))<1e-6
        configs[arm]=cfg
        document['training'][arm]=dict(status=receipt['status'],identities=IDS,updates=16,updates_per_person=2,
            parameters=385,trace=trace,elapsed_seconds=receipt['elapsed_seconds'],process_memory=receipt['process_memory'],
            peak_allocated_bytes=receipt['peak_allocated_bytes'],peak_reserved_bytes=receipt['peak_reserved_bytes'],
            frozen_weights_have_gradients=False,frozen_vae_have_gradients=receipt['frozen_vae_have_gradients'],
            backend_flags_restored=True,decoded_training_photographs=receipt['decoded_training_photographs'],
            config_sha256=sha(folder/'config.json'),receipt_sha256=sha(folder/'receipt.json'),trace_sha256=sha(folder/'trace.json'),
            checkpoints={k:public_pin(v) for k,v in receipt['checkpoints'].items()},source_pins=[public_pin(x) for x in cfg['source_pins']])
    for key in ['training_ids','seed','timesteps','optimizer','training_resolution','trainable_parameters','cache_receipt','starting_checkpoint']:
        assert configs['epsilon'][key]==configs['rgb'][key],key
    traces={arm:document['training'][arm]['trace'] for arm in ['epsilon','rgb']}
    initial_forward_keys=['full_epsilon_mse','hole_epsilon_mse','epsilon_objective']
    assert all(traces['epsilon'][0][key]==traces['rgb'][0][key] for key in initial_forward_keys)
    document['matching_checks']=dict(initial_checkpoint_bytes_equal=True,optimizer_reset_equal=True,
        first_forward_epsilon_exact=True,
        first_forward={key:traces['epsilon'][0][key] for key in initial_forward_keys})
    summary=read(EVAL/'summary.json');paired=read(EVAL/'paired_results.json');generation=read(EVAL/'generation_receipt.json')
    assert len(summary['rows'])==15 and paired['summary_sha256']==sha(EVAL/'summary.json')
    assert summary['config_sha256']==sha(EVAL/'config.json') and generation['neutral_replay_passed']
    for row in summary['rows']:
        if row['status']=='complete':
            output=EVAL/row['file'];assert sha(output)==row['sha256'] and sha(output.with_name(output.stem+'_raw.png'))==row['raw_sha256']
    document['evaluation']=dict(summary_sha256=sha(EVAL/'summary.json'),paired_results_sha256=sha(EVAL/'paired_results.json'),
        generation_status=generation['status'],candidate_attempts=generation['candidate_attempts'],reused_controls=9,
        numerical_gate=paired['numerical_gate'],visual_review_passed=paired['visual_review_passed'],
        failed_gates=paired['failed_gates'],comparisons=paired['comparisons'],coverage=paired['coverage'],
        matched_region_supports=paired['matched_region_supports'],neutral_replay_passed=paired['neutral_replay_passed'],
        records=[{k:r[k] for k in ['identity','condition','location','seed','status','reused','sha256','raw_sha256','metrics','regions','target_detections','output_detections','error'] if k in r} for r in summary['rows']],
        website_promotion=False,replication_completed=False)
    metric_keys=['psnr','ssim','lpips','hole_psnr','hole_ssim','hole_lpips','eye_lpips','facenet_cosine','arcface_conditioning_cosine']
    means={}
    for arm in ['control3','local','prior','epsilon','rgb']:
        rows=[row for row in summary['rows'] if row['condition']==arm]
        assert len(rows)==3 and {row['identity'] for row in rows}=={'263','6681','10149'}
        means[arm]={}
        for key in metric_keys:
            values=[row.get('metrics',{}).get(key) for row in rows]
            values=[value for value in values if isinstance(value,(int,float)) and math.isfinite(value)]
            means[arm][key]=dict(mean=sum(values)/len(values) if values else None,valid=len(values),expected=3)
    for arm,comparison in paired['comparisons'].items():
        for key,value in comparison['metrics'].items():
            # Full support is required by the experiment gate; compare independently calculated means.
            if value['n']==3:
                assert abs(value['control']-means[arm][key]['mean'])<1e-12
                assert abs(value['candidate']-means['rgb'][key]['mean'])<1e-12
    document['evaluation']['metric_means']=means
    out=ROOT/'research/masked_sdxl_decoded_continuation_v1_evidence.json'
    out.write_text(json.dumps(document,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(status='verified',updates_per_arm=16,score_records=15,numerical_gate=paired['numerical_gate'])),flush=True)


if __name__=='__main__':main()
