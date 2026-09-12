"""Compare frozen validation protocols; never select controls on assessment rows."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    names={'lama':'benchmark_v2','resshift':'benchmark_v2_resshift'}
    reports={}; data={}; cases={}
    for model,run in names.items():
        reports[model]=json.loads((ROOT/'research'/f'{run}_results.json').read_text())
        cases[model]=json.loads((ROOT/'outputs'/run/'cases.json').read_text())
        data[model]=list(csv.DictReader((ROOT/'outputs'/run/'metrics.csv').open()))
        assert len(data[model])==7200
    assert cases['lama']==cases['resshift'], 'Case metadata mismatch'
    assert reports['lama']['config_sha256']==reports['resshift']['config_sha256']
    assert reports['lama']['manifest_sha256']==reports['resshift']['manifest_sha256']
    # Compare saved input and mask bytes, not only metadata or seeds.
    checked=0
    for c in cases['lama']:
        folder=f"{c['hq_id']}_{c['family']}"
        for name in ['observed','true_mask','accurate','under','over','shift','mixed_boundary']:
            blobs=[(ROOT/'outputs'/run/'cases'/folder/f'{name}.png').read_bytes() for run in names.values()]
            assert blobs[0]==blobs[1], f'Input mismatch: {folder}/{name}'
            checked+=1
    chosen={m:str(r['selected_radius']) for m,r in reports.items()}
    keys=lambda r:(r['identity'],r['family'],r['condition'])
    selected={m:{keys(r):r for r in rows if r['partition']=='assessment' and r['radius']==chosen[m]} for m,rows in data.items()}
    assert selected['lama'].keys()==selected['resshift'].keys()
    identities=sorted({k[0] for k in selected['lama']})
    rng=np.random.default_rng(20260911)
    boot=rng.integers(0,len(identities),size=(2000,len(identities)))
    differences={}
    for metric in ['full_face_lpips','hole_mae','visible_mae']:
        delta=np.array([np.mean([float(selected['resshift'][k][metric])-float(selected['lama'][k][metric]) for k in selected['lama'] if k[0]==identity]) for identity in identities])
        differences[metric]={'mean':float(delta.mean()),'identity_bootstrap_95_interval':np.percentile(delta[boot].mean(axis=1),[2.5,97.5]).tolist()}
    fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
    lines=['# Matched backbone development comparison','','Both models use identical images, synthetic corruptions, supplied masks, and identity partitions. Each radius is chosen on tuning LPIPS independently; assessment identities are not used for selection.','','| Backbone | Selected radius | Assessment LPIPS | Hole MAE | Visible MAE |','|---|---:|---:|---:|---:|']
    for model,r in reports.items():
        summary=[x for x in r['summary'] if x['partition']=='assessment' and x['radius']!='oracle']
        chosen_row=next(x for x in summary if str(x['radius'])==chosen[model])
        lines.append(f"| {model} | {chosen[model]} | {chosen_row['full_face_lpips']:.5f} | {chosen_row['hole_mae']:.5f} | {chosen_row['visible_mae']:.5f} |")
        ax.plot([x['visible_mae'] for x in summary],[x['full_face_lpips'] for x in summary],marker='o',label=model)
        for x in summary: ax.annotate(str(x['radius']),(x['visible_mae'],x['full_face_lpips']),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Mean change on genuinely visible pixels',ylabel='Full-face LPIPS',title='Matched validation: reconstruction versus preservation')
    ax.legend();ax.grid(alpha=.2)
    out=ROOT/'outputs/backbone_comparison';out.mkdir(exist_ok=True)
    fig.savefig(out/'tradeoff.png',dpi=180);fig.savefig(out/'tradeoff.svg');plt.close(fig)
    lines+=['','## Paired assessment differences','','ResShift minus LaMa, using each model’s tuning-selected radius. Lower is better for all metrics. Intervals resample 48 identities, with 2,000 replicates.']
    for metric,d in differences.items(): lines.append(f"- {metric}: {d['mean']:.6f}; 95% interval [{d['identity_bootstrap_95_interval'][0]:.6f}, {d['identity_bootstrap_95_interval'][1]:.6f}].")
    lines+=['','## Limits and next experiment','','This compares these checkpoints under this development protocol, not intrinsic architectures or published paper performance. Training data and pretraining exposure differ and remain unresolved. One diffusion seed is fixed per image/family and shared across its controls; sampling variation is not estimated. Exact masks are privileged diagnostics. The synthetic mask families are not area-matched, and final test images remain unused for inference.','','Radius 8 is the boundary of the current search. Extend the tuning sweep and add feathered compositing before claiming an optimized correction control. A learned method must improve the reconstruction–preservation curve beyond these controls. No novel method has been evaluated.']
    lines+=['','## Qualitative inspection','','The preselected first assessment identity in outputs/benchmark_v2_resshift/preview.png shows residual synthetic corruption with uncorrected masks. In the eye-region mixed-boundary case, radius 4 produces glasses absent from the target. The exact-mask diagnostic looks more plausible but does not recover every hidden detail. These examples illustrate why aggregate perceptual gains cannot establish identity fidelity. The preview is a diagnostic example, not a measured failure rate.']
    (ROOT/'research/MILESTONE_03.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    result={'matched_input_files':checked,'cases':len(cases['lama']),'assessment_identities':len(identities),'selected_radii':chosen,'resshift_minus_lama':differences,'metrics_sha256':{m:hashlib.sha256((ROOT/'outputs'/run/'metrics.csv').read_bytes()).hexdigest() for m,run in names.items()}}
    (ROOT/'research/backbone_comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
