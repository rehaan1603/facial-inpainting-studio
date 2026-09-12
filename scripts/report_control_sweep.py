import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--backbone',choices=['lama','resshift'],default='lama');args=parser.parse_args()
    model=args.backbone;out=ROOT/'outputs'/f'control_sweep_{model}'
    report=json.loads((ROOT/'research'/f'control_sweep_{model}.json').read_text())
    rows=list(csv.DictReader((out/'metrics.csv').open()))
    oldrun='benchmark_v2' if model=='lama' else 'benchmark_v2_resshift'
    old=list(csv.DictReader((ROOT/'outputs'/oldrun/'metrics.csv').open()))
    new=[r for r in rows if r['partition']=='assessment']
    baseline=[r for r in old if r['partition']=='assessment' and r['radius']=='8']
    key=lambda r:(r['identity'],r['family'],r['condition'])
    newmap={key(r):r for r in new};oldmap={key(r):r for r in baseline}
    assert len(newmap)==len(new)==720 and newmap.keys()==oldmap.keys()
    assert sum(r['partition']=='tuning' for r in rows)==20160
    ids=sorted({k[0] for k in newmap});rng=np.random.default_rng(20260911)
    resamples=rng.integers(0,len(ids),(2000,len(ids)))
    delta={}
    metrics=['full_face_lpips','hole_mae','visible_mae']
    for metric in metrics:
        values=np.array([np.mean([float(newmap[k][metric])-float(oldmap[k][metric]) for k in newmap if k[0]==identity]) for identity in ids])
        delta[metric]={'mean':float(values.mean()),'bootstrap_95_interval':np.percentile(values[resamples].mean(axis=1),[2.5,97.5]).tolist()}
    selection=json.loads((out/'selection.json').read_text())
    grid=selection['tuning_means'];radii=sorted({int(k.split(',')[0]) for k in grid});widths=sorted({int(k.split(',')[1]) for k in grid})
    fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
    for width in widths:ax.plot(radii,[grid[f'{r},{width}'] for r in radii],marker='o',label=f'Feather {width}px')
    ax.set(xlabel='Dilation radius (pixels)',ylabel='Mean tuning LPIPS',title=f'{model}: tuning simple correction controls')
    ax.legend();ax.grid(alpha=.2);fig.savefig(out/'tuning.png',dpi=180);fig.savefig(out/'tuning.svg');plt.close(fig)
    oldmean={m:float(np.mean([float(r[m]) for r in baseline])) for m in metrics}
    lines=[f'# Expanded {model} controls','','All 28 radius/blending combinations were evaluated on 48 tuning identities. The minimum tuning LPIPS determined one global combination, frozen before assessment inference. The existing 48 development assessment identities were reused; final test remains unused.','',
        f"Selected radius: **{report['selected_radius']} pixels**. Selected inward feather width: **{report['selected_feather']} pixels**.",'',
        '| Assessment metric | Previous radius 8, hard blend | New selected control |','|---|---:|---:|']
    for m in metrics:lines.append(f"| {m} | {oldmean[m]:.6f} | {report['assessment'][m]:.6f} |")
    lines+=['','## Paired changes','','New minus previous radius-8 control; lower is better. Confidence intervals resample 48 identities, 2,000 times.']
    if report['selected_radius']==8 and report['selected_feather']==0:
        lines+=['','The selected control is unchanged. Zero paired differences and zero-width intervals here mean the same deterministic outputs were reproduced; they do not establish absence of uncertainty for generalization or other controls.']
    lines.append('')
    for m,d in delta.items():lines.append(f"- {m}: {d['mean']:.6f}, 95% interval [{d['bootstrap_95_interval'][0]:.6f}, {d['bootstrap_95_interval'][1]:.6f}].")
    lines+=['','## Condition breakdown','','| Condition | Old LPIPS | New LPIPS | Old visible MAE | New visible MAE |','|---|---:|---:|---:|---:|']
    for condition in ['accurate','under','over','shift','mixed_boundary']:
        a=[r for r in baseline if r['condition']==condition];b=[r for r in new if r['condition']==condition]
        mean=lambda rr,m:float(np.mean([float(r[m]) for r in rr]))
        lines.append(f"| {condition} | {mean(a,'full_face_lpips'):.5f} | {mean(b,'full_face_lpips'):.5f} | {mean(a,'visible_mae'):.5f} | {mean(b,'visible_mae'):.5f} |")
    lines+=['','## Interpretation limits','','This is a simple control, not a learned or novel method. Selection optimizes full-face LPIPS and does not enforce a visible-pixel preservation tolerance. Assess the regional errors and condition breakdown before calling the result better overall. Feathering uses only the expanded supplied mask; clean target parsing never defines inference blending.','','This remains repeated development assessment, with synthetic textures, unmatched mask areas, and unresolved checkpoint pretraining exposure. The unchanged final test will be required after design decisions are locked. Diffusion sampling variation is not estimated.']
    if report['selected_radius']==max(radii):lines+=['','The selected radius is at the upper search boundary; an optimum has not been established.']
    other='resshift' if model=='lama' else 'lama'
    next_step=('Both backbone control searches have results. Add area-matched corruptions and real occluder assets before defining a learned contribution.' if (ROOT/'research'/f'control_sweep_{other}.json').exists() else 'Run the same control search on the other backbone, then add area-matched corruptions and real occluder assets before defining a learned contribution.')
    lines+=['','## Next step','',next_step]
    (ROOT/'research'/f'CONTROL_RESULTS_{model.upper()}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'research'/f'control_comparison_{model}.json').write_text(json.dumps({'selected':selection,'assessment_deltas':delta,'previous_assessment':oldmean,'new_assessment':report['assessment']},indent=2))
    print(json.dumps({'selected':(report['selected_radius'],report['selected_feather']),'previous':oldmean,'new':report['assessment'],'deltas':delta},indent=2))


if __name__=='__main__':main()
