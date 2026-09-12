"""Summarize frozen controls with identity-level paired uncertainty."""
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
    out=ROOT/'outputs/area_matched_v3_margin12_evaluation'
    rows=list(csv.DictReader((out/'metrics.csv').open()))
    cases=json.loads((ROOT/'outputs/area_matched_v3_margin12/cases.json').read_text())
    cases=[c for c in cases if c['partition']=='assessment']
    assert len(rows)==len(cases)*12
    key=lambda r:(r['identity'],r['location'],r['missing_pixels'],r['condition'])
    controls=[('lama','8','0'),('resshift','8','0'),('resshift','12','4')]
    sets={control:{key(r):r for r in rows if (r['backbone'],r['radius'],r['feather'])==control} for control in controls}
    assert len({tuple(sorted(s.keys())) for s in sets.values()})==1
    assert all(len(s)==len(cases)*4 for s in sets.values())
    identity_ids=sorted({r['identity'] for r in rows});assert len(identity_ids)==48
    metric_names=['full_face_lpips','hole_mae','visible_mae']
    assert all(np.isfinite(float(r[m])) for r in rows for m in metric_names)
    mean=lambda rr,m:float(np.mean([float(r[m]) for r in rr]))
    summaries=[{'backbone':b,'radius':int(r),'feather':int(w),**{m:mean(list(sets[(b,r,w)].values()),m) for m in metric_names}} for b,r,w in controls]
    rng=np.random.default_rng(20260912);resample=rng.integers(0,48,(2000,48))
    newer=sets[('resshift','12','4')];older=sets[('resshift','8','0')];delta={}
    for metric in metric_names:
        values=np.array([np.mean([float(newer[k][metric])-float(older[k][metric]) for k in newer if k[0]==identity]) for identity in identity_ids])
        delta[metric]={'mean':float(values.mean()),'bootstrap_95_interval':np.percentile(values[resample].mean(axis=1),[2.5,97.5]).tolist()}
    lines=['# Area-matched development benchmark','','The benchmark matches true missing area, false-positive and false-negative mask counts, mask geometry, and synthetic occluder RGB across eye-center, mouth-center, and upper-image locations. All supplied masks have at least 12 pixels of image margin, preventing clipping under the evaluated correction radii.','','## Data and protocol','','- 48 assessment identities; three area levels, three locations, four mask conditions.',f'- {len(cases)} location/area cases and {len(rows)} model/control metric rows.',
        '- Exact areas: 1,966, 3,932 and 6,554 pixels (approximately 3%, 6% and 10% of a 256-square image).',
        '- Error count is round(0.2 × missing pixels). Under masks remove that count; over masks add it; mixed masks do both. Counts match across locations within each error condition, not across different conditions.',
        '- Controls were frozen from v2: LaMa radius 8/hard blend, ResShift radius 8/hard blend and radius 12/feather 4. No v3 tuning.',
        '- Dataset construction also retains 47 tuning identities, but they are not evaluated here. One tuning identity lacks an eye center; all its area groups are excluded and recorded.',
        '- The same 48 development assessment identities were used previously. This is a controlled diagnostic, not a fresh test set.','','## Aggregate assessment','','| Backbone | Radius | Feather | LPIPS | Hole MAE | Visible MAE |','|---|---:|---:|---:|---:|---:|']
    for s in summaries:lines.append(f"| {s['backbone']} | {s['radius']} | {s['feather']} | {s['full_face_lpips']:.5f} | {s['hole_mae']:.5f} | {s['visible_mae']:.5f} |")
    lines+=['','## Paired ResShift control changes','','Radius 12/feather 4 minus radius 8/hard blend. The 95% intervals resample identities, retaining correlated locations, areas and masks together; 2,000 replicates.','']
    for m,d in delta.items():lines.append(f"- {m}: {d['mean']:.6f}, interval [{d['bootstrap_95_interval'][0]:.6f}, {d['bootstrap_95_interval'][1]:.6f}].")
    if all(d['mean']>0 for d in delta.values()):
        lines+=['','The radius-12/feathered setting is worse on all three aggregate metrics here. Its earlier v2 tuning gain does not transfer to this changed protocol. This supports checking robustness across corruption protocols; it does not establish a novel method or prove that area matching alone caused the reversal, because shape and error construction also changed.']
    lines+=['','## Location breakdown','','| Location | Control | LPIPS | Hole MAE | Visible MAE |','|---|---|---:|---:|---:|']
    location_summary=[]
    for location in ['eye_center','mouth_center','upper_image']:
        for control in controls:
            subset=[r for r in sets[control].values() if r['location']==location]
            vals={m:mean(subset,m) for m in metric_names};location_summary.append({'location':location,'control':control,**vals})
            lines.append(f"| {location} | {'/'.join(control)} | {vals['full_face_lpips']:.5f} | {vals['hole_mae']:.5f} | {vals['visible_mae']:.5f} |")
    lines+=['','## Limits','','Locations describe placement centers, not guaranteed full coverage of an anatomical feature. The upper-image location may cover hair, forehead or background. Shapes are translated ellipse-like masks with spatially varying boundary perturbations, not real occluder silhouettes. Equal geometry and texture remove selected confounds; they do not make the surrounding image content identical or establish causal semantic difficulty.','','Checkpoint pretraining exposure remains unresolved; only one diffusion seed per case is used. No identity fidelity claim, trained new method, or publication claim is established. Final test data remains unused for inference.','','## Provenance and next step','','The canonical files are outputs/area_matched_v3_margin12 and outputs/area_matched_v3_margin12_evaluation. The earlier directories without margin12 are superseded engineering artifacts from a boundary-clipping check and must not be used as results.','','Next, add held-out realistic occluder assets and finish the pretraining/near-duplicate audit. Any learned correction must be compared against generic learned refinement and blending controls, with a preset preservation tolerance.']
    (ROOT/'research/MILESTONE_04.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
    for s in summaries:
        ax.scatter(s['visible_mae'],s['full_face_lpips'],s=65,label=f"{s['backbone']} r={s['radius']}, feather={s['feather']}")
    ax.set(xlabel='Mean change on genuinely visible pixels',ylabel='Full-face LPIPS',title='Area-matched development assessment');ax.grid(alpha=.2);ax.legend()
    fig.savefig(out/'tradeoff.png',dpi=180);fig.savefig(out/'tradeoff.svg');plt.close(fig)
    report={'rows':len(rows),'assessment_identities':48,'summaries':summaries,'resshift_control_delta':delta,'by_location':location_summary,'metrics_sha256':hashlib.sha256((out/'metrics.csv').read_bytes()).hexdigest()}
    (ROOT/'research/area_matched_v3_results.json').write_text(json.dumps(report,indent=2));print(json.dumps({'summaries':summaries,'delta':delta},indent=2))


if __name__=='__main__':main()
