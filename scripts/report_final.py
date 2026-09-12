"""Regenerate final tables, paired intervals and scientific figures from raw measurements."""
import csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];METRICS=['full_face_lpips','hole_mae','visible_mae']
def read(path):
    rows=list(csv.DictReader(path.open()))
    assert all(np.isfinite(float(r[m])) for r in rows for m in METRICS)
    return rows
def means(rows):return {m:float(np.mean([float(r[m]) for r in rows])) for m in METRICS}
def paired(a,b,key,unit):
    aa={key(r):r for r in a};bb={key(r):r for r in b};assert len(aa)==len(a) and len(bb)==len(b) and aa.keys()==bb.keys()
    units=sorted({unit(r) for r in a});rng=np.random.default_rng(314159);resample=rng.integers(0,len(units),(2000,len(units)));report={}
    for m in METRICS:
        delta=np.array([np.mean([float(aa[k][m])-float(bb[k][m]) for k in aa if unit(aa[k])==u]) for u in units])
        report[m]={'mean':float(delta.mean()),'bootstrap_95_interval':np.percentile(delta[resample].mean(1),[2.5,97.5]).tolist()}
    return report
def table(rows,labels):
    lines=['| '+' | '.join(labels+['LPIPS','Hole MAE','Visible MAE'])+' |','|'+'|'.join(['---']*len(labels)+['---:']*3)+'|']
    for row in rows:lines.append('| '+' | '.join([str(row[k]) for k in labels]+[f'{row[m]:.6f}' for m in METRICS])+' |')
    return '\n'.join(lines)

def main():
    learned=read(ROOT/'outputs/refiner_evaluation/metrics.csv');fixed=read(ROOT/'outputs/area_matched_v3_margin12_evaluation/metrics.csv');test=read(ROOT/'outputs/object_test/metrics.csv')
    assert len(learned)==13824 and len(fixed)==5184 and len(test)==5632
    summaries=[];seed_summaries=[]
    for backbone in ['lama','resshift']:
        for variant in ['generic','preservation_weighted']:
            rows=[r for r in learned if r['backbone']==backbone and r['variant']==variant]
            summaries.append({'backbone':backbone,'method':variant,'seeds':3 if backbone=='lama' else 1,**means(rows)})
            for seed in sorted({r['training_seed'] for r in rows}):seed_summaries.append({'backbone':backbone,'method':variant,'seed':seed,**means([r for r in rows if r['training_seed']==seed])})
    for b,r,w in [('lama','8','0'),('resshift','8','0'),('resshift','12','4')]:
        summaries.append({'backbone':b,'method':f'dilation {r}, feather {w}','seeds':1,**means([x for x in fixed if (x['backbone'],x['radius'],x['feather'])==(b,r,w)])})
    dev_deltas={}
    for backbone in ['lama','resshift']:
        aa=[r for r in learned if r['backbone']==backbone and r['variant']=='preservation_weighted'];bb=[r for r in learned if r['backbone']==backbone and r['variant']=='generic']
        dev_deltas[backbone]=paired(aa,bb,lambda r:(r['identity'],r['location'],r['missing_pixels'],r['condition'],r['training_seed']),lambda r:r['identity'])
    test_summary=[];test_deltas={};accurate=[]
    for dataset in ['celebahq','lapa']:
        for backbone in ['lama','resshift']:
            subset=[r for r in test if r['dataset']==dataset and r['backbone']==backbone]
            for method in sorted({r['method'] for r in subset}):
                rows=[r for r in subset if r['method']==method]
                test_summary.append({'dataset':dataset,'backbone':backbone,'method':method,**means(rows)})
                accurate.append({'dataset':dataset,'backbone':backbone,'method':method,**means([r for r in rows if r['condition']=='accurate'])})
            for reference in ['generic','supplied','dilate8']:
                test_deltas[f'{dataset}/{backbone}/weighted-minus-{reference}']=paired([r for r in subset if r['method']=='preservation_weighted'],[r for r in subset if r['method']==reference],lambda r:(r['case_id'],r['condition']),lambda r:r['unit'])
    losses=[]
    for p in sorted((ROOT/'outputs/learned_refiner').glob('*/training.json')):
        r=json.loads(p.read_text());losses.append({k:r[k] for k in ['variant','seed','parameters','steps','seconds_this_session','peak_allocated_MiB']})
    result={'development':summaries,'development_by_seed':seed_summaries,'development_weighted_minus_generic':dev_deltas,'object_test':test_summary,'object_test_accurate_masks':accurate,'object_test_deltas':test_deltas,'training_compute':losses,'raw_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'outputs/refiner_evaluation/metrics.csv',ROOT/'outputs/object_test/metrics.csv',ROOT/'outputs/area_matched_v3_margin12_evaluation/metrics.csv']},'limits':'Bootstrap conditions on fitted seeds and fixed 12 object assets. LaPa has no verified identity grouping. Exploratory comparisons, no multiple-comparison correction, practical-effect margin, or superiority declaration.'}
    strata=defaultdict(list)
    fields=['dataset','backbone','method','condition','category']
    for row in test:strata[tuple(row[k] for k in fields)].append(row)
    with (ROOT/'outputs/object_test/strata.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields+['images']+METRICS);writer.writeheader()
        for key,rows in sorted(strata.items()):writer.writerow({**dict(zip(fields,key)),'images':len(rows),**means(rows)})
    (ROOT/'research/final_results.json').write_text(json.dumps(result,indent=2))
    lines=['# Completed local experiments','','These tables are generated from raw metrics; lower is better for all three measures. Visible MAE measures change to genuinely unoccluded pixels. It is not an identity-recognition score.','','## Controlled development assessment','','48 previously used development identity labels, 432 location/area cases, four supplied-mask conditions. Learned LaMa results average three matched seeds; ResShift learned results use predeclared seed 17 only. Fixed controls use one deterministic LaMa run or one diffusion seed per case.','',table(summaries,['backbone','method','seeds']),'','### Seed-level results','',table(seed_summaries,['backbone','method','seed']),'','### Weighted minus generic, paired development changes','','Intervals are percentile bootstrap intervals over 48 identity labels (2,000 replicates), keeping masks, locations and training seeds together. They condition on these three fitted seeds rather than estimating population-wide training uncertainty.','']
    for b,dd in dev_deltas.items():
        for m,d in dd.items():lines.append(f"- {b}, {m}: {d['mean']:+.6f}, 95% interval [{d['bootstrap_95_interval'][0]:+.6f}, {d['bootstrap_95_interval'][1]:+.6f}].")
    lines+=['','## Frozen object-composite test','','64 HQ test identity labels and 64 LaPa test images; four error conditions; 12 fixed, photographed COCO object cutouts from six categories. These assets were absent from refiner training. This is a synthetic paired composite test, not real occluded-face capture. Its 5,632 rows include unchanged-input controls. LaPa uses a ground-truth-landmark-defined crop for consistent face framing, unavailable in an unconstrained deployment; cross-source numbers also reflect this preprocessing difference.','',table(test_summary,['dataset','backbone','method']),'','### Accurate-mask stratum','',table(accurate,['dataset','backbone','method']),'','### Weighted minus generic on the new test','','Bootstrap over 64 HQ identity labels or 64 LaPa images, conditional on the fixed assets. LaPa identity independence is unresolved; intervals may understate uncertainty if identities repeat.','']
    for key,dd in test_deltas.items():
        if not key.endswith('generic'):continue
        for m,d in dd.items():lines.append(f"- {key}, {m}: {d['mean']:+.6f}, 95% interval [{d['bootstrap_95_interval'][0]:+.6f}, {d['bootstrap_95_interval'][1]:+.6f}].")
    lines+=['','## Compute and reproducibility','',f"Six 119,057-parameter refiner runs, 1,500 steps and batch size 16 each: {sum(r['seconds_this_session'] for r in losses):.1f} seconds recorded training-loop time in these sessions; maximum allocated memory {max(r['peak_allocated_MiB'] for r in losses):.1f} MiB. Cache preparation, dataset audits, downloads, checkpoint startup, and evaluation are excluded. This is not total project energy or wall time.",'','No generative backbone was retrained. Checkpoints, reviewed manifests, threshold selections, frozen test protocol and raw hashes are recorded locally. Training losses were still improving at the last budgeted step, so convergence is not established.','','## Scientific limits','','The weighted loss is a standard control, not an established novel contribution. No nearest-method reproduction (VCNet/OSOR), soft-alpha learned control, identity fidelity evaluation, blinded human study, real paired capture, or comprehensive pretraining-membership audit is provided. Final test inference is now complete; these cases must not be treated as unused for subsequent method development. The engineering pipeline and scoped experiments are complete. An IEEE submission is not ready merely because this package runs.']
    (ROOT/'research/FINAL_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    qa=json.loads((ROOT/'research/object_protocol_verification.json').read_text())
    with (ROOT/'research/FINAL_RESULTS.md').open('a',encoding='utf-8') as f:
        f.write(f"\nObject-protocol QA verified all {qa['cases']} cases: true masks are nonempty and do not touch the image boundary; mask polarity and unchanged visible pixels pass. Missing area ranges from {100*qa['minimum_missing_fraction']:.2f}% to {100*qa['maximum_missing_fraction']:.2f}%; {qa['padded_crops']} LaPa crops require black padding outside the source image. Location, area and texture differ from development, so differences cannot be attributed to object appearance alone. Category/condition breakdowns, including sample counts, are in outputs/object_test/strata.csv.\n")
        efficiency=json.loads((ROOT/'research/inference_efficiency.json').read_text())
        f.write('\n## Warm inference on this laptop\n\n'+efficiency['scope']+'\n\n| Backbone | Refiner | Median ms | P95 ms | Allocated MiB |\n|---|---|---:|---:|---:|\n')
        for r in efficiency['results']:f.write(f"| {r['backbone']} | {r['refinement']} | {r['median_ms']:.2f} | {r['p95_ms']:.2f} | {r['peak_allocated_MiB']:.1f} |\n")
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,dataset in zip(axes,['celebahq','lapa']):
        for r in test_summary:
            if r['dataset']!=dataset:continue
            ax.scatter(r['visible_mae'],r['full_face_lpips'],marker='o' if r['backbone']=='lama' else '^',s=45,label=f"{r['backbone']}: {r['method']}")
        ax.set(title=f'{dataset}: object-composite test',xlabel='Visible-region MAE',ylabel='Full-image LPIPS');ax.grid(alpha=.2)
    axes[1].legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=7)
    fig.savefig(ROOT/'outputs/object_test/tradeoff.png',dpi=160);fig.savefig(ROOT/'outputs/object_test/tradeoff.svg');plt.close(fig)
    print('Generated final report, machine-readable tables and plots')
if __name__=='__main__':main()
