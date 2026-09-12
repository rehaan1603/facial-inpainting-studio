"""Report training-budget and compositing contrasts on the same development cases."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_final import means,paired,table
ROOT=Path(__file__).resolve().parents[1]
def main():
    path=ROOT/'outputs/extension_evaluation_v1/metrics.csv';rows=list(csv.DictReader(path.open()));assert len(rows)==27648
    initial=list(csv.DictReader((ROOT/'outputs/refiner_evaluation/metrics.csv').open()));summaries=[];contrasts={};curves=[]
    key=lambda r:(r['identity'],r['location'],r['missing_pixels'],r['condition'],r['training_seed']);unit=lambda r:r['identity']
    for backbone in ['lama','resshift']:
        for variant in ['generic','preservation_weighted']:
            before=[r for r in initial if r['backbone']==backbone and r['variant']==variant]
            hard=[r for r in rows if r['backbone']==backbone and r['variant']==variant and r['compositor']=='hard']
            soft=[r for r in rows if r['backbone']==backbone and r['variant']==variant and r['compositor']=='probability_alpha']
            for budget,compositor,rr in [(1500,'hard',before),(6000,'hard',hard),(6000,'probability_alpha',soft)]:summaries.append({'backbone':backbone,'variant':variant,'budget':budget,'compositor':compositor,**means(rr)})
            contrasts[f'{backbone}/{variant}/6000-minus-1500-hard']=paired(hard,before,key,unit)
            contrasts[f'{backbone}/{variant}/soft-minus-hard-6000']=paired(soft,hard,key,unit)
        weighted=[r for r in rows if r['backbone']==backbone and r['variant']=='preservation_weighted' and r['compositor']=='hard'];generic=[r for r in rows if r['backbone']==backbone and r['variant']=='generic' and r['compositor']=='hard']
        contrasts[f'{backbone}/weighted-minus-generic-6000']=paired(weighted,generic,key,unit)
    for p in sorted((ROOT/'outputs/refiner_extension_v1').glob('*/training.json')):
        r=json.loads(p.read_text());best_step=min(r['history'],key=lambda h:h['tuning_loss'])['step'];curves.append({'variant':r['variant'],'seed':r['seed'],'best_step':best_step,'initial_tuning_loss':r['history'][5]['tuning_loss'],'best_tuning_loss':r['best_tuning_loss'],'last_three_losses':[h['tuning_loss'] for h in r['history'][-3:]],'extension_seconds':r['extension_seconds']})
    report={'summaries':summaries,'contrasts':contrasts,'training':curves,'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':'Reused development identities only. Three seeds on LaMa, seed17 on ResShift. Bootstrap over 48 identity labels, conditional on fitted seeds. Soft probabilities are segmentation scores, not a separately RGB-trained alpha head. No novel-method claim.'}
    (ROOT/'research/extension_results.json').write_text(json.dumps(report,indent=2))
    lines=['# Training-budget and compositing follow-up','','The six original refiners have now each received 6,000 training updates, four times the first-stage budget. Their initial checkpoints and metrics are preserved. All results below reuse the 48 development assessment identity labels; no test inference was performed in this extension.','','## Reconstruction and preservation','',table(summaries,['backbone','variant','budget','compositor']),'','## Paired changes','','Each contrast is first named setting minus second; lower is better. The 95% intervals use 2,000 paired resamples of identity labels, retaining locations, mask errors and training seeds together. Three seeds are evaluated on LaMa and predefined seed 17 on ResShift; intervals do not estimate uncertainty over all possible training seeds.','']
    for name,metrics in contrasts.items():
        lines.append(f'### {name}')
        lines.append('')
        for metric,d in metrics.items():lines.append(f"- {metric}: {d['mean']:+.6f}, interval [{d['bootstrap_95_interval'][0]:+.6f}, {d['bootstrap_95_interval'][1]:+.6f}].")
        lines.append('')
    lines+=['## Interpretation','','The budget comparison retains the same training and threshold-selection rules, with a new tuning-selected checkpoint and threshold for each extended run. It measures the resulting longer-training pipeline, not a fixed-threshold weight-only intervention. The soft-compositing comparison shares each checkpoint, threshold, candidate and effective mask exactly. It blends using the learned segmentation score; this is a useful simple control, but does not substitute for a reconstruction-trained alpha head or an OSOR reproduction.','','A smaller tuning loss is not proof of convergence or better reconstructed faces. The complete learning histories and reconstruction contrasts above determine which first-stage conclusions survive. Data/identity and pretraining-exposure limitations remain. The original test sample is already observed evidence; future confirmatory testing needs reserved cases and frozen choices.','','## Training records','','| Variant | Seed | Best update | Initial tuning loss | Best tuning loss |','|---|---:|---:|---:|---:|']
    for r in curves:lines.append(f"| {r['variant']} | {r['seed']} | {r['best_step']} | {r['initial_tuning_loss']:.6f} | {r['best_tuning_loss']:.6f} |")
    (ROOT/'research/TRAINING_EXTENSION_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,variant in zip(axes,['generic','preservation_weighted']):
        for seed in [17,29,43]:
            h=json.loads((ROOT/f'outputs/refiner_extension_v1/{variant}_{seed}/training.json').read_text())['history'];ax.plot([x['step'] for x in h],[x['tuning_loss'] for x in h],label=f'Seed {seed}')
        ax.axvline(1500,color='gray',ls='--',lw=1);ax.set(title=variant.replace('_',' '),xlabel='Training updates',ylabel='Variant-specific tuning loss',yscale='log');ax.legend();ax.grid(alpha=.2)
    fig.savefig(ROOT/'outputs/extension_evaluation_v1/learning_curves.png',dpi=180);fig.savefig(ROOT/'outputs/extension_evaluation_v1/learning_curves.svg');plt.close(fig);print('Extension report generated')
if __name__=='__main__':main()
