"""Export scientific summary without publishing face photographs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def main():
    datasets=[json.loads((ROOT/f'research/extended_evaluation_{s}_v1.json').read_text(encoding='utf-8')) for s in ['original','count']]
    rows=[r for d in datasets for r in d['rows'] if r['scale']==.8 and r['strength']==.99 and r['compositor']=='poisson']
    figure,axes=plt.subplots(1,3,figsize=(12,4.6))
    for axis,metric,label in zip(axes,['facenet_cosine','niqe','brisque'],['FaceNet cosine (higher better)','NIQE (lower better)','BRISQUE (lower better)']):
        values={n:{r['identity']:r[metric] for r in rows if r['reference_count']==n} for n in [1,2,4]}
        ids=sorted(i for i in values[1] if all(values[n][i] is not None for n in values))
        matrix=np.array([[values[n][i] for n in [1,2,4]] for i in ids])
        for series in matrix:
            axis.plot([1,2,4],series,color='#718096',alpha=.3,lw=.9)
        rng=np.random.default_rng(20260914)
        boot=matrix[rng.integers(0,len(ids),(10000,len(ids)))].mean(1)
        means=matrix.mean(0)
        intervals=np.quantile(boot,[.025,.975],axis=0)
        axis.errorbar([1,2,4],means,yerr=[means-intervals[0],intervals[1]-means],fmt='o-',color='#135b8d',capsize=4,lw=2)
        axis.set(xlabel='Number of references',ylabel=label,xticks=[1,2,4],title=f'{len(ids)} jointly scorable identities')
        axis.spines[['top','right']].set_visible(False)
        axis.grid(axis='y',alpha=.15)
    figure.suptitle('Fixed-subset reference-count ablation · development only',fontsize=13)
    figure.text(.5,.015,'Strength 0.99, seed 17, Poisson composition. Thin lines: identities; bars: descriptive 95% identity bootstrap intervals.',ha='center',fontsize=8)
    figure.tight_layout(rect=(0,.05,1,.94))
    destination=ROOT/'research/figures'
    destination.mkdir(exist_ok=True)
    for extension in ['png','pdf']:
        figure.savefig(destination/f'reference_count_ablation_v1.{extension}',dpi=200)
    plt.close(figure)

if __name__=='__main__':
    main()
