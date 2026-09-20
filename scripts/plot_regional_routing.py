"""Numerical, photo-free diagnostic plot showing identities rather than inflated row counts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def main():
    rows=json.loads((ROOT/'outputs/regional_routing_v2/evaluation.json').read_text())['rows']
    policies=['concat','equal','identity','quality','mask_aware','regional'];labels=['Original\nconcat','Equal','Identity','Quality','Mask-aware\nglobal','Regional']
    identities=sorted({r['identity'] for r in rows});colors=['#0072B2','#D55E00','#009E73','#CC79A7']
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,metric,title in zip(axes,['facenet_cosine','facenet_gallery_cosine'],['Target similarity','Withheld-gallery similarity']):
        means=[]
        for identity,color in zip(identities,colors):
            series=[]
            for policy in policies:
                values=[r['evaluation'].get('metrics',{}).get(metric) for r in rows if r['identity']==identity and r['policy']==policy]
                values=[v for v in values if v is not None];series.append(np.mean(values) if values else np.nan)
            means.append(series);ax.plot(range(6),series,'o-',color=color,alpha=.8,lw=1,label='Identity '+identity)
        ax.plot(range(6),np.nanmean(means,axis=0),'D--',color='#222222',lw=2,label='Identity mean')
        ax.set_xticks(range(6),labels);ax.set_title(title);ax.set_ylabel('FaceNet cosine similarity (higher is better)');ax.grid(axis='y',alpha=.2)
    axes[1].legend(fontsize=8,loc='best');fig.suptitle('Regional routing: four development identities, three conditions, two seeds\nUntrained mechanism study; no reserved-final data',fontsize=12)
    out=ROOT/'research/figures';out.mkdir(exist_ok=True)
    for suffix in ['png','pdf']:fig.savefig(out/f'regional_routing_v2_per_identity.{suffix}',dpi=180)
    plt.close(fig)

if __name__=='__main__':main()
