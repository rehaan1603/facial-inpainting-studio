"""Report all routing arms and identity-level paired statistics."""
import itertools,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw
from src.research_integrity import ROOT,sha,write_new

def paired(rows,other):
    lookup={(r['case_id'],r['seed'],r['policy']):r for r in rows};deltas={};count=0
    for r in rows:
        if r['policy']!='regional':continue
        b=lookup[(r['case_id'],r['seed'],other)]
        a_value=r['evaluation'].get('metrics',{}).get('facenet_cosine');b_value=b['evaluation'].get('metrics',{}).get('facenet_cosine')
        if a_value is None or b_value is None:continue
        deltas.setdefault(r['identity'],[]).append(a_value-b_value);count+=1
    means=np.array([np.mean(v) for v in deltas.values()])
    if len(means)==0:return {'identities':0,'pairs':0,'mean_delta':None,'p_exact':None,'ci95':None}
    average=float(means.mean());permutations=[abs(float(np.mean(means*np.asarray(signs)))) for signs in itertools.product([-1,1],repeat=len(means))]
    p=float(np.mean(np.asarray(permutations)>=abs(average)-1e-12));rng=np.random.default_rng(20260920)
    boot=rng.choice(means,(10000,len(means)),replace=True).mean(1)
    return {'identities':len(means),'pairs':count,'mean_delta':average,'p_exact':p,'ci95':np.quantile(boot,[.025,.975]).tolist(),
            'identity_deltas':{k:float(np.mean(v)) for k,v in deltas.items()}}

def main():
    base=ROOT/'outputs/regional_routing_v2';data=json.loads((base/'evaluation.json').read_text());rows=data['rows']
    cfg=json.loads((ROOT/'research/protocols/regional_routing_protocol_v2.json').read_text());policies=cfg['policies']
    names=['facenet_cosine','facenet_gallery_cosine','arcface_conditioning_cosine','arcface_conditioning_gallery_cosine','niqe','brisque','lpips','ssim_rgb','psnr_rgb','hole_psnr','hole_mae','visible_mae']
    summary={}
    for policy in policies:
        group=[r for r in rows if r['policy']==policy];entry={'rows':len(group),'failed':sum(r['evaluation']['status']!='complete' for r in group)}
        for metric in names:
            values=[r['evaluation'].get('metrics',{}).get(metric) for r in group];values=[v for v in values if v is not None]
            entry[metric]={'mean':float(np.mean(values)) if values else None,'valid':len(values)}
        summary[policy]=entry
    comparisons={p:paired(rows,p) for p in policies if p!='regional'};previous=0
    for i,p in enumerate(sorted(comparisons,key=lambda p:comparisons[p]['p_exact'] if comparisons[p]['p_exact'] is not None else 1)):
        v=comparisons[p]['p_exact'];previous=max(previous,min(1.,(5-i)*(v if v is not None else 1.)));comparisons[p]['p_holm']=previous
    condition_means={}
    for condition in sorted({r['condition'] for r in rows}):
        condition_means[condition]={}
        for policy in policies:
            vals=[r['evaluation'].get('metrics',{}).get('facenet_cosine') for r in rows if r['condition']==condition and r['policy']==policy];vals=[v for v in vals if v is not None]
            condition_means[condition][policy]={'mean':float(np.mean(vals)) if vals else None,'valid':len(vals)}
    record={'summary':summary,'primary_comparisons':comparisons,'condition_means':condition_means,'evaluation_sha256':sha(base/'evaluation.json'),'final_test_used':False,'trainable_parameters':0}
    write_new(base/'summary.json',record);write_new(ROOT/'research/regional_routing_results_v2.json',record)
    fmt=lambda v:'NA' if v is None else f'{v:.4f}'
    lines=['# Regional routing v2 — development results','','Four previously observed development identity groups; three conditions; two seeds; six policies. 144 logical rows, with 24 original concatenation controls reused by verified hashes. No reserved-final images used; no training performed.','',
           '| Policy | FaceNet target ↑ | FaceNet gallery ↑ | NIQE ↓ | BRISQUE ↓ | Hole MAE ↓ | Valid target / 24 |','|---|---:|---:|---:|---:|---:|---:|']
    for p,r in summary.items():lines.append(f"| {p} | {fmt(r['facenet_cosine']['mean'])} | {fmt(r['facenet_gallery_cosine']['mean'])} | {fmt(r['niqe']['mean'])} | {fmt(r['brisque']['mean'])} | {fmt(r['hole_mae']['mean'])} | {r['facenet_cosine']['valid']} |")
    lines+=['','| Regional minus | Identities | Pairs | Mean delta | Bootstrap 95% interval | Exact p | Holm p |','|---|---:|---:|---:|---|---:|---:|']
    for p,r in comparisons.items():lines.append(f"| {p} | {r['identities']} | {r['pairs']} | {fmt(r['mean_delta'])} | {r['ci95']} | {fmt(r['p_exact'])} | {fmt(r['p_holm'])} |")
    lines+=['','This small reused development sample cannot establish unknown-identity generalization or superiority. Bootstrap intervals with four identities are unstable. Whole-image metrics can conceal local distortions. The numerical JSON retains condition means and coverage for all metrics.','',
            'Equal routing averages separately normalized attention outputs; original concatenation normalizes across all reference tokens jointly. The original concatenation control is therefore essential. All routing arms retain four references and equal weights outside damage. Regional routing uses global FaceID descriptors, not local image patches or learned spatial correspondence. It is built on existing Diffusers mask functionality and is not a novelty claim.','',
            'Every generated output and all failures remain local under outputs/regional_routing_v2. Numerical evidence can be published without dataset photographs. Broader development validation, actual visibility estimation, resolution calibration, spatial reference features, external baselines and final evaluation remain outstanding.']
    (ROOT/'research/REGIONAL_ROUTING_RESULTS_V2.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    cases={c['case_id']:c for c in json.loads((ROOT/'outputs/generalization_v1/cases/manifest.json').read_text())['cases']}
    folder=base/'contact_sheets';folder.mkdir(exist_ok=True)
    for cid,c in cases.items():
        for seed in cfg['seeds']:
            images=[('Clean evaluation target',c['evaluation_only_target']['path']),('Damaged input',c['observed'])]
            images.extend((p,next(r for r in rows if r['case_id']==cid and r['seed']==seed and r['policy']==p).get('output')) for p in policies)
            sheet=Image.new('RGB',(4*256,2*280),'white');draw=ImageDraw.Draw(sheet)
            for i,(label,path) in enumerate(images):
                x,y=i%4*256,i//4*280;draw.text((x+5,y+4),label,fill='black')
                if path:
                    with Image.open(path) as im:sheet.paste(im.convert('RGB').resize((256,256)),(x,y+24))
                else:draw.text((x+5,y+70),'FAILED',fill='red')
            sheet.save(folder/f'{cid}_seed{seed}.png')
    safe_rows=[{k:v for k,v in r.items() if k!='output'} for r in rows]
    diagnostics=[]
    for r in rows:
        if r['status']!='complete' or r['policy']=='concat':continue
        meta_path=Path(r['output']).with_suffix('.json')
        if sha(meta_path)!=r['metadata_sha256']:raise ValueError('Generation metadata changed')
        meta=json.loads(meta_path.read_text());routing=meta['regional_routing']
        diagnostics.append({'case_id':r['case_id'],'seed':r['seed'],'policy':r['policy'],
                            'reference_sha256':[v['sha256'] for v in meta['references']],
                            'global_weights':routing['global_weights'],'region_weights':routing['region_weights'],
                            'map_sha256':routing['map_sha256'],'weight_sum_max_error':routing['weight_sum_max_error'],
                            'analyzer_config_sha256':routing['analysis']['config_sha256'],
                            'input_sha256':meta['input_sha256'],'mask_sha256':meta['mask_sha256'],
                            'seconds':meta['inference_seconds_including_offload'],'peak_allocated_bytes':meta['peak_allocated_bytes'],
                            'adapter_sha256':meta['adapter_sha256'],'environment':meta['environment']})
    write_new(ROOT/'research/regional_routing_evidence_v2.json',{'signature':json.loads((base/'signature.json').read_text()),'evaluation_signature':json.loads((base/'evaluation_signature.json').read_text()),'rows':safe_rows,'routing_diagnostics':diagnostics})
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
