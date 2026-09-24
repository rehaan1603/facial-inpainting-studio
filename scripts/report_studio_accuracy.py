"""Summarize the completed studio audit with per-case evidence and denominators."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha,write_new
from studio_accuracy_audit import OUT,BASE,PROTOCOL,cases


def main():
    data=json.loads((OUT/'evaluation.json').read_text());rows=data['rows']
    assert len(rows)==44
    names=['facenet_cosine','facenet_gallery_cosine','arcface_conditioning_cosine','arcface_conditioning_gallery_cosine','hole_mae','visible_mae','lpips','ssim_rgb','psnr_rgb','hole_psnr','niqe','brisque']
    groups={};pairs={}
    lookup={(r['case_id'],r['arm']):r for r in rows}
    for kind in ['removal','mixed']:
        for arm in sorted({r['arm'] for r in rows if r['kind']==kind}):
            subset=[r for r in rows if r['kind']==kind and r['arm']==arm]
            valid=[r for r in subset if r['evaluation']['status']=='complete'];key=kind+'/'+arm
            groups[key]={'planned':len(subset),'scored':len(valid),'metrics':{}}
            for name in names:
                values=[r['evaluation']['metrics'].get(name) for r in valid];values=[v for v in values if v is not None]
                groups[key]['metrics'][name]={'mean':float(np.mean(values)) if values else None,'valid':len(values)}
            groups[key]['known_pixels_exact']=sum(r['evaluation']['metrics']['known_pixels_unchanged'] for r in valid)
            if arm=='observed':continue
            pairs[key]={}
            for name in ['facenet_cosine','facenet_gallery_cosine','hole_mae','lpips']:
                differences=[]
                for r in valid:
                    control=lookup[(r['case_id'],'observed')]['evaluation']
                    a=r['evaluation']['metrics'].get(name);b=control.get('metrics',{}).get(name)
                    if a is not None and b is not None:differences.append(float(a-b))
                benefit=1 if name.startswith('facenet') else -1
                pairs[key][name]={'mean_delta_vs_observed':float(np.mean(differences)) if differences else None,'improved_identities':sum(d*benefit>0 for d in differences),'valid_pairs':len(differences)}
    result=dict(scope='Four already-observed development identities. Descriptive quality audit; no population accuracy or novelty claim.',protocol_sha256=sha(PROTOCOL),evaluation_sha256=sha(OUT/'evaluation.json'),expected_rows=44,scored_rows=sum(r['evaluation']['status']=='complete' for r in rows),groups=groups,paired_observed=pairs,rows=rows,final_test_used=False)
    historical=json.loads((BASE/'evaluation.json').read_text())['rows'];regression=[]
    for c in cases():
        if c['kind']!='removal':continue
        old=next(r for r in historical if r['key']==c['case_id']+'_s0.99_a0.8_seed17')
        meta_path=Path(old['output']).with_suffix('.json');assert sha(meta_path)==old['metadata_sha256']
        meta=json.loads(meta_path.read_text());assert sha(old['output'])==old['output_sha256']
        assert meta['input_sha256']==c['observed_sha256'] and meta['mask_sha256']==c['mask_sha256']
        assert meta['model_resolution']==[512,512] and meta['blending']['mode']=='poisson'
        assert meta['seed']==17 and meta['steps']==30 and meta['adapter_scale']==.8
        new=lookup[(c['case_id'],'reference_512')]['evaluation']
        regression.append({'case_id':c['case_id'],'old_output_sha256':old['output_sha256'],
            'historical_metrics':old['evaluation']['metrics'],'current_metrics':new.get('metrics',{}),
            'delta':{n:new.get('metrics',{}).get(n)-old['evaluation']['metrics'][n] if new.get('metrics',{}).get(n) is not None and old['evaluation']['metrics'].get(n) is not None else None for n in ['facenet_cosine','facenet_gallery_cosine','hole_mae','lpips']}})
    result['historical_512_regression_check']=dict(scope='Exploratory regression check against saved pre-repair generations, added after first audit case scored. Same inputs/masks, seed, adapter scale, steps, resolution and Poisson blending; preprocessing and strength differ.',source_evaluation_sha256=sha(BASE/'evaluation.json'),cases=regression)
    # Local paths contain no portable evidence; published rows keep hashes and metrics only.
    result['rows']=[{k:v for k,v in r.items() if k!='output'} for r in rows]
    write_new(ROOT/'research/studio_accuracy_v1.json',result)
    lines=['# Studio generation accuracy audit — 23 September 2026','',result['scope'],'',
           f"Completed scores: **{result['scored_rows']}/44**, including eight unchanged-image controls. Planned generations: **36**. No quality-based retries or discarded failures.",'',
           'FaceNet/ArcFace values are cosine similarities, not accuracy percentages. ArcFace is a conditioning-related diagnostic; FaceNet is the independent identity evaluator. All metrics retain missing-detection counts. Whole-image metrics can conceal errors in a small mask; masked MAE and local visual review remain essential.','',
           '| Condition / mode | Scored | FaceNet ↑ | Gallery ↑ | Mask MAE ↓ | LPIPS ↓ | SSIM ↑ | NIQE ↓ | BRISQUE ↓ |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v):return 'N/A' if v is None else f'{v:.4f}'
    for key,g in groups.items():
        cols=[fmt(g['metrics'][n]['mean']) for n in ['facenet_cosine','facenet_gallery_cosine','hole_mae','lpips','ssim_rgb','niqe','brisque']]
        for i,n in enumerate(['facenet_cosine','facenet_gallery_cosine']):cols[i]+=f" ({g['metrics'][n]['valid']}/{g['planned']})"
        lines.append('| '+key+f" | {g['scored']}/{g['planned']} | "+' | '.join(cols)+' |')
    lines+=['','## Paired checks against unchanged input','', '| Condition / mode | Identity improved | Mask error improved | Perceptual error improved |','|---|---:|---:|---:|']
    for key,stats in pairs.items():
        counts=[f"{stats[n]['improved_identities']}/{stats[n]['valid_pairs']}" for n in ['facenet_cosine','hole_mae','lpips']]
        lines.append('| '+key+' | '+' | '.join(counts)+' |')
    lines+=['','## Previous 512-pixel reference path: regression check','',result['historical_512_regression_check']['scope'],'','| Case | Identity delta (new minus old) | Mask MAE delta | LPIPS delta |','|---|---:|---:|---:|']
    for r in regression:lines.append('| '+r['case_id']+' | '+' | '.join(fmt(r['delta'][n]) for n in ['facenet_cosine','hole_mae','lpips'])+' |')
    lines+=['','## Limits and remaining work','',
            'This audit covers each offered generation variant with the exact painted mask, not every mask treatment, seed, damage severity or uploaded image. Partial-damage modes use a fixed user-style evidence weight of 128/255; these are not calibrated confidence predictions. ReF-LDM is deliberately excluded from complete removal because it does not fill missing regions.',
            '', 'All four identities were already used in development. Four independent identities do not establish generalization; no significance or publication-ready novelty claim is made. Better average scores do not guarantee correct expression, gaze, or hidden anatomy. Reserved final-test identities remain untouched.',
            '', 'Per-image metrics, missing detections, output hashes, and paired counts are in `studio_accuracy_v1.json`. Local comparison sheets include clean targets only for evaluation and remain excluded from GitHub. Visual findings are recorded separately after inspection.']
    with (ROOT/'research/STUDIO_ACCURACY_V1.md').open('x',encoding='utf-8') as stream:
        stream.write('\n'.join(lines)+'\n')
    evaluations={e['case_id']:e for e in json.loads((BASE/'evaluation_manifest.json').read_text())['cases']}
    for c in cases():
        e=evaluations[c['case_id']];assert sha(e['target']['path'])==e['target']['sha256']
        entries=[('CLEAN TARGET',e['target']['path'])]+[(r['arm'],r.get('output')) for r in rows if r['case_id']==c['case_id']]
        sheet=Image.new('RGB',(1024,280*((len(entries)+3)//4)),'white');draw=ImageDraw.Draw(sheet)
        for i,(label,path) in enumerate(entries):
            x=(i%4)*256;y=(i//4)*280;draw.text((x+5,y+5),label+(' (FAILED)' if not path else ''),fill='black')
            if path:sheet.paste(Image.open(path).convert('RGB').resize((256,256)),(x,y+24))
        sheet.save(OUT/(c['case_id']+'_review.png'))
    print(json.dumps({'scored':result['scored_rows'],'groups':groups},indent=2))


if __name__=='__main__':main()
