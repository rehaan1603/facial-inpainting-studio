"""Signed, resumable development-only reference contribution diagnostic."""
import csv, hashlib, json, time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch, lpips
from reference_inpaint import reconstruct, CACHE
from atomic_records import write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/reference_diagnostic_evaluation_v3'
MEASURES=['full_face_lpips','hole_mae','visible_mae','inner_boundary_mae']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rgb(path):
    with Image.open(path) as im:return np.asarray(im.convert('RGB')).copy()
def write(path,value):
    write_json(path,value)

def report(rows):
    assert len(rows)==96 and len({r['identity'] for r in rows})==12
    table={}
    for r in rows:
        key=f"scale{r['scale']}_strength{r['strength']}_{r['compositor']}"
        table.setdefault(key,[]).append(r)
    summary={k:{m:float(np.mean([r[m] for r in group])) for m in MEASURES} for k,group in table.items()}
    contrasts=[]
    def contrast(a,b,label):
        aa={r['identity']:r for r in table[a]};bb={r['identity']:r for r in table[b]};ids=sorted(aa);assert set(aa)==set(bb)
        rng=np.random.default_rng(20260912);indices=rng.integers(0,len(ids),(10000,len(ids)));values={}
        for measure in MEASURES:
            delta=np.array([aa[i][measure]-bb[i][measure] for i in ids]);boot=delta[indices].mean(1)
            values[measure]={'mean':float(delta.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist()}
        contrasts.append({'comparison':label,'a':a,'b':b,'a_minus_b':values})
    for strength in [1.0,.99]:
        for compositor in ['hard','poisson']:
            contrast(f'scale0.8_strength{strength}_{compositor}',f'scale0.0_strength{strength}_{compositor}',f'Reference on minus off; strength {strength}; {compositor}')
    for scale in [0.0,.8]:
        for compositor in ['hard','poisson']:
            contrast(f'scale{scale}_strength0.99_{compositor}',f'scale{scale}_strength1.0_{compositor}',f'Strength .99 minus 1; scale {scale}; {compositor}')
        for strength in [1.0,.99]:
            contrast(f'scale{scale}_strength{strength}_poisson',f'scale{scale}_strength{strength}_hard',f'Poisson minus hard; scale {scale}; strength {strength}')
    result={'scope':'Development diagnostic only; no identity fidelity or novelty claim','identities':12,'generated_candidates':48,'scored_rows':96,'means':summary,'paired_identity_bootstrap':contrasts,'metrics_sha256':sha(OUT/'metrics.csv')}
    write(ROOT/'research/reference_diagnostic_results.json',result)
    lines=['# Reference contribution: development diagnostic','', 'All 48 candidates and 96 scored rows completed on twelve development identity labels. The same generator, seed and references were used in every paired contrast. These results do not constitute a final test, an identity-fidelity evaluation or a novel-method claim.','', '| Configuration | LPIPS | Hole MAE | Visible MAE | Inner boundary MAE |','|---|---:|---:|---:|---:|']
    for key,values in summary.items():lines.append('| '+key+' | '+' | '.join(f'{values[m]:.6f}' for m in MEASURES)+' |')
    lines+=['','Lower is better for these error measures. Exact outside-mask preservation is imposed by composition, not learned by the model.','', '## Paired LPIPS differences','', '| Contrast (A minus B) | Mean | 95% identity bootstrap interval |','|---|---:|---|']
    for c in contrasts:
        v=c['a_minus_b']['full_face_lpips'];lines.append(f"| {c['comparison']} | {v['mean']:.6f} | [{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}] |")
    lines+=['','Negative differences favour A. All other paired measures are recorded in reference_diagnostic_results.json. Intervals are exploratory and not adjusted for multiple contrasts. Strength 0.99 uses 29 denoising steps versus 30 at strength 1.0.','', 'Limitations: twelve detector-success identities, one synthetic eye mask per person, single seed, no poor-reference or inaccurate-mask strata, uncertain label/pretraining independence and no independent identity evaluator. Do not use this diagnostic to claim real-world identity recovery. See REFERENCE_DIAGNOSTIC_PROTOCOL_V3.md and outputs/reference_diagnostics_v1/README.md.']
    (ROOT/'research/REFERENCE_DIAGNOSTIC_RESULTS.md').write_text('\n'.join(lines)+'\n')

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.hub.set_dir(str(CACHE/'torch'))
    manifest_path=ROOT/'outputs/reference_diagnostics_v1/manifest.json';manifest=json.loads(manifest_path.read_text());cases=manifest['cases'];assert len(cases)==12
    paths=[manifest_path,Path(__file__),ROOT/'scripts/reference_inpaint.py',ROOT/'scripts/reference_blending.py',ROOT/'scripts/atomic_records.py',ROOT/'research/REFERENCE_DIAGNOSTIC_PROTOCOL_V3.md',ROOT/'research/reference-environment-v2-lock.txt',ROOT/'research/reference_adapter_provenance.json',ROOT/'research/reference_faces_provenance.json',ROOT/'research/osor_downloads.json']
    signature={str(p.relative_to(ROOT)):sha(p) for p in paths}
    from importlib.metadata import version
    signature['runtime_versions']={name:version(name) for name in ['torch','torchvision','diffusers','transformers','tokenizers','regex','onnx','ml_dtypes','insightface','opencv-python','lpips']}
    guard=OUT/'signature.json'
    if guard.exists():assert json.loads(guard.read_text())==signature,'Inputs changed; use a new versioned run.'
    else:write(guard,signature)
    perceptual=lpips.LPIPS(net='alex').cpu().eval();runtime={};rows=[];completed=0
    tensor=lambda a:torch.from_numpy(a.transpose(2,0,1).copy()).float()[None]/127.5-1
    for c in cases:
        image=ROOT/c['observed'];mask=ROOT/c['mask'];assert sha(image)==c['observed_sha256'] and sha(mask)==c['mask_sha256']
        for item in c['images']:assert sha(ROOT/item['file'])==item['processed_sha256']
        refs=[ROOT/i['file'] for i in c['images'] if i['role'].startswith('reference_')]
        target_path=ROOT/next(i['file'] for i in c['images'] if i['role']=='target')
        for scale in [0.0,.8]:
            for strength in [1.0,.99]:
                name=f"{c['case_id']}_scale{scale}_strength{strength}";folder=OUT/name;folder.mkdir(exist_ok=True);chunk=folder/'scores.json'
                if chunk.exists():rows.extend(json.loads(chunk.read_text()));completed+=1;continue
                write(OUT/'progress.json',{'status':'running','completed_candidates':completed,'total_candidates':48,'current':name})
                try:
                    meta=reconstruct(image,mask,refs,folder/'result.png',scale=scale,strength=strength,runtime=runtime,progress_path=folder/'progress.json')
                    # Scoring target is deliberately absent from the reconstruction API call.
                    target=rgb(target_path);observed=rgb(image);binary=np.asarray(Image.open(mask).convert('L'))>=128
                    boundary=binary & ~cv2.erode(binary.astype('uint8'),np.ones((9,9),np.uint8)).astype(bool);scores=[]
                    for compositor,suffix in [('hard','result_hard.png'),('poisson','result.png')]:
                        result=rgb(folder/suffix);assert np.array_equal(result[~binary],observed[~binary])
                        err=np.abs(result.astype('float32')-target.astype('float32'))/255
                        with torch.inference_mode():lp=float(perceptual(tensor(result),tensor(target)).item())
                        assert np.isfinite(lp)
                        scores.append({'identity':c['identity_label'],'case_id':c['case_id'],'scale':scale,'strength':strength,'compositor':compositor,'full_face_lpips':lp,'hole_mae':float(err[binary].mean()),'visible_mae':float(err[~binary].mean()),'inner_boundary_mae':float(err[boundary].mean()),'inference_seconds':meta['inference_seconds_including_offload'],'output_sha256':sha(folder/suffix)})
                    write(chunk,scores);rows.extend(scores);completed+=1
                    write(OUT/'progress.json',{'status':'running','completed_candidates':completed,'total_candidates':48,'scored_rows':len(rows)})
                    print(f'Diagnostic {completed}/48 candidates',flush=True)
                except Exception as error:
                    write(folder/'failure.json',{'error':str(error)});write(OUT/'progress.json',{'status':'failed','completed_candidates':completed,'total_candidates':48,'current':name,'error':str(error)});raise
    assert len(rows)==96
    with (OUT/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    report(rows);write(OUT/'progress.json',{'status':'complete','completed_candidates':48,'total_candidates':48,'scored_rows':96})

if __name__=='__main__':main()
