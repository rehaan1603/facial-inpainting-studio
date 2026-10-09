"""Supplementary anatomical diagnostics; preserve all primary evaluation files."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity
from extended_evaluation_metrics import ROOT, sha, read_rgb
from region_metrics_v1 import pixel_statistics

OUT=ROOT/'outputs/eye_brow_diagnostic_v1'
IDS=['263','6681','10149']

def main():
    OUT.mkdir(exist_ok=True)
    cases={c['identity']:c for c in json.loads((ROOT/'outputs/masked_task_v1/cases.json').read_text()) if c['location']=='eyes'}
    studies={'guidance_replication':'focused_eye_guidance_replication_v2'}
    if (ROOT/'outputs/focused_eye_appearance_v1/summary.json').exists():studies['appearance']='focused_eye_appearance_v1'
    if (ROOT/'outputs/focused_eye_factorial_v1/summary.json').exists():studies['interaction']='focused_eye_factorial_v1'
    results=[];receipts={}
    for study,folder in studies.items():
        summary=json.loads((ROOT/'outputs'/folder/'summary.json').read_text())
        for row in summary['rows']:
            ident=row['identity'];c=cases[ident]
            if row['status']!='complete':results.append(dict(study=study,identity=ident,seed=row['seed'],status=row['status']));continue
            output=read_rgb(ROOT/'outputs'/folder/row['file'],row['sha256']);target=read_rgb(c['target'],c['target_sha256'])
            assert sha(c['mask'])==c['mask_sha256'];missing=np.asarray(Image.open(c['mask']))>127
            hq=int(c['hq_id']);base=Path.home()/'Downloads/archive/CelebAMask-HQ/CelebAMask-HQ-mask-anno'/str(hq//2000)
            regions={};files=[]
            for name,parts in {'eyes':['l_eye','r_eye'],'brows':['l_brow','r_brow']}.items():
                mask=np.zeros(missing.shape,bool)
                for part in parts:
                    p=base/f'{hq:05d}_{part}.png';assert p.is_file(),f'Missing anatomical annotation {p}'
                    mask|=np.asarray(Image.open(p).convert('L').resize((512,512),Image.Resampling.NEAREST))>127
                    files.append(dict(path=str(p),sha256=sha(p)))
                regions[name]=mask&missing
            _,map_=structural_similarity(output.astype(np.float64)/255,target.astype(np.float64)/255,
                        data_range=1,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,win_size=11,full=True)
            if map_.ndim==3:map_=map_.mean(2)
            values={}
            for name,mask in regions.items():
                valid=binary_erosion(mask,structure=np.ones((11,11),bool),border_value=0)
                values[name]=dict(pixel_statistics(output,target,mask),ssim=float(map_[valid].mean()) if valid.any() else None,
                                  valid_ssim_centres=int(valid.sum()))
            receipts[ident]=files
            if study=='guidance_replication':
                old=row['regions']['eyes']['ssim'];new=values['eyes']['ssim']
                assert old==new or (old is not None and new is not None and abs(old-new)<1e-12), 'Anatomical eye SSIM differs from the frozen scorer'
            results.append(dict(study=study,identity=ident,seed=row['seed'],arm=row.get('condition'),status='complete',regions=values))
    lines=['# Supplementary eye and eyebrow fidelity diagnostics','',
           'Primary masked-region metrics and success criteria remain unchanged. This appendix responds to observed iris/gaze/brow errors: the larger erased rectangle includes substantial skin and is different from annotated anatomical eyes and eyebrows. Official target masks are used ONLY after inference for evaluation.','',
           'Pixel PSNR/MAE use only each annotated region inside the original missing mask. SSIM uses the same Gaussian sigma 1.5, 11-by-11, population-variance RGB map as the frozen evaluator, averaging only fully contained windows. Sparse or absent valid windows are disclosed; null SSIM is not replaced with zero. No iris-only pigmentation or gaze-angle accuracy is measured.','',
           '| Study | Person | Seed | Arm | Region | Pixel PSNR | MAE | SSIM | Valid centres | Pixels |',
           '|---|---|---:|---|---|---:|---:|---:|---:|---:|']
    for row in results:
        if row['status']!='complete':lines.append(f"| {row['study']} | {row['identity']} | {row['seed']} | FAILED | NA | NA | NA | NA | 0 | 0 |");continue
        for name,r in row['regions'].items():
            fmt=lambda v:'NA' if v is None else f'{v:.6f}'
            lines.append(f"| {row['study']} | {row['identity']} | {row['seed']} | {row['arm']} | {name} | {fmt(r['psnr'])} | {fmt(r['mae'])} | {fmt(r['ssim'])} | {r['valid_ssim_centres']} | {r['pixels']} |")
    lines += ['','Anatomical eye SSIM from this calculation matches the existing stored scorer within 1e-12. Brow annotations and this diagnostic are supplementary, not a replacement protocol or a newly tuned loss. See all paired primary measurements in FOCUSED_EYE_GUIDANCE_REPLICATION_RESULTS_V2.md.']
    (OUT/'results.json').write_text(json.dumps(dict(rows=results,annotation_sources=receipts,source_sha256=sha(__file__),primary_metrics_changed=False),indent=2,allow_nan=False))
    (ROOT/'research/EYE_BROW_FIDELITY_DIAGNOSTIC_V1.md').write_text('\n'.join(lines)+'\n')
    print('Supplementary anatomical diagnostics:',len(results),'rows')

if __name__=='__main__':main()
