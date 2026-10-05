"""Publish measured crop-ablation evidence without modifying prior experiments."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from reference_intervention_core import ROOT

def main():
    out=ROOT/'outputs/identity_crop_v3';r=json.loads((out/'scores.json').read_text())
    trace=json.loads((out/'identity_crop_trace.json').read_text());audit=[x for x in trace if 'identity_raw' in x]
    ratios=[x['weighted_identity_grad_norm']/x['diffusion_grad_norm'] for x in audit]
    crops=json.loads((out/'crops.json').read_text())['records'];tr=json.loads((out/'training.json').read_text())['arms'][0]
    keys=['psnr','ssim','lpips','facenet_cosine','arcface_conditioning_cosine','eye_bbox_lpips']
    with (ROOT/'research/identity_crop_v3_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['identity','arm','seed','condition',*keys,'status','reused_from']);writer.writeheader()
        for x in r['rows']:writer.writerow({**{k:x.get(k,'') for k in ['identity','arm','seed','condition','status','reused_from']},**{k:x.get('metrics',{}).get(k) for k in keys}})
    lines=['# Fixed face-crop identity supervision: v3', '', '3 October 2026. Existing dataset, separate from production and previous experiment outputs.', '',
        '## Change and controlled comparison', '',
        'Changed only the training recognition crop: frozen target MTCNN bounds replace full-image resizing. Preserve the v2 193-parameter architecture, initialization seed, 64-update sample/timestep schedule, optimizer, periodic supervision and loss weight 0.5408350358795648. Both target and predicted training image use the same fixed box. Evaluation still independently detects target/output faces. Ground-truth training crops never enter inference.', '',
        'All 16 training targets have one detected face. Antialiased bilinear tensor resizing matches crop bounds, output size and standardization, but avoids PIL uint8 rounding. Minimum crop embedding agreement with the evaluator on training targets: '+f"{min(x['embedding_cosine_vs_evaluator_crop'] for x in crops):.8f}.", '',
        'Two crop checks pass: invalid boxes are rejected and gradients remain inside the crop. Backbone and recognition encoder stay frozen.', '',
        '## Execution and loss diagnostics', '',
        f"64 GPU updates completed in {tr['seconds']:.2f} seconds; peak allocated tensor memory {tr['peak_mib']:.1f} MiB. This exceeds 8 GiB and does not establish dedicated-VRAM headroom.", '',
        f"Identity supervision ran at {len(audit)} steps across {len({x['identity'] for x in audit})} of the 16 training identities. Diffusion training still covers all 16. The inherited every-fourth-update schedule does not give equal identity-supervision coverage.", '',
        f"Weighted identity/diffusion gradient norm ratio: {min(ratios):.3f} to {max(ratios):.3f}. Opposing gradient directions: {sum(x['gradient_cosine']<0 for x in audit)}/{len(audit)} checks. This diagnoses the risk of one-example weight calibration; it does not by itself prove why a final image improves or worsens.", '',
        '## Matched results', '',
        'Four observed development identities, inference seeds 17/29/43, clean and wrong-person-patch references. 24 fresh cropped-candidate images; 72 controls reused only after byte-hash verification. All 96 rows are evaluated under the same frozen metrics. Reused controls are not new generation successes.', '',
        '| Method / condition | PSNR | SSIM | LPIPS | FaceNet | ArcFace (n) | Eye crop LPIPS |', '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ['baseline','regional_v1','identity_full','identity_crop']:
        for c in ['clean','wrong_patch']:
            s=r['summary'][f'{arm}/{c}/all'];v=[f"{s[k]['mean']:.6f}" if s[k]['mean'] is not None else 'NA' for k in keys]
            v[4]+=f" ({s['arcface_conditioning_cosine']['n']}/12)"
            lines.append(f'| {arm} / {c} | '+' | '.join(v)+' |')
    lines+=['', 'Higher PSNR/SSIM/cosine and lower LPIPS are better. FaceNet is training supervision, not an independent metric. Eye crop LPIPS includes context and is not strictly pixel-masked. This remains full-image restoration, not erased-eye completion.', '',
        '## Decision and next work', '',
        'Cropped-candidate promotion checks: `'+json.dumps(r['gates']['identity_crop'])+'`.', '',
        'Any metric regressions, incomplete supports and seed-dependent results remain visible. No production replacement follows a failing screen. The next objective experiment should balance supervision across all training identities and examine bounded gradient contributions across multiple training examples, separately from any architecture change. Independently suppressing all misleading references remains a subsequent gating ablation.', '',
        'Only four observed identities and one training seed are available here. Three inference seeds do not establish training-seed robustness. Final held-out data and the separate Human Faces work remain unused. Current numeric tolerance is 1e-6 reproducibility tolerance, not a perceptual non-inferiority margin.', '',
        '[All measurements and paired statistics](identity_crop_v3_results.json), [CSV](identity_crop_v3_metrics.csv), [provenance](identity_crop_v3_provenance.json). The v2 report is preserved in [previous objective experiment](FINAL_METHOD_REPORT.md).', '']
    (ROOT/'research/IDENTITY_CROP_RESULTS_V3.md').write_text('\n'.join(lines),encoding='utf-8')
    records=[]
    for name in ['crops.json','config.json','training.json','identity_crop_trace.json','evaluation/provenance.json']:
        p=out/name;records.append(dict(source='outputs/identity_crop_v3/'+name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),data=json.loads(p.read_text())))
    (ROOT/'research/identity_crop_v3_provenance.json').write_text(json.dumps(dict(records=records),indent=2),encoding='utf-8')
    sheets=out/'comparisons';sheets.mkdir(exist_ok=True)
    for ident in ['263','6681','7400','10149']:
        for c in ['clean','wrong_patch']:
            sheet=Image.new('RGB',(1280,290),'white');d=ImageDraw.Draw(sheet)
            panels=[('Target / evaluation only',ROOT/'outputs/reference_intervention_pilot_v1/images'/ident/'target.png')]
            panels += [(a,out/'evaluation'/f'{ident}_{a}_17_{c}.png') for a in ['baseline','regional_v1','identity_full','identity_crop']]
            for i,(label,p) in enumerate(panels):
                d.text((i*256+4,5),label,fill='black')
                if p.exists():
                    with Image.open(p) as im:sheet.paste(im.convert('RGB').resize((256,256)),(i*256,28))
            sheet.save(sheets/f'{ident}_17_{c}.png')

if __name__=='__main__':main()
