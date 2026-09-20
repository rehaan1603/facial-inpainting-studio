"""Report all local-fusion arms with identity-level paired inference."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, write_new
from report_distortion_aware import METRICS, paired

BASE = ROOT / 'outputs/local_latent_v1'
POLICIES = ['global', 'regional_global', 'single', 'equal', 'quality', 'damage']

def main():
    rows = json.loads((BASE / 'evaluation.json').read_text())['rows']
    assert len(rows) == 48 and len({r['key'] for r in rows}) == 48
    for row in rows:
        if row['status'] == 'complete':
            assert sha(row['output']) == row['output_sha256']
    summaries, grouped = {}, {}
    for policy in POLICIES:
        group = [r for r in rows if r['policy'] == policy]
        summaries[policy] = {'rows': len(group), 'metrics': {}}
        grouped[policy] = []
        for identity in sorted({r['identity'] for r in group}):
            identity_rows = [r for r in group if r['identity'] == identity]
            assert {r['seed'] for r in identity_rows} == {17, 29}
            values = {}
            for metric in METRICS:
                vals = [r['evaluation'].get('metrics', {}).get(metric) for r in identity_rows]
                values[metric] = float(np.mean(vals)) if all(v is not None for v in vals) else None
            grouped[policy].append({'identity': identity, 'evaluation': {'metrics': values}})
        for metric in METRICS:
            vals = [r['evaluation'].get('metrics', {}).get(metric) for r in group]
            valid = [v for v in vals if v is not None]
            summaries[policy]['metrics'][metric] = {'mean': float(np.mean(valid)) if valid else None, 'valid_rows': len(valid)}
    contrasts = {p: paired(grouped['damage'], grouped[p]) for p in POLICIES if p != 'damage'}
    previous = 0
    for index, policy in enumerate(sorted(contrasts, key=lambda p: contrasts[p]['p_exact'] or 0)):
        c = contrasts[policy]
        previous = max(previous, min(1, (len(contrasts)-index) * (c['p_exact'] if c['p_exact'] is not None else 1)))
        c['p_holm'] = previous
    safeguards = {p: {m: paired(grouped['damage'], grouped[p], m) for m in METRICS} for p in POLICIES if p != 'damage'}
    result = {'rows': len(rows), 'new_generations': sum('reused_from' not in r for r in rows), 'summaries': summaries, 'primary_damage_minus_control': contrasts, 'descriptive_metrics': safeguards, 'evaluation_sha256': sha(BASE/'evaluation.json'), 'failed_rows': [r['key'] for r in rows if r['evaluation']['status'] != 'complete'], 'final_test_used': False, 'statistical_unit': 'identity, paired mean over both seeds; incomplete metric pairs excluded'}
    write_new(BASE/'summary.json', result)
    write_new(ROOT/'research/local_latent_results_v1.json', result)
    write_new(ROOT/'research/local_latent_evidence_v1.json', {'evaluation_signature': json.loads((BASE/'evaluation_signature.json').read_text()), 'rows': [{k:v for k,v in r.items() if k not in ['output', 'metadata']} for r in rows]})
    lines = ['# Aligned local-feature fusion: development diagnostic', '', 'Four previously observed development identities, mixed damage, seeds 17 and 29. Six matched policies; 48 scored rows. Native aligned reference VAE features are injected into masked-image conditioning after the first denoising step. Global FaceID context remains identical. No trainable parameters and no reserved-final use.', '', '| Policy | FaceNet target | Gallery | LPIPS | Hole MAE | NIQE | BRISQUE |', '|---|---:|---:|---:|---:|---:|---:|']
    for p in POLICIES:
        lines.append('| '+p+' | '+' | '.join('NA' if summaries[p]['metrics'][m]['mean'] is None else f"{summaries[p]['metrics'][m]['mean']:.4f}" for m in ['facenet_cosine','facenet_gallery_cosine','lpips','hole_mae','niqe','brisque'])+' |')
    lines += ['', 'Primary contrasts: damage-conditioned minus each control. Seeds are averaged within identity before uncertainty estimates; five tests use Holm correction.', '', '| Control | Delta | 95% identity bootstrap CI | Exact p | Holm p |', '|---|---:|---|---:|---:|']
    for p,c in contrasts.items():
        lines.append(f"| {p} | {c['mean_delta']:.4f} | {c['ci95']} | {c['p_exact']:.3f} | {c['p_holm']:.3f} |")
    lines += ['', 'All metrics and detection coverage are retained in local_latent_results_v1.json. Four identity units cannot support precise inference (minimum two-sided exact p = 0.125). This mixed-only mean is not directly comparable to the historical three-condition mean of 0.8180. Five-point similarity alignment does not solve 3D pose, expression or occlusion. Native feature injection is out of the frozen UNet training distribution; successful execution does not prove useful conditioning.', '', 'Zero-gain passthrough and spatial injection are unit-tested; an independent full GPU zero-gain equivalence run has not yet been performed. No learned adapter or candidate reranking has been run in this phase.']
    (ROOT/'research/LOCAL_LATENT_RESULTS_V1.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    infer = {c['case_id']: c for c in json.loads((ROOT/'outputs/distortion_aware_v1/inference_manifest.json').read_text())['cases']}
    evaluation = {c['case_id']: c for c in json.loads((ROOT/'outputs/distortion_aware_v1/evaluation_manifest.json').read_text())['cases']}
    out = BASE/'contact_sheets'; out.mkdir(exist_ok=True)
    for cid in sorted({r['case_id'] for r in rows}):
        for seed in [17,29]:
            photos = [('Clean scoring target', evaluation[cid]['target']['path']), ('Observed', infer[cid]['observed'])]
            photos += [(p,next(r['output'] for r in rows if r['case_id']==cid and r['seed']==seed and r['policy']==p)) for p in POLICIES]
            sheet = Image.new('RGB',(1024,560),'white'); draw=ImageDraw.Draw(sheet)
            for n,(label,path) in enumerate(photos):
                x,y=n%4*256,n//4*280; draw.text((x+4,y+4),label,fill='black')
                with Image.open(path) as im: sheet.paste(im.convert('RGB').resize((256,256)),(x,y+24))
            sheet.save(out/f'{cid}_seed{seed}.png')
    print(json.dumps({'summaries': summaries, 'contrasts': contrasts},indent=2))

if __name__ == '__main__': main()
