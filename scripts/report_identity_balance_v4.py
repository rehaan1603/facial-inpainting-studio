"""Verify v4 controls and publish measured candidate comparison and provenance."""
import json,hashlib,csv
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/identity_balance_v4'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 prior=json.loads((ROOT/'outputs/identity_compatibility_v2/scores.json').read_text())
 old={(r['identity'],r['arm'],r['seed'],r['condition']):r for r in prior['rows']}
 results={};receipt={};keys=['psnr','ssim','lpips','facenet_cosine','arcface_conditioning_cosine','eye_bbox_lpips']
 lines=['# Balanced identity supervision and bounded gradients: v4','', '3 October 2026. Completed development experiment; production model unchanged.','', '| Method / condition | PSNR | SSIM | LPIPS | FaceNet | ArcFace (valid n) | Eye LPIPS |','|---|---:|---:|---:|---:|---:|---:|']
 csvrows=[]
 for arm in ['balanced_fixed','balanced_capped']:
  root=BASE/arm;r=json.loads((root/'scores.json').read_text());results[arm]=r
  assert len(r['rows'])==96 and all('metrics' in x and 'evaluation_error' not in x for x in r['rows'])
  trace=json.loads((root/'trace.json').read_text());counts=Counter(x['identity'] for x in trace if 'identity_raw' in x)
  assert len(counts)==16 and set(counts.values())=={4}
  verified=0
  for x in r['rows']:
   assert sha(root/'evaluation'/x['file'])==x['sha256']
   if x['arm']!=arm:
    original='identity' if x['arm']=='identity_full' else x['arm']
    p=old[(x['identity'],original,x['seed'],x['condition'])]
    assert p['sha256']==x['sha256'] and p['metrics']==x['metrics'], (arm,x['file'],'control mismatch')
    verified+=1
   else:csvrows.append({**{k:x[k] for k in ['identity','arm','seed','condition']},**{k:x['metrics'][k] for k in keys}})
  assert verified==72
  ratios=[x['weighted_identity_grad_norm']/max(x['diffusion_grad_norm'],1e-12) for x in trace]
  if arm=='balanced_capped':assert max(ratios)<1.00001
  receipt[arm]=dict(verified_controls=verified,new_images=24,supervised_identity_counts=dict(counts),gradient_ratio_min=min(ratios),gradient_ratio_max=max(ratios),files={name:sha(root/name) for name in ['config.json','training.json','adapter.pt','trace.json','scores.json']},gate=r['gates'][arm])
  for method in (['baseline',arm] if arm=='balanced_fixed' else [arm]):
   for condition in ['clean','wrong_patch']:
    values=r['summary'][f'{method}/{condition}/all']
    cells=[f"{values[k]['mean']:.6f}" if values[k]['mean'] is not None else 'unavailable' for k in keys]
    cells[4]+=f" ({values[keys[4]]['n']})"
    lines.append('| '+method+' / '+condition+' | '+' | '.join(cells)+' |')
 lines.extend(['','## Verification','','Both arms completed 64 GPU updates, with four identity-supervised updates for every training identity. Each generated 24 fresh images. All 144 reused control records match their original image hashes and metric values exactly; these represent the same 72 controls counted twice, not 144 independent samples.',''])
 for arm,v in receipt.items():
  lines.extend([f"- {arm}: identity/reconstruction gradient ratio {v['gradient_ratio_min']:.3f}–{v['gradient_ratio_max']:.3f}; joint promotion gate **{v['gate']['promote']}**."])
 lines.extend(['','## Interpretation and remaining work','','The fixed-versus-capped pair isolates loss weighting with identical supervision coverage. Comparison with v3 also changes identity-supervision frequency. FaceNet is supervised and is not an independent identity metric. ArcFace coverage is incomplete and remains explicit in the gate. One training seed and four repeatedly observed development identities cannot establish generalization to unknown client photographs. Full-image blur/noise restoration is tested here, not erased-eye inpainting.','', 'Review each failed criterion before choosing the next experiment. A successful gradient cap is an optimization check, not evidence of improved visual quality. Do not replace the production model automatically. More training seeds, broader corruptions and a separately held-out assessment remain necessary after a candidate passes the development criteria.'])
 (ROOT/'research/IDENTITY_BALANCE_RESULTS_V4.md').write_text('\n'.join(lines),encoding='utf-8')
 (ROOT/'research/identity_balance_v4_provenance.json').write_text(json.dumps(receipt,indent=2))
 with (ROOT/'research/identity_balance_v4_metrics.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['identity','arm','seed','condition',*keys]);writer.writeheader();writer.writerows(csvrows)
 print(json.dumps({k:v['gate'] for k,v in receipt.items()},indent=2))
if __name__=='__main__':main()
