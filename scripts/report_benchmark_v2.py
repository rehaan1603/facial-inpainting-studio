import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'research/benchmark_v2_results.json').read_text())
out=ROOT/'outputs/benchmark_v2'
rows=[x for x in r['summary'] if x['partition']=='assessment' and x['radius']!='oracle']
fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
ax.plot([x['visible_mae'] for x in rows],[x['full_face_lpips'] for x in rows],marker='o',color='#235789')
for x in rows: ax.annotate(f"r={x['radius']}",(x['visible_mae'],x['full_face_lpips']),xytext=(7,5),textcoords='offset points')
ax.set(xlabel='Mean change on genuinely visible pixels (lower is better)',
       ylabel='Full-face LPIPS (lower is better)',title='LaMa: reconstruction and preservation trade-off')
ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
fig.text(.5,-.025,'48 assessment identities · 15 synthetic cases per identity · validation only',ha='center',fontsize=9)
fig.savefig(out/'tradeoff.png',dpi=180,bbox_inches='tight');fig.savefig(out/'tradeoff.svg',bbox_inches='tight');plt.close(fig)
lines=['# Expanded validation benchmark','',
    'A simple dilation improves aggregate perceptual reconstruction while increasing changes to valid facial pixels. This establishes a useful engineering comparison; no new method has been trained or evaluated.','',
    '## Protocol','',
    '- 96 distinct validation identities, excluding all identities from the first pilot.',
    '- 48 tuning and 48 assessment identities with no overlap; final test remains unused for inference.',
    '- Three mask families: brush, eye region, mouth region.',
    '- Five conditions: accurate, undersized, oversized, translated, and spatially varying boundary error.',
    '- Dilation radii 0, 2, 4, 8; one global radius chosen by tuning LPIPS only.',
    '- LPIPS 0.1.4, AlexNet v0.1 weights and ImageNet backbone, RGB normalized to [-1,1].',
    '- 7,200 metric rows including the repeated exact-mask diagnostic; correlated masks are not independent observations.','',
    '## Assessment results','', '| Radius | Full-face LPIPS | True-hole MAE | Visible-pixel MAE |', '|---|---:|---:|---:|']
for x in rows: lines.append(f"| {x['radius']} | {x['full_face_lpips']:.5f} | {x['hole_mae']:.5f} | {x['visible_mae']:.5f} |")
d=r['assessment_lpips_selected_minus_radius0']
lines += ['',f"The tuning-selected radius is {r['selected_radius']}. Its paired assessment LPIPS difference relative to radius 0 is {d['mean']:.5f}, with a 95% identity-bootstrap interval [{d['bootstrap_95_interval'][0]:.5f}, {d['bootstrap_95_interval'][1]:.5f}] over 2,000 resamples. Negative values favor dilation. This interval describes this fixed synthetic development sample, not unseen real-world performance.", '',
    '## Interpretation and limits','',
    'The best radius lies at the largest tested value, so the sweep has not established an optimal dilation radius. Expand the tuning range before fixing a competitive final control. Do not present improvement over radius 0 alone as a learned contribution.', '',
    'The semantic families are not area-matched, and the corruption textures are synthetic. Clean parsing is used only to construct corruptions, never as an inference input. Exact-mask results use privileged information and are diagnostic. The model is a generic third-party LaMa export with unverified numerical equivalence to the original release.', '',
    'The bootstrap selects no parameters on assessment data, but this remains development validation that can inform later design. A future final evaluation needs locked decisions and untouched test data. No claims about identity preservation, real occluder removal, or state-of-the-art quality are supported yet.', '',
    '## Next step','',
    'Validate a face-specific baseline, extend simple correction controls, and construct area-matched cases. A proposed learned correction must improve the reconstruction–preservation curve against these controls. See benchmark_v2_results.json and outputs/benchmark_v2/metrics.csv for the complete measurements.']
smoke_path=ROOT/'research/resshift_smoke.json'
if smoke_path.exists():
    smoke=json.loads(smoke_path.read_text())
    lines += ['', '## Face-specific baseline compatibility', '',
        'The official ResShift face checkpoint and VAE loaded strictly against pinned source commit bb03b7d21614cace01787e097c8a6ab6b945227d. Four previously used validation cases completed at 256 pixels with four diffusion steps. This is a compatibility smoke test, not a matched model comparison or an official reproduction.', '',
        f"Peak allocated GPU memory was {smoke['peak_allocated_MiB']:.1f} MiB. The three calls after initialization took about 0.22 seconds each. This is not a controlled end-to-end latency benchmark.", '',
        'The preview contains plausible completions but visible differences from the hidden target, including eye details. Zero visible-region error follows from exact-mask compositing; it is not evidence of learned preservation. Inaccurate-mask behavior still needs a matched evaluation.', '',
        'The original configuration references FFHQ and the VAE is named celeba256; pretraining exposure is unresolved. Retain the S-Lab noncommercial license and record this caveat in comparisons. Sources, checkpoint hashes, and adapter settings are in resshift_provenance.json and resshift_downloads.json.']
(ROOT/'research/MILESTONE_02.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Wrote MILESTONE_02.md and tradeoff figure')
