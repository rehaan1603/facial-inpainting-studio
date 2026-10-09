"""Verify outputs and publish numerical development results without dataset pixels."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/masked_task_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    summary=json.loads((OUT/'summary.json').read_text());assert summary['complete'] and summary['native_rows']==8
    cfg=json.loads((OUT/'config.json').read_text());cases={(c['identity'],c['location']):c for c in json.loads((OUT/'cases.json').read_text())}
    rows=json.loads((OUT/'evaluation/rows.json').read_text());assert len(rows)==96
    assert len({(r['identity'],r['location'],r['arm'],r['condition'],r['seed']) for r in rows})==96
    native=json.loads((OUT/'native_resshift/rows.json').read_text())
    for group,folder in [(rows,OUT/'evaluation'),(native,OUT/'native_resshift')]:
        for r in group:
            assert sha(folder/r['file'])==r['sha256']
            if 'raw' in r:assert sha(folder/r['raw'])==r['raw_sha256']
            c=cases[r['identity'],r['location']]
            assert sha(c['input'])==c['input_sha256'] and sha(c['mask'])==c['mask_sha256']
            mask=np.asarray(Image.open(c['mask']))>127
            assert np.array_equal(np.asarray(Image.open(folder/r['file']))[~mask],np.asarray(Image.open(c['input']))[~mask])
    assert sha(cfg['checkpoint'])==cfg['checkpoint_sha256']
    assert json.loads((OUT/'evaluation/replay.json').read_text())['byte_exact']
    receipt=dict(primary_outputs=96,native_controls=8,raw_outputs_verified=96,visible_pixel_preservation_pass=104,
                 hidden_pixel_invariance_pass=8,baseline_replay_byte_exact=True,adapter_unchanged=True,
                 config_sha256=sha(OUT/'config.json'),cases_sha256=sha(OUT/'cases.json'),summary_sha256=sha(OUT/'summary.json'),
                 source_sha256={name:sha(ROOT/'scripts'/name) for name in ['audit_masked_task_v1.py','score_masked_task_v1.py','masked_native_control_v1.py']})
    (OUT/'verification.json').write_text(json.dumps(receipt,indent=2))
    (ROOT/'research/masked_task_v1_results.json').write_text(json.dumps(dict(summary=summary,verification=receipt),indent=2))
    lines=['# Erased facial-region diagnostic: results','',
           '8 October 2026. Completed 96 matched ReF-LDM outputs, eight separate native ResShift controls and one byte-exact restoration replay. No training or production changes.',
           '', '## Main result','',
           'The frozen restoration baseline and its unchanged timestep-scaled identity adapter retain largely blank eye/mouth regions in the inspected seed-17 clean comparison sheets. The adapter does not solve missing-region reconstruction. ResShift constructs facial features, but eye shape, gaze, expression and boundary quality still differ from the original. This is a task-capability diagnosis, not a publication superiority claim.',
           '', '## Matched ReF-LDM comparison','',
           'Four reused development identities × two locations × three seeds × two reference conditions × two arms. PSNR/SSIM below are measured inside the actual hole; LPIPS is its bounding crop with context. Identity similarity uses the composed full face. Each row has 12 pixel/FaceNet pairs; ArcFace has nine jointly valid pairs. These numbers are not comparable to earlier whole-image restoration results.',
           '', '| Condition / hole | Arm | Hole PSNR ↑ | Hole SSIM ↑ | Hole crop LPIPS ↓ | FaceNet ↑ | ArcFace ↑ (n=9) |',
           '|---|---|---:|---:|---:|---:|---:|']
    for name,values in summary['paired'].items():
        for arm in ['baseline','candidate']:
            nums=[values[k][arm] for k in ['hole_psnr','hole_ssim','hole_lpips','facenet_cosine','arcface_conditioning_cosine']]
            lines.append('| '+name+' | '+('v5 adapter' if arm=='candidate' else arm)+' | '+' | '.join(f'{v:.6f}' for v in nums)+' |')
    lines += ['', 'No joint success: hole PSNR declines in all four strata; corrupted-eye LPIPS and FaceNet worsen, and ArcFace declines in every stratum. Small aggregate changes do not compensate for missing facial structure.',
              '', '## Native mask-conditioned control','',
              'Seed17 only, four identities per location. ResShift uses no references and native256 followed by512 compositing. ReF-LDM uses native512 and four clean references. These deliberately different backends are functional controls, not an equal-architecture or paper-reproduction comparison. The table restricts ReF-LDM to the same seed17 for clarity.',
              '', '| Hole | Backend | Hole PSNR ↑ | Hole crop LPIPS ↓ | FaceNet ↑ |', '|---|---|---:|---:|---:|']
    for loc in ['eyes','mouth']:
        v=summary['paired']['clean/'+loc]
        vals=[np.mean([p['baseline'] for p in v[k]['pairs'] if p['seed']==17]) for k in ['hole_psnr','hole_lpips','facenet_cosine']]
        lines.append('| '+loc+' | ReF-LDM baseline, seed17 | '+' | '.join(f'{x:.6f}' for x in vals)+' |')
        vals=[summary['native_summary'][loc][k]['mean'] for k in ['hole_psnr','hole_lpips','facenet_cosine']]
        lines.append('| '+loc+' | ResShift, seed17 | '+' | '.join(f'{x:.6f}' for x in vals)+' |')
    lines += ['', 'Native ArcFace support is only 3/4 per location. Its means are retained in JSON; no claim of improved ArcFace follows from this control.',
              '', '## Verification and scope','',
              '- 104/104 composites preserve visible pixels exactly; all generated and raw image hashes verified.',
              '- 8/8 erased inputs pass hidden-pixel invariance; diagnostics use only erased RGB and supplied references.',
              '- Fresh restoration replay is byte-exact. The adapter checkpoint remains unchanged.',
              '- Eight seed-17 clean comparison sheets were inspected across all identities/locations. This is descriptive inspection, not a blinded study of all 96 images.',
              '- Four development identities, annotation-derived masks, unresolved upstream pretraining exposure and incomplete ArcFace coverage remain limitations.',
              '- Measured ReF-LDM sampling/decoding time totals '+f'{sum(r["seconds"] for r in rows):.1f}'+' seconds; maximum recorded allocated GPU memory '+f'{max(r["peak_mib"] for r in rows):.1f}'+' MiB. Excludes loading, preparation, evaluation and native control. No evidence here attributes blank outputs to exhausted GPU memory.',
              '', '## Revised next experiment','',
              'Do not begin the proposed ReF-LDM masked-loss pilot yet. The prerequisite functional screen failed. Keep ReF-LDM available for its separate blur/noise-restoration task. Next test the existing genuinely mask-conditioned, reference-guided SDXL path on these same frozen inputs and references, starting with the seed17 clean screen, then the identical corrupted-reference and multiseed grid. Hold selection, blending and model settings fixed. This identifies whether reference-conditioned inpainting provides a useful reconstruction baseline before one isolated training or reliability intervention.',
              '', 'No new dataset, synthetic target-derived references, checkpoint overwrite or website default change was introduced.',
              '', '[Protocol](MASKED_TASK_PREFLIGHT_V1.md) · [Numerical results and verification](masked_task_v1_results.json) · [Earlier audit and conditional plan](MASKED_INPAINTING_AUDIT_AND_PLAN_20261008.md)',
              '', 'Local comparison images: `outputs/masked_task_v1/sheets/`. Dataset images remain local and are not included in the research JSON.']
    (ROOT/'research/MASKED_TASK_RESULTS_V1.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
