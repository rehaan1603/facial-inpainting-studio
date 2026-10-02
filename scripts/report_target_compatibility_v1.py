"""Build descriptive tables from all retained pilot scores, without selecting winners."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/target_compatibility_v1'

def main():
    report=json.loads((OUT/'scores.json').read_text())
    rows=report['rows']
    lines=['# Target-conditioned regional compatibility: development pilot', '',
        'Date: 2 October 2026. Existing CelebA-HQ research only; new Human Faces work is set aside.', '',
        '**Task: full-image blur/noise restoration, not erased-region inpainting.** Four already-observed development identities, one seed. No independent or reserved-final evaluation. This is not publication-level superiority evidence.', '',
        '## Completed implementation and execution', '',
        '- Frozen ReF-LDM with a 193-parameter regional attention-log-prior network in 16 attention layers. Features combine observed degraded-target/reference latent similarity, pose and quality proxies, and reference consensus.',
        '- Two arms trained on the same 16 identities for 128 updates each; backbone weights frozen. Neutral initialization reproduced baseline predictions exactly.',
        '- 96 generations: four identities, three methods, eight reference conditions. Eight archived clean/wrong-patch baseline replays are byte-exact.',
        '- Conditions: clean, blur, noise, JPEG, downsampling, occlusion, lighting and wrong-person patch. Genuine pose mismatch remains untested as a controlled intervention.',
        '- All rows and failures retained. The `no_target` arm ablates explicit target similarity/pose; minimum target/reference detection confidence remains, so it is not strictly target-free.', '',
        '## Declared engineering gate', '',
        str(report['engineering_gate']), '',
        '**Decision: do not promote or expand this candidate.** LPIPS improvement is below the declared threshold. Three FaceNet measurements are unavailable (263: baseline/JPEG, full/noise, full/JPEG); partial-support identity deltas below are descriptive, not complete matched identity evidence. All 96 images were generated and scored for available metrics; failed identity detections are retained.', '',
        'Requires corrupted-reference LPIPS delta <= -0.005, PSNR delta >= -0.1 dB, FaceNet delta >= -0.005, at least three of four identities improving LPIPS, and complete paired support. Passing would only justify further development.', '',
        '| Contrast / stratum | LPIPS delta | Relative LPIPS reduction | PSNR delta (dB) | FaceNet cosine delta | LPIPS improved/worsened |',
        '|---|---:|---:|---:|---:|---:|']
    for name,c in report['contrasts'].items():
        l=c['lpips'];lines.append(f"| full minus {name} | {l['mean_delta']:+.6f} | {l['relative_error_reduction_percent']:+.3f}% | {c['psnr']['mean_delta']:+.4f} | {c['facenet_cosine']['mean_delta']:+.6f} | {l['improved']}/{l['worsened']} |")
    lines+=['', 'Negative LPIPS delta is better; positive PSNR/identity delta is better. Conditions are averaged within each person before the four-person comparison. Identity-cluster bootstrap intervals, medians, standard deviations and paired support are in the JSON; they are descriptive with only four observed people.', '',
        '## Per-condition whole-image means', '',
        '| Method / condition | LPIPS | PSNR | SSIM | FaceNet | ArcFace diagnostic | NIQE | BRISQUE |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in report['summary'].items():
        vals=[s[k]['mean'] for k in ['lpips','psnr','ssim','facenet_cosine','arcface_conditioning_cosine','niqe','brisque']]
        lines.append('| '+name+' | '+' | '.join('unavailable' if v is None else f'{v:.5f}' for v in vals)+' |')
    lines+=['', '## Facial-component checks (all seven corrupted conditions)', '',
        '| Method | Region | MAE | PSNR | Bbox LPIPS | Scored rows |', '|---|---|---:|---:|---:|---:|']
    for arm in ['baseline','no_target','full']:
        for region in ['eyes','nose','mouth','other']:
            rs=[r['regions'][region] for r in rows if r['arm']==arm and r['condition']!='clean' and 'regions' in r]
            vals=[]
            for key in ['mae','psnr','bbox_lpips']:
                valid=[r[key] for r in rs if r[key] is not None];vals.append(f'{np.mean(valid):.6f}' if valid else 'unavailable')
            lines.append(f'| {arm} | {region} | '+ ' | '.join(vals)+f' | {len(rs)} |')
    lines+=['', 'Components use official target annotations only during offline scoring. Bbox LPIPS includes ten-pixel context and is **not strictly masked LPIPS**. Fully contained 11x11 SSIM windows are required; thin regions may have no valid SSIM. All pixels are degraded here, so the visible region is empty and masked equals whole-image.', '',
        '## Loss-gradient audit', '', '| Arm | Loss | Weight | Mean raw value at audit steps | Mean weighted gradient norm |', '|---|---|---:|---:|---:|']
    for arm in ['no_target','full']:
        trace=json.loads((OUT/f'training/{arm}_trace.json').read_text())
        for loss in ['reconstruction','face_roi']:
            audits=[r['gradient_audit'][loss] for r in trace if r['gradient_audit']]
            lines.append(f"| {arm} | {loss} | {audits[0]['weight']} | {np.mean([a['raw'] for a in audits]):.6f} | {np.mean([a['weighted_gradient_norm'] for a in audits]):.8f} |")
    lines+=['', 'These are latent noise-prediction losses, not image-space reconstruction/identity losses. Eight gradient audits per arm record raw/weighted terms and gradients. Image perceptual, identity, visible-preservation and reference-consistency training losses remain unimplemented in this pilot; no claim that all requested losses have been tested.', '',
        '## Visual inspection', '',
        'Inspected 263/wrong_patch, 6681/noise, 7400/occlusion and 10149/blur comparison sheets. Candidate/control differences are subtle; smoothing and altered eye, mouth or fine facial details remain relative to the target. No consistent visual fidelity gain was apparent. All 32 sheets are saved, but this does not claim all 32 received human review.', '',
        '## Limitations and next decision', '',
        '- Corruption type and diffusion timestep blocks are coupled during training; the follow-up must balance them independently.',
        '- Adapter regions are canonical rectangles, not pose-aligned anatomical regions. Pose features are 2D proxies, not measured 3D pose.',
        '- Full, target-feature-ablated and baseline runs are complete. Global, target-only, reliability-only and other component ablations are not complete.',
        '- Reference selection already has close prior art; see [ten-paper literature and protocol matrix](MATCHED_COMPARISON_AND_LITERATURE_20261002.md). This module is not established novelty.',
        '- If either gate fails, keep the baseline deployed and diagnose on development data. Do not expand to reserved final identities or represent a failed pilot as an improvement.',
        '- Still required: a justified image-space objective, fully target-free control, matched inpainting panel, balanced training, multi-seed independent validation after a passing pilot, and final manuscript/human assessment.', '',
        'Numerical source: [all scores and paired statistics](target_compatibility_results_v1.json). Local comparison images are in `outputs/target_compatibility_v1/comparisons`; dataset photos are not distributed in the repository.', '']
    (ROOT/'research/TARGET_COMPATIBILITY_RESULTS_V1.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
