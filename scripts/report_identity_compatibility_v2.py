"""Generate research report and tables from immutable experiment receipts."""
import csv,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from reference_intervention_core import ROOT

def main():
    out=ROOT/'outputs/identity_compatibility_v2'
    r=json.loads((out/'scores.json').read_text());cfg=json.loads((out/'config.json').read_text());tr=json.loads((out/'training.json').read_text())
    keys=['psnr','ssim','lpips','facenet_cosine','arcface_conditioning_cosine','eye_bbox_lpips']
    csvpath=ROOT/'research/identity_compatibility_v2_metrics.csv'
    with csvpath.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['identity','arm','seed','condition','status',*keys,'mae','niqe','brisque','facenet_detection','arcface_detection'])
        writer.writeheader()
        for row in r['rows']:
            writer.writerow({**{k:row[k] for k in ['identity','arm','seed','condition','status']},
                **{k:row.get('metrics',{}).get(k) for k in [*keys,'mae','niqe','brisque']},
                'facenet_detection':row.get('output_detections',{}).get('facenet',{}).get('status'),
                'arcface_detection':row.get('output_detections',{}).get('arcface_conditioning',{}).get('status')})
    lines=['# Facial Inpainting Studio: method investigation report', '',
        'Updated 2 October 2026. Current research checkpoint, not a finalized successful method. Existing dataset only. All prior evidence and the local application remain preserved.', '',
        '## 1. Research question', '',
        'Can a compact target-reference adapter improve perceptual restoration and corrupted-reference robustness without reducing identity or eye-region fidelity?', '',
        '## 2. Baseline', '',
        'Frozen local ReF-LDM, 512 RGB, four references, 50 DDIM steps, guidance 1.5, bfloat16, uniform_trailing. Individually encoded references differ from author benchmark RGB concatenation. Compare identical local inputs/settings, not incompatible published scores.', '',
        '## 3. Proposed method', '',
        'Retain the 193-parameter v1 compatibility architecture. Compare a new balanced diffusion-only control with an otherwise identical identity-supervised arm. Both train for 64 updates on the same 16 identities. This experiment isolates the added objective within the new pair; the old 128-update adapter has a different training budget.', '',
        '## 4. Architecture', '',
        'Degraded target and reference VQ features produce regional cosine, magnitude, consensus and quality/pose descriptors. A 10-to-16-to-1 MLP generates relative reference attention log-priors in 16 frozen attention layers. Five fixed regions include both eyes, nose, mouth and context. Neutral final-layer initialization reproduces the baseline.', '',
        '## 5. Mathematical formulation', '',
        '`b(r,i) = log_softmax_i(MLP(features(target,r,reference_i))) + log(N)`', '',
        '`attention = softmax(Q K^T / sqrt(d) + b) V`', '',
        'Reference priors affect the conditional branch, with zero self-attention prior. Relative normalization cannot independently suppress every reference; independent reliability attenuation remains a separate proposed ablation.', '',
        '## 6. Training objective', '',
        '`Ldiff = 0.25 * mean(epsilon_error^2) + 0.75 * mean(face_rectangle_epsilon_error^2)`', '',
        '`L = Ldiff + lambda_id * (1 - cosine(FaceNet(decoded_x0), FaceNet(target)))`', '',
        'Identity loss runs every fourth update. The frozen FaceNet encoder receives a full aligned image resized to 160, a surrogate that differs from MTCNN evaluation cropping. The identity weight is the first supervised-step diffusion/identity gradient-norm ratio, capped at 10. No development scores select the weight or checkpoint. Backbone and encoder stay frozen. This is not full implementation of consistency, regularization, perceptual and all regional objectives.', '',
        '| Arm | Updates | Identity weight | Training seconds | Peak allocated MiB |', '|---|---:|---:|---:|---:|']
    for a in tr['arms']:lines.append(f"| {a['arm']} | {a['steps']} | {a['identity_weight']} | {a['seconds']:.2f} | {a['peak_mib']:.1f} |")
    lines+=['', 'The identity run exceeds 8 GiB in allocated tensors. It completed on this laptop, but this does not establish comfortable dedicated-VRAM headroom or distinguish shared-memory spillover.', '',
        '## 7. Experimental protocol', '',
        'Four observed development identities: 263, 6681, 7400, 10149. Three inference seeds: 17, 29, 43. Two reference conditions: clean and central wrong-person patch. Four methods give 96 generations. Training seed is 20261002; corruption/timestep assignments are independently shuffled. No reserved final identities or new Human Faces dataset used. This task is whole-image blur/noise restoration, not erased-region inpainting.', '',
        '## 8. Quantitative results', '',
        '| Method / reference condition | PSNR | SSIM | LPIPS | FaceNet | ArcFace (valid n) | Eye crop LPIPS |', '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ['baseline','regional_v1','diffusion','identity']:
        for condition in ['clean','wrong_patch']:
            s=r['summary'][f'{arm}/{condition}/all'];vals=[f"{s[k]['mean']:.6f}" for k in keys]
            vals[4]+=f" ({s['arcface_conditioning_cosine']['n']}/12)"
            lines.append(f'| {arm} / {condition} | '+' | '.join(vals)+' |')
    lines+=['', 'These are three-seed means. They must not be compared to the earlier seed-17-only baseline as a before/after effect. All whole-image and FaceNet entries have 12 valid measurements per row; ArcFace has 8.', '',
        '## 9. Regional results', '',
        'Identity supervision improves average eye bounding-box LPIPS in both conditions relative to the baseline, but not at every seed. Bounding-box LPIPS includes ten-pixel context and is not strictly pixel-masked LPIPS. Official parsing masks are evaluation-only. Whole-image damage means masked metrics equal full-image metrics; the visible region is empty. Nose, mouth, left/right eye and other-region values remain in JSON.', '',
        '## 10. Identity results', '',
        'FaceNet averages decrease relative to baseline in both conditions. With a misleading patch, adding identity loss recovers some of the diffusion-only control drop, but does not reach the baseline. FaceNet is supervised in this arm and therefore is not an independent evaluator. ArcFace does not train the adapter; its incomplete support remains explicit.', '',
        '## 11. Corruption robustness', '',
        'Training includes clean, blur, noise and wrong-person patch. This new screen evaluates clean and wrong-person patch only. Previous v1 eight-condition results remain separate. Eye/mouth swaps, irrelevant and mixed references, and one bad among four good references still require a frozen protocol. The latter has five total references and needs a five-reference baseline.', '',
        '## 12. Ablations', '',
        'Completed: original local backbone; previous 193-parameter regional adapter; new balanced diffusion-only control; matched identity-supervised arm. The earlier 545-parameter adapter is not rerun in this three-seed panel. Independent reliability gating, global/regional variants, consistency, and the full requested factorial ablations remain to be tested.', '',
        '## 13. Failures and rejection', '',
        '96/96 generations and metric rows completed. No candidate meets all per-seed safeguards. None replaces the website baseline. ArcFace target detection fails for identity 7400; seed-17 outputs for 10149 also lack ArcFace detections. These failures already occur in archived baseline scores, not a new evaluator inconsistency. Every new-panel FaceNet output and target detects one face.', '',
        '## 14. Statistical analysis', '',
        'JSON reports mean, SD, median and counts by condition/seed/method, plus paired identity deltas, improved/worsened counts and 2,000 identity-cluster bootstrap intervals. Four identity units give exploratory intervals only; no statistical-significance claim. Three inference seeds do not substitute for three training seeds. The 1e-6 tolerance is numerical reproducibility tolerance, not a justified clinical/perceptual non-inferiority margin.', '',
        '## 15. Limitations', '',
        'Small reused development cohort; full-image training crop differs from recognition evaluation; one-step decoded x0 supervision differs from final multi-step output; fixed anatomical rectangles; limited corruption coverage; constrained VRAM; unresolved pretraining overlap; no unknown-person generalization evidence.', '',
        '## 16. Prior-art comparison', '',
        'Target queries and reference-region selection have prior art. Adding a frozen identity loss also has prior art. Current results do not support a new advantageous mechanism. Published context remains in [paper comparison](PAPER_VS_PROJECT_COMPARISON_20261002.md); local matched evidence is the primary comparison.', '',
        '## 17. Remaining work and next experiment', '',
        'Before larger training, align the training recognition crop to fixed training-face detections and compare the same two arms with an image/identity objective closer to final output. Audit gradients across multiple training examples instead of trusting one calibration example. Keep inference evaluators frozen. Independently rejecting unreliable references is the next architectural ablation after the objective test. Do not expand the final evaluation or repeat hyperparameter tuning on it. Larger 20–50-person development, multiple targets, other corruptions, true external portraits and final testing remain pending.', '',
        '## 18. What the evidence supports', '',
        'Successful engineering: exact metric reproduction, differentiable frozen identity supervision, matched GPU training, 96 multi-seed generations, retained failures and explicit paired evaluation. The tested identity objective improves average eye crop LPIPS but fails the combined identity/perceptual requirement. The strongest deployable candidate remains the original local ReF-LDM baseline. Work remains in progress; a successful novelty hypothesis is not established by this experiment.', '',
        'Files: [audit](IDENTITY_COMPATIBILITY_AUDIT_V2.md), [all metrics and statistics](identity_compatibility_v2_results.json), [CSV](identity_compatibility_v2_metrics.csv). Training configs/checkpoints/traces remain under `outputs/identity_compatibility_v2`. No dataset images or weights are published.', '']
    (ROOT/'research/FINAL_METHOD_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    sheets=out/'comparisons';sheets.mkdir(exist_ok=True)
    for identity in cfg['evaluation']['identities']:
        for seed in cfg['evaluation']['seeds']:
            for condition in cfg['evaluation']['conditions']:
                sheet=Image.new('RGB',(1536,290),'white');d=ImageDraw.Draw(sheet)
                base=ROOT/'outputs/reference_intervention_pilot_v1/images'/identity
                panels=[('Target, scoring only',base/'target.png'),('Damaged',base/'input.png')]
                panels += [(a,out/'evaluation'/f'{identity}_{a}_{seed}_{condition}.png') for a in ['baseline','regional_v1','diffusion','identity']]
                for i,(label,p) in enumerate(panels):
                    d.text((i*256+5,5),label,fill='black')
                    if p.exists():
                        with Image.open(p) as im:sheet.paste(im.convert('RGB').resize((256,256)),(i*256,28))
                sheet.save(sheets/f'{identity}_{seed}_{condition}.png')
    print('Report, CSV and all 24 comparison sheets saved')

if __name__=='__main__':main()
