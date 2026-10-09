"""Publish focused three-person results and copy labelled previews into local packs."""
import json,hashlib,shutil,statistics
from pathlib import Path
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
PACK=Path.home()/'Downloads/All Face Datasets/Ready to test sets/Reference reconstruction - 3 people - 2026-10-09'
IDS=['263','6681','10149']
METRICS=['hole_psnr','hole_ssim','hole_lpips','facenet_cosine','arcface_conditioning_cosine']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    directories={name:ROOT/'outputs'/folder for name,folder in [('512 hard','masked_sdxl_v1'),('1024 hard','masked_sdxl_resolution_v1'),('1024 blended','focused_reference_finish_v1')]}
    summaries={name:json.loads((folder/'summary.json').read_text()) for name,folder in directories.items()}
    cases={(r['identity'],r['location']):r for r in json.loads((ROOT/'outputs/masked_task_v1/cases.json').read_text())}
    selected={};tables=[]
    for name,data in summaries.items():
        selected[name]={(r['identity'],r['location']):r for r in data['rows'] if r['identity'] in IDS and r['seed']==17 and r['condition']=='clean' and r['status']=='complete'}
        assert len(selected[name])==6
        for location in ['eyes','mouth']:
            row=dict(method=name,location=location,n=3)
            for key in METRICS:
                vals=[selected[name][ident,location]['metrics'][key] for ident in IDS]
                assert all(v is not None for v in vals);row[key]=statistics.mean(vals)
            tables.append(row)
    receipt=dict(date='2026-10-09',identities=IDS,training=False,scope='Three repeated-development identities, one inference seed, clean references; not unknown-client accuracy',
                 selected_tables=tables,summary_sha256={k:sha(directories[k]/'summary.json') for k in directories},
                 baseline_attempts=48,baseline_completions=36,baseline_reference_failures=12,
                 resolution_attempts=8,resolution_completions=6,resolution_reference_failures=2,
                 limitations=['Reference-detection failure identity7400 is retained in broader results','ArcFace conditions FaceID','Identity cosine is not accuracy percentage','Gaze and expression may be wrong'])
    lines=['# Focused reference reconstruction: three-person development results','',
           '9 October 2026. User requested focus on three usable-reference faces and local test packs. Existing identities263,6681,10149 are fixed across every comparison. No training, new dataset, synthetic references or checkpoint replacement.',
           '', '## What changed','',
           'Two sequential isolated comparisons: (1) existing SDXL generation resolution512→1024, holding references/seed17/steps30/scale0.8/strength1/guidance5 fixed; (2) existing Poisson boundary harmonization applied to the identical1024 raw outputs. Resolution also changes latent-noise dimensions and invokes existing VAE tiling, so equal seeds are seed labels, not identical noise tensors. Original512 and1024 raw/hard outputs are retained.',
           '', '## Matched results','',
           'Every row below uses the SAME three identities, seed17 and clean references. PSNR/SSIM concern the actual hole. LPIPS is a contextual hole crop. Both identity metrics have3/3 valid pairs per row; ArcFace is conditioning-related, not independent. These cosine scores are not calibrated accuracy percentages.',
           '', '| Hole | Method | Hole PSNR ↑ | Hole SSIM ↑ | Hole crop LPIPS ↓ | FaceNet ↑ | ArcFace ↑ |', '|---|---|---:|---:|---:|---:|---:|']
    for row in tables:lines.append('| '+row['location']+' | '+row['method']+' | '+' | '.join(f'{row[k]:.6f}' for k in METRICS)+' |')
    lines += ['', '## Interpretation','',
              'The final tested settings improve all five average metrics relative to512 hard composition on this three-person clean-reference screen. Resolution alone reduces SSIM; blending removes much of the colour discontinuity. Relative to1024 hard composition, blended eye ArcFace decreases slightly, so this is not a claim that every stage improves every metric. Eye colour, gaze and expression remain imperfect in visual comparisons. Do not promote these observations to population-level accuracy or a novelty claim.',
              '', 'The broader512 run attempted48 cases:36 outputs and12 reference-detection failures, all for identity7400. The1024 screen attempted8 cases:6 outputs and2 analogous failures. None were replaced or silently dropped. Three-face focus was explicitly requested by the user; results on those three must remain labelled development results.',
              '', '## Reproduce in the local website','',
              'Select **Use1–4 reference photos**, add all four photos, choose **1024px**, enable **Match colour at edges**, select **Use mask exactly**, and enable **Ignore opaque cover colour**. Use the damaged input and its matching mask from the same test folder. This routes through the existing neutralizing wrapper at strength1. The clean original is for evaluation only and must not be included as a reference.',
              '', 'Test-pack location: `Downloads/All Face Datasets/Ready to test sets/Reference reconstruction - 3 people - 2026-10-09/`.',
              '', '## Remaining focused work','',
              'Keep these cases fixed while testing identity/gaze preservation and robustness to a misleading reference. A visual improvement and higher whole-face cosine do not establish correct eye geometry. Repeat a selected candidate across the previously fixed seeds and corrupted references before changing general website defaults. No global default or model checkpoint has changed in this experiment.',
              '', '[Audit](MASKED_INPAINTING_AUDIT_AND_PLAN_20261008.md) · [SDXL protocol](MASKED_SDXL_PREFLIGHT_V1.md) · [Resolution protocol](MASKED_SDXL_RESOLUTION_PREFLIGHT_V1.md) · [Numerical receipt](focused_reference_v1_results.json)']
    for i,ident in enumerate(IDS,1):
        folder=PACK/f'Set {i:02d} - Person {ident}';dest=folder/'Tested outputs';dest.mkdir(exist_ok=True)
        for location in ['eyes','mouth']:
            c=cases[ident,location];panels=[('Original - evaluation',Path(c['target'])),('Damaged input',Path(c['input']))]
            for name in ['512 hard','1024 hard','1024 blended']:
                r=selected[name][ident,location];source=directories[name]/r['file'];assert sha(source)==r['sha256']
                p=dest/f'{location.capitalize()} - {name}.png'
                if p.exists():assert sha(p)==sha(source)
                else:shutil.copy2(source,p)
                if name!='1024 hard':panels.append((name,source))
            canvas=Image.new('RGB',(1024,290),'white');draw=ImageDraw.Draw(canvas)
            for col,(label,p) in enumerate(panels):
                canvas.paste(Image.open(p).convert('RGB').resize((256,256)),(col*256,34));draw.text((col*256+6,10),label,fill='black')
            canvas.save(dest/f'{location.capitalize()} - comparison.png')
        (folder/'TEST SETTINGS.txt').write_text('Tested reference settings\n\nModel: Use1–4 reference photos\nReferences: all FOUR Reference photos\nProcessing detail:1024px\nMask treatment: Use mask exactly\nMatch colour at edges: ON\nIgnore opaque cover colour: ON\n\nStart with Eyes test/Damaged input.png plus its mask.\nOriginal is evaluation-only.\nTested outputs contains preserved512,1024 and blended results.\nImprovement is measured on these cases; gaze/expression can still differ.\n',encoding='utf-8')
    website=ROOT/'outputs/focused_reference_website_v1/receipt.json'
    if website.exists():
        web=json.loads(website.read_text());receipt['website_checks']=web
        lines+=['','## Actual website verification','',f'{len(web)} recorded website runs; {sum(r["status"]=="complete" for r in web)} complete; {sum(r.get("experiment_replay_exact",False) for r in web)} byte-exact matches to the corresponding offline candidate. See JSON for individual jobs.']
    strength=ROOT/'outputs/focused_reference_strength_v1'
    if (strength/'summary.json').exists():
        trial=json.loads((strength/'summary.json').read_text());pairs=[]
        lines+=['','## Lower reference-strength probe: retained as experimental','',
                'Only FaceID scale changed from 0.8 to 0.4 on the same three eye cases. Mean hole PSNR/SSIM/LPIPS and identity improve, but person10149 loses substantial FaceNet similarity. The lower gain is not selected as a universal three-person setting. No per-person scale selection is presented as an automatic method.',
                '', '| Person | FaceNet at 0.8 | FaceNet at 0.4 | Delta |', '|---|---:|---:|---:|']
        for r in trial['rows']:
            ident=r['identity'];control=selected['1024 blended'][ident,'eyes']
            diff={k:r['metrics'][k]-control['metrics'][k] for k in METRICS}
            pairs.append(dict(identity=ident,control=control['metrics'],candidate=r['metrics'],deltas=diff))
            lines.append(f'| {ident} | {control["metrics"]["facenet_cosine"]:.6f} | {r["metrics"]["facenet_cosine"]:.6f} | {diff["facenet_cosine"]:+.6f} |')
            folder=PACK/f'Set {IDS.index(ident)+1:02d} - Person {ident}'/'Experimental - not selected';folder.mkdir(exist_ok=True)
            p=folder/'Eyes - reference strength 0.4.png';source=strength/r['file']
            if p.exists():assert sha(p)==sha(source)
            else:shutil.copy2(source,p)
            (folder/'READ ME.txt').write_text('Lower reference strength improved some metrics, but reduced identity similarity for one person. This is an experimental alternative, not the selected three-person setting. See TEST SETTINGS.txt for the website-tested setup.\n',encoding='utf-8')
        receipt['lower_strength_probe']=dict(selected=False,reason='Identity regression for one person; no per-person outcome selection',pairs=pairs)
    lines+=['','## Reference integration check','',
            'A read-only inspection found correct normalized face embeddings, conditional/unconditional ordering and all four references reaching attention as64 projected tokens. No concrete integration defect was found. The reference path supplies identity vectors, not explicit gaze, iris colour or spatial correspondence. The author describes Portrait as not requiring a LoRA; adding one blindly would not address the observed failure. [Author model card](https://huggingface.co/h94/IP-Adapter-FaceID).',
            '', 'The source image is currently reduced to512 before optional1024 inference. That limits retained source detail for high-resolution uploads; these frozen tests already use512 inputs, so changing source preprocessing would require a separate test.']
    (ROOT/'research/focused_reference_v1_results.json').write_text(json.dumps(receipt,indent=2))
    document='\n'.join(lines)+'\n'
    for old,new in [('Use1–4','Use 1–4'),('1024px','1024 px'),('source512','source 512'),('as64','as 64'),('to512','to 512'),('optional1024','optional 1024'),('use512','use 512')]:document=document.replace(old,new)
    (ROOT/'research/FOCUSED_REFERENCE_RESULTS_V1.md').write_text(document,encoding='utf-8')
    print(json.dumps(tables,indent=2))


if __name__=='__main__':main()
