"""Generate fixed held-out pilot cases without using clean targets in sampling."""
import json
import time
import numpy as np
from PIL import Image, ImageDraw
import torch
from reference_intervention_core import ROOT, ReferenceBias, install_adapter, load_model, sha


def main():
    out = ROOT/'outputs/reference_intervention_pilot_v1'
    manifest = json.loads((out/'manifest.json').read_text())
    receipts = json.loads((out/'cache_receipt.json').read_text())
    training = json.loads((out/'training/receipt.json').read_text())
    assert len(training['arms']) == 2
    assert training['core_sha256'] == sha(ROOT/'scripts/reference_intervention_core.py')
    assert receipts['manifest_sha256'] == sha(out/'manifest.json')
    dest = out/'evaluation'
    dest.mkdir(exist_ok=False)
    signature = dict(script_sha256=sha(__file__), training_receipt_sha256=sha(out/'training/receipt.json'),
        steps=50, seed=17, guidance=1.5, schedule='uniform_trailing', precision='bfloat16 autocast',
        status='engineering comparison, not exact paper reproduction', final_test_used=False)
    (dest/'signature.json').write_text(json.dumps(signature, indent=2))
    model = load_model().cuda()
    adapter = ReferenceBias().cuda()
    install_adapter(model, adapter)
    from ldm.models.diffusion.ddim import DDIMSampler
    sampler = DDIMSampler(model, print_tqdm=False, schedule='uniform_trailing')
    rows = []
    for record in receipts['records']:
        if record['role'] != 'adapter_holdout':
            continue
        identity = record['identity']
        path = out/'latents'/(identity+'.pt')
        assert sha(path) == record['sha256']
        bundle = torch.load(path, weights_only=True)
        # Target latent intentionally not transferred to GPU or supplied to sampler.
        lq = bundle['lq'].cuda()
        for arm in ['baseline', 'reconstruction', 'intervention', 'no_reference']:
            adapter.enabled = arm in ['reconstruction', 'intervention']
            adapter.drop_references = arm == 'no_reference'
            if adapter.enabled:
                checkpoint = out/'training'/(arm+'.pt')
                expected = next(r['checkpoint_sha256'] for r in training['arms'] if r['arm']==arm)
                assert sha(checkpoint) == expected
                adapter.load_state_dict(torch.load(checkpoint, weights_only=True))
            for condition in (['clean'] if arm=='no_reference' else ['clean', 'corrupt']):
                refs = bundle['references'].clone()
                if condition == 'corrupt':
                    refs[:1] = bundle['corrupt_reference']
                cond = dict(lq_image=lq, ref_image=torch.cat(list(refs.split(1)), -1).cuda())
                null = dict(lq_image=lq, ref_image=torch.zeros_like(cond['ref_image']))
                torch.manual_seed(17)
                noise = torch.randn(1, 8, 64, 64, device='cuda')
                start = time.monotonic()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    latent, _ = sampler.sample(S=50, unconditional_guidance_scale=1.5, conditioning=cond,
                        unconditional_conditioning=null, shape=[8,64,64], x_T=noise, batch_size=1, verbose=False)
                    decoded = ((model.decode_first_stage(latent)+1)/2).clamp(0,1)
                assert torch.isfinite(decoded).all()
                array = (decoded[0].permute(1,2,0).float().cpu().numpy()*255).round().astype(np.uint8)
                filename = f'{identity}_{arm}_{condition}.png'
                Image.fromarray(array).save(dest/filename)
                rows.append(dict(identity=identity, arm=arm, condition=condition, file=filename,
                                 sha256=sha(dest/filename), seconds=time.monotonic()-start))
                (dest/'rows.json').write_text(json.dumps(rows, indent=2))
                print(filename, flush=True)
    # Contact sheets are evaluation-only: never returned to the generator.
    for record in receipts['records']:
        if record['role'] != 'adapter_holdout':
            continue
        identity = record['identity']
        panels = [('Target', out/'images'/identity/'target.png'), ('Input', out/'images'/identity/'input.png')]
        panels += [(r['arm']+' '+r['condition'], dest/r['file']) for r in rows if r['identity']==identity]
        sheet = Image.new('RGB', (256*3, 284*3), 'white')
        draw = ImageDraw.Draw(sheet)
        for index,(label,path) in enumerate(panels):
            x,y = index%3*256,index//3*284
            with Image.open(path) as image:
                sheet.paste(image.resize((256,256)), (x,y+28))
            draw.text((x+4,y+5),label,fill='black')
        sheet.save(dest/(identity+'_sheet.jpg'))
    print('28 reconstructions complete', flush=True)


if __name__ == '__main__':
    main()
