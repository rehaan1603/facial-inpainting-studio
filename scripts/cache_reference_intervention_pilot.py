"""Encode only explicitly selected pilot images; keep all pixels/latents local."""
import json
import time
import numpy as np
from PIL import Image
import torch
from reference_intervention_core import ROOT, load_model, sha


def main():
    out = ROOT / 'outputs/reference_intervention_pilot_v1'
    manifest = json.loads((out / 'manifest.json').read_text())
    cache = out / 'latents'
    cache.mkdir(exist_ok=False)
    model = load_model()
    model.first_stage_model.cuda()
    start = time.monotonic()
    records = []
    rng = np.random.default_rng(manifest['seed'])
    cases = manifest['cases']
    def read(row):
        assert sha(row['image_path']) == row['source_sha256']
        with Image.open(row['image_path']) as image:
            return image.convert('RGB').resize((512, 512), Image.Resampling.BICUBIC)
    def encode(image):
        array = np.asarray(image).copy()
        x = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).cuda().float() / 127.5 - 1
        with torch.no_grad():
            return model.encode_first_stage(x).cpu()
    for index, case in enumerate(cases):
        assert case['identity'] not in manifest['excluded_identities']
        target = read(case['target'])
        degraded = target.resize((64, 64), Image.Resampling.BICUBIC).resize((512, 512), Image.Resampling.BICUBIC)
        degraded = Image.fromarray(np.clip(np.asarray(degraded).astype(float) + rng.normal(0, 8, (512, 512, 3)), 0, 255).astype(np.uint8))
        refs = [read(row) for row in case['references']]
        donor_index = (index + 1) % 16 if case['role'] == 'train' else 16 + ((index - 16 + 1) % 4)
        donor = read(cases[donor_index]['references'][0])
        corrupted = refs[0].copy()
        corrupted.paste(donor.crop((128, 128, 384, 384)), (128, 128))
        bundle = dict(target=encode(target) * model.scale_factor, lq=encode(degraded),
                      references=torch.cat([encode(ref) for ref in refs]), corrupt_reference=encode(corrupted))
        path = cache / (case['identity'] + '.pt')
        torch.save(bundle, path)
        preview = out / 'images' / case['identity']
        preview.mkdir(parents=True)
        target.save(preview/'target.png'); degraded.save(preview/'input.png'); corrupted.save(preview/'corrupted_reference.png')
        records.append(dict(identity=case['identity'], role=case['role'], sha256=sha(path), donor_identity=cases[donor_index]['identity']))
        print(f'Encoded {index+1}/{len(cases)}', flush=True)
    (out/'cache_receipt.json').write_text(json.dumps(dict(manifest_sha256=sha(out/'manifest.json'),
        encoder_script_sha256=sha(__file__), core_sha256=sha(ROOT/'scripts/reference_intervention_core.py'),
        records=records, elapsed_seconds=time.monotonic()-start, precision='float32 VQ',
        reference_encoding='individually encoded; differs from author concatenated-width image encoding'), indent=2))


if __name__ == '__main__':
    main()
