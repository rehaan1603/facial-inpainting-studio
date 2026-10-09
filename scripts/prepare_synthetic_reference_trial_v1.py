"""Persist all generated assets and build a damaged-input-only diagnostic."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/synthetic_references_v1'
PACK = Path.home() / 'Downloads/Human dataset/Ready to test - Human Faces v1'
DELIVERY = PACK / 'Synthetic references - experimental 2026-10-02'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    receipt = json.loads((OUT/'asset_sources.json').read_text())
    inference = json.loads((PACK/'inference_manifest.json').read_text())
    DELIVERY.mkdir(exist_ok=False)
    cases = []
    for person in range(1, 5):
        case = next(c for c in inference['cases'] if c['case_id'] == f'photo_{person:02d}_central_missing')
        refs = []
        folder = OUT/'references'/f'photo_{person:02d}'
        folder.mkdir(parents=True, exist_ok=False)
        delivered = DELIVERY/f'Photo {person:02d}'
        delivered.mkdir()
        for name in ['input', 'mask']:
            item = case[name]
            assert sha(item['path']) == item['sha256']
            shutil.copy2(item['path'], delivered/(name+'.png'))
        for row in receipt['rows']:
            if row['person'] != person or row['status'] != 'generated':
                continue
            assert Path(row['source_input']).resolve() == Path(case['input']['path']).resolve()
            original = Path(row['generated_path'])
            filename = f"view_{row['view']:02d}_SYNTHETIC.png"
            dest = folder/filename
            shutil.copy2(original, dest)
            shutil.copy2(dest, delivered/filename)
            refs.append(dict(path=str(dest.resolve()), sha256=sha(dest), view=row['view']))
        assert len(refs) >= 2
        cases.append(dict(case_id=case['case_id'], input=case['input'], mask=case['mask'], references=refs))
    protocol = dict(date='2026-10-02', scope='Four-photo exploratory development diagnostic; synthetic references, not genuine observations',
                    generation_tool='built-in image_gen; exact model revision and seed unavailable',
                    reference_attempts=12, generated=11, blocked=1,
                    synthesis_policy='Only damaged input supplied; outputs retained without target-based selection',
                    clean_target_used_in_synthesis=False, arms=['reference_off', 'first_synthetic', 'all_synthetic'],
                    model='Existing SDXL inpainting + FaceID Portrait', seed=17, steps=30, strength=1.,
                    model_resolution=512, guidance=5., adapter_scale=.8, off_scale=0.,
                    preprocessing='Masked RGB neutralized to 127 for all three matched SDXL arms',
                    composition='Existing Poisson; exact original visible pixels preserved',
                    historical_comparators='Saved LaMa and edge-matched ResShift outputs; different backbones, not causal reference ablations',
                    asset_sources_sha256=sha(OUT/'asset_sources.json'), cases=cases)
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2))
    shutil.copy2(OUT/'asset_sources.json', DELIVERY/'PROMPTS AND GENERATION RECORD.json')
    (DELIVERY/'READ FIRST.txt').write_text('''EXPERIMENTAL SYNTHETIC REFERENCES
These views were generated using only each masked input, never its clean comparison.
Eyes and other missing details are imagined, not verified photographs of the person.
There are 11 images from 12 attempts: Photo 01 has two views after a tool block;
Photos 02–04 each have three. No failed view was silently replaced.
Requested left/right angles were not always faithfully produced. These are not
independent identity observations or extra statistically independent test samples.
Use only the matching Photo folder's input, mask and references together.
The accompanying experiment compares reference-off, first-view and all-view SDXL
inpainting, plus saved ResShift/LaMa baselines. All results will be retained.
Do not train on these views or interpret them as research ground truth.
''')
    print(json.dumps({'references':11,'cases':4,'delivery':str(DELIVERY)}))


if __name__ == '__main__':
    main()
