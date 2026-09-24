"""Studio missing-area inference; frozen research generator remains unchanged.

Remove the occluder's appearance before generation, then composite against the
actual observed image. Partial-damage evidence mode uses its separate pathway.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from reference_blending import harmonize_reference


def neutralize(observed, mask):
    result = observed.copy()
    result[mask] = 127
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['image', 'mask', 'output', 'progress']:
        p.add_argument('--'+name, type=Path, required=name!='progress')
    p.add_argument('--references', nargs='+', type=Path, required=True)
    p.add_argument('--blend', choices=['poisson', 'hard'], default='poisson')
    p.add_argument('--model-resolution', type=int, choices=[256,512,1024], default=512)
    p.add_argument('--keep-observation', action='store_true')
    p.add_argument('--strength', type=float, default=1)
    p.add_argument('--select', action='store_true')
    a = p.parse_args()
    try:
        from reference_inpaint import reconstruct, load_rgb, sha
        if a.model_resolution==256:
            from studio_reference_engine import reconstruct
        observed = np.asarray(load_rgb(a.image).resize((512,512), Image.Resampling.LANCZOS))
        mask = np.asarray(Image.open(a.mask).convert('L').resize((512,512), Image.Resampling.NEAREST)) >= 128
        refs = a.references
        selection = None
        if a.select:
            from mask_aware_inpaint import ReferenceAnalyzer, select
            selection = ReferenceAnalyzer().analyze(a.image, a.mask, refs)
            refs = select(selection, 'mask_aware', 17)
            selection.update(policy='mask_aware', selected_paths=refs)
            if not refs:
                raise ValueError('No usable references. Supply clear single-face photos.')
        conditioning = a.output.with_name(a.output.stem+'_conditioning.png')
        if a.output.exists() or conditioning.exists():
            raise ValueError('Use a new output path.')
        a.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(observed if a.keep_observation else neutralize(observed, mask)).save(conditioning)
        metadata = reconstruct(conditioning, a.mask, refs, a.output,
            strength=a.strength if a.keep_observation else 1, blend='hard', model_resolution=a.model_resolution, progress_path=a.progress)
        generated = np.asarray(Image.open(a.output.with_name(a.output.stem+'_raw.png')).convert('RGB'))
        result, blending = harmonize_reference(generated, observed, mask, a.blend)
        hard, _ = harmonize_reference(generated, observed, mask, 'hard')
        Image.fromarray(result).save(a.output)
        Image.fromarray(hard).save(a.output.with_name(a.output.stem+'_hard.png'))
        Image.fromarray(observed).save(a.output.with_name(a.output.stem+'_input.png'))
        metadata.update(input_sha256=sha(a.image), conditioning_sha256=sha(conditioning),
            result_sha256=sha(a.output), blending=blending,
            studio_preprocessing='Original observation retained; original visible pixels preserved.' if a.keep_observation else 'Masked RGB replaced by neutral 127; full denoising; original visible pixels preserved.',
            quality_scope='Functional correction; no guarantee of identity, expression or gaze recovery.')
        if selection is not None: metadata['reference_selection'] = selection
        a.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    except Exception as error:
        if a.progress:
            from atomic_records import write_json
            write_json(a.progress, {'error':str(error), 'message':'Reconstruction failed'}, required=False)
        raise


if __name__ == '__main__':
    main()
