"""Studio-only reference validation/framing before the frozen ReFLDM runner.

Run in reference_env_v2 for CPU face detection. The separate ReFLDM process is
started only after validation; it is the sole process that loads a GPU model.
"""
import argparse
import gc
import json
import subprocess
from pathlib import Path

from atomic_records import write_json
from studio_reference_preparation import prepare_references

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['image', 'mask', 'output', 'progress']:
        parser.add_argument('--' + name, type=Path, required=name != 'progress')
    parser.add_argument('--references', type=Path, nargs='+', required=True)
    args = parser.parse_args()

    def progress(message, **extra):
        print(message, flush=True)
        if args.progress:
            write_json(args.progress, {'message': message, **extra}, required=False)

    try:
        if not 3 <= len(args.references) <= 4:
            raise ValueError('Provide 3–4 same-person reference photos.')
        destinations = [args.output, args.output.with_suffix('.json'), args.output.with_name(args.output.stem + '_raw.png')]
        inputs = {path.resolve() for path in [args.image, args.mask, *args.references]}
        if any(path.resolve() in inputs or path.exists() for path in destinations):
            raise ValueError('Use a new output path that does not overwrite an input or an earlier result.')
        cache = Path(json.loads((ROOT / 'configs/local.json').read_text())['cache'])
        python = cache / 'refldm_env_v1/Scripts/python.exe'
        face_dir = cache / 'reference_models/insightface/models/buffalo_l'
        if not python.is_file():
            raise ValueError('The experimental restoration environment is not installed.')
        if not face_dir.is_dir():
            raise ValueError('The reference face detector is not installed.')
        progress('Checking faces and framing the restoration reference photos…')
        import cv2
        from insightface.app import FaceAnalysis
        cv2.setNumThreads(1)
        analyser = FaceAnalysis(name=str(face_dir), allowed_modules=['detection'], providers=['CPUExecutionProvider'])
        analyser.prepare(ctx_id=-1, det_size=(640, 640))
        prepared, entries = prepare_references(args.references, args.output.with_name(args.output.stem + '_reference_preparation'), analyser)
        del analyser
        gc.collect()
        write_json(args.output.with_name(args.output.stem + '_reference_preparation.json'), {'references': entries})
        progress('Restoring partial facial damage with the validated reference photos…')
        command = [str(python), str(ROOT / 'scripts/studio_inference_worker.py'), str(ROOT / 'scripts/refldm_restore.py'),
                   '--image', str(args.image), '--mask', str(args.mask), '--references', *[str(path) for path in prepared], '--output', str(args.output)]
        child_log = args.output.with_name(args.output.stem + '_restoration.log')
        # Windows detached processes cannot reliably inherit an implicit console
        # stream. Give the child an explicit file, including startup tracebacks.
        with child_log.open('w', encoding='utf-8') as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=log,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if completed.returncode:
            print(child_log.read_text(encoding='utf-8', errors='replace')[-8000:], flush=True)
            raise ValueError(f'Partial-damage restoration failed (exit code {completed.returncode}). See the saved inference log.')
        metadata = json.loads(args.output.with_suffix('.json').read_text())
        metadata.update(
            studio_reference_preparation=entries,
            original_reference_sha256=[entry['original_sha256'] for entry in entries],
            studio_preprocessing='Single-face CPU validation; rectangular references receive a square face crop without stretching. Already-square PNG inputs remain unchanged.',
            preparation_quality_scope='Framing and validation fix only; not verified improvement in identity accuracy or recovery of the true face.',
        )
        write_json(args.output.with_suffix('.json'), metadata)
        progress('Partial-damage restoration ready')
    except Exception as error:
        progress('Restoration failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
