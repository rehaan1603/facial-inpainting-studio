"""Studio postprocessing in the existing OpenCV environment; no model training."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from reference_blending import harmonize_reference


def main():
    parser = argparse.ArgumentParser()
    for name in ['input', 'prediction', 'mask', 'output', 'metadata']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {args.input.resolve(), args.prediction.resolve(), args.mask.resolve()}:
        raise ValueError('Keep the input, mask and raw prediction intact.')
    cv2.setNumThreads(1)
    observed = np.asarray(Image.open(args.input).convert('RGB'))
    generated = np.asarray(Image.open(args.prediction).convert('RGB'))
    mask = np.asarray(Image.open(args.mask).convert('L')) >= 128
    result, metadata = harmonize_reference(generated, observed, mask, mode='poisson')
    if not np.array_equal(result[~mask], observed[~mask]):
        raise RuntimeError('Edge matching changed reliable pixels.')
    Image.fromarray(result).save(args.output)
    args.metadata.write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
