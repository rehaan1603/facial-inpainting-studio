"""Persist a new case without overwriting a previous experiment."""
from pathlib import Path
from PIL import Image
from src.research_integrity import sha, write_new

def save_case(folder, observed, mask, metadata):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    Image.fromarray(observed).save(folder/'observed.png')
    Image.fromarray(mask.astype('uint8')*255).save(folder/'mask.png')
    metadata=dict(metadata,observed_sha256=sha(folder/'observed.png'),mask_sha256=sha(folder/'mask.png'))
    write_new(folder/'distortion.json',metadata)
    return metadata
