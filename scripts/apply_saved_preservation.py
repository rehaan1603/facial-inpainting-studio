"""Apply an editable evidence map to a verified saved reconstruction; no model run."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.preservation.confidence import preserve_observation
from src.research_integrity import sha

def apply(image,mask,confidence,generated,metadata,output):
    paths=[Path(p) for p in [image,mask,confidence,generated,metadata]]
    image,mask,confidence,generated,metadata=paths;output=Path(output)
    if output.suffix.lower()!='.png':raise ValueError('Output must be a PNG')
    destinations=[output,output.with_suffix('.json')]
    if any(p.exists() or p.resolve() in {x.resolve() for x in paths} for p in destinations):
        raise ValueError('Use a new output path; inputs and existing results are protected')
    record=json.loads(metadata.read_text(encoding='utf-8'))
    for key,path in [('input_sha256',image),('mask_sha256',mask),('result_sha256',generated)]:
        if record.get(key)!=sha(path):raise ValueError(f'Saved generation does not match {key}')
    with Image.open(image) as im:source=ImageOps.exif_transpose(im).convert('RGB');size=source.size;observed=np.asarray(source.resize((512,512),Image.Resampling.LANCZOS))
    with Image.open(mask) as im:
        if im.size!=size:raise ValueError('Mask must match the oriented image')
        binary=np.asarray(im.convert('L').resize((512,512),Image.Resampling.NEAREST))>=128
    with Image.open(confidence) as im:
        if im.size!=size:raise ValueError('Confidence must match the oriented image')
        weights=np.asarray(im.convert('L').resize((512,512),Image.Resampling.NEAREST),dtype=float)/255
    with Image.open(generated) as im:reconstruction=np.asarray(im.convert('RGB'))
    if reconstruction.shape!=observed.shape:raise ValueError('Expected a 512-pixel saved reconstruction')
    if not np.array_equal(observed[~binary],reconstruction[~binary]):raise ValueError('Saved output changed reliable pixels')
    result=preserve_observation(observed,reconstruction,binary,weights)
    output.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(result).save(output)
    receipt={'mode':'saved_reconstruction_preservation','new_generation':False,'input_sha256':sha(image),'mask_sha256':sha(mask),'confidence_sha256':sha(confidence),'parent_output_sha256':sha(generated),'parent_metadata_sha256':sha(metadata),'result_sha256':sha(output),'known_pixels_unchanged':bool(np.array_equal(result[~binary],observed[~binary])),'scope':'User-supplied evidence blending; no quality or true-face recovery guarantee.'}
    output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['image','mask','confidence','generated','metadata','output']:parser.add_argument('--'+name,required=True,type=Path)
    print(json.dumps(apply(**vars(parser.parse_args())),indent=2))
if __name__=='__main__':main()
