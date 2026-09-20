"""Experimental CLI with an explicit user-editable observation-confidence map."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.preservation.confidence import preserve_observation
from src.research_integrity import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['image','mask','confidence','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--references',type=Path,nargs='+',required=True);p.add_argument('--strength',type=float,required=True);p.add_argument('--scale',type=float,default=.8);p.add_argument('--seed',type=int,default=17)
    a=p.parse_args();base=a.output.with_name(a.output.stem+'_base.png')
    inputs={f.resolve() for f in [a.image,a.mask,a.confidence,*a.references]}
    destinations=[a.output,a.output.with_suffix('.json'),base,base.with_suffix('.json'),*[base.with_name(base.stem+s) for s in ['_raw.png','_hard.png','_input.png','_effective_mask.png']]]
    if any(f.exists() or f.resolve() in inputs for f in destinations):raise ValueError('Use new output paths; inputs and previous results are protected')
    with Image.open(a.image) as image:original_size=ImageOps.exif_transpose(image).size
    with Image.open(a.mask) as im:
        if im.size!=original_size:raise ValueError('Mask must match image')
        mask=np.asarray(im.convert('L').resize((512,512),Image.Resampling.NEAREST))>=128
    with Image.open(a.confidence) as im:
        if im.size!=original_size:raise ValueError('Confidence map must match image')
        confidence=np.asarray(im.convert('L').resize((512,512),Image.Resampling.NEAREST),dtype=float)/255
    if not np.all(confidence[~mask]==1):raise ValueError('Outside-mask confidence must be white (255)')
    from reference_inpaint import reconstruct
    metadata=reconstruct(a.image,a.mask,a.references,base,seed=a.seed,strength=a.strength,scale=a.scale)
    with Image.open(base.with_name(base.stem+'_input.png')) as im:observed=np.asarray(im.convert('RGB'))
    with Image.open(base) as im:generated=np.asarray(im.convert('RGB'))
    result=preserve_observation(observed,generated,mask,confidence);Image.fromarray(result).save(a.output)
    metadata.update(experimental_mode='user_confidence_preservation',confidence_sha256=sha(a.confidence),parent_output_sha256=sha(base),result_sha256=sha(a.output),scope='Uncalibrated user-supplied confidence; not blind restoration or proven improvement')
    a.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')


if __name__=='__main__':main()
