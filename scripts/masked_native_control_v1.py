"""Native ResShift smoke control; separate resolution, not matched ReF-LDM."""
import json,time
import numpy as np
from PIL import Image
from reference_intervention_core import ROOT,sha
import resshift_adapter


def main():
    root=ROOT/'outputs/masked_task_v1';out=root/'native_resshift';out.mkdir(exist_ok=False)
    # Redirect the wrapper's provenance output into this experiment, preserving old reports.
    (out/'configs').mkdir();(out/'research').mkdir()
    (out/'configs/local.json').write_bytes((ROOT/'configs/local.json').read_bytes())
    resshift_adapter.ROOT=out
    model=resshift_adapter.ResShiftFace();rows=[]
    for c in json.loads((root/'cases.json').read_text()):
        assert sha(c['input'])==c['input_sha256'] and sha(c['mask'])==c['mask_sha256']
        original=np.asarray(Image.open(c['input']).convert('RGB'))
        fullmask=np.asarray(Image.open(c['mask']).convert('L'))>127
        # Max pooling preserves all supplied holes at native resolution.
        mask=fullmask.reshape(256,2,256,2).any(axis=(1,3))
        rgb=np.asarray(Image.fromarray(original).resize((256,256),Image.Resampling.BICUBIC)).copy()
        rgb[mask]=127
        start=time.monotonic();native=model(rgb.astype('float32')/255,mask,seed=17)
        native=(native*255).round().clip(0,255).astype('uint8')
        up=np.asarray(Image.fromarray(native).resize((512,512),Image.Resampling.BICUBIC))
        result=np.where(fullmask[...,None],up,original)
        assert np.array_equal(result[~fullmask],original[~fullmask])
        p=out/f'{c["identity"]}_{c["location"]}.png';Image.fromarray(result).save(p)
        Image.fromarray(native).save(out/f'{c["identity"]}_{c["location"]}_native256.png')
        rows.append(dict(identity=c['identity'],location=c['location'],file=p.name,sha256=sha(p),seconds=time.monotonic()-start,
                         seed=17,native_resolution=256,references_used=False,visible_pixels_exact=True))
        (out/'rows.json').write_text(json.dumps(rows,indent=2));print('NATIVE',len(rows),p.name,flush=True)


if __name__=='__main__':main()
