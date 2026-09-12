"""Local 256-pixel research inference. Mask white=unknown. No uploads."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from refiner import MaskRefiner

ROOT=Path(__file__).resolve().parents[1]


class RefinerPredictor:
    def __init__(self,path):
        state=torch.load(path,map_location='cpu',weights_only=True)
        self.model=MaskRefiner(state['width']).cuda().eval();self.model.load_state_dict(state['model'],strict=True)
    @torch.inference_mode()
    def probability(self,observed,mask):
        x=torch.from_numpy(observed.transpose(2,0,1).copy()).cuda()[None]
        m=torch.from_numpy(mask.astype('float32')).cuda()[None,None]
        logits=self.model(F.interpolate(x,(128,128),mode='bilinear',align_corners=False),F.interpolate(m,(128,128),mode='nearest'))
        return F.interpolate(logits.sigmoid(),observed.shape[:2],mode='bilinear',align_corners=False)[0,0].cpu().numpy()


class Inpainter:
    def __init__(self,backbone):
        self.backbone=backbone;local=json.loads((ROOT/'configs/local.json').read_text())
        if backbone=='lama':self.model=torch.jit.load(str(Path(local['cache'])/'models/big-lama.pt'),map_location='cuda').eval()
        else:
            from resshift_adapter import ResShiftFace
            self.model=ResShiftFace()
    @torch.inference_mode()
    def __call__(self,observed,mask,seed=17):
        if not mask.any():return observed.copy()
        if self.backbone=='resshift':return self.model(observed,mask,seed=seed)
        x=torch.from_numpy(observed.transpose(2,0,1).copy()).cuda()[None];m=torch.from_numpy(mask.astype('float32')).cuda()[None,None]
        raw=self.model(x,m)[0].permute(1,2,0).cpu().numpy().clip(0,1)
        return np.where(mask[...,None],raw,observed)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--image',type=Path,required=True);p.add_argument('--mask',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--backbone',choices=['lama','resshift'],default='lama');p.add_argument('--refiner',type=Path);p.add_argument('--threshold',type=float,default=.5);p.add_argument('--seed',type=int,default=17);a=p.parse_args()
    if not 0<a.threshold<1:p.error('Threshold must be between 0 and 1')
    if a.output.resolve() in [a.image.resolve(),a.mask.resolve()]:p.error('Output must not overwrite an input')
    if a.output.suffix.lower()!='.png':p.error('Use a .png output to avoid lossy output encoding')
    torch.set_num_threads(4)
    with Image.open(a.image) as im:
        original_size=im.size;observed=np.array(im.convert('RGB').resize((256,256),Image.Resampling.LANCZOS)).astype('float32')/255
    with Image.open(a.mask) as im:
        if im.size!=original_size:p.error('Image and mask dimensions must match')
        supplied=np.array(im.convert('L').resize((256,256),Image.Resampling.NEAREST))>=128
    effective=supplied if a.refiner is None else RefinerPredictor(a.refiner).probability(observed,supplied)>=a.threshold
    pred=Inpainter(a.backbone)(observed,effective,a.seed)
    a.output.parent.mkdir(parents=True,exist_ok=True);Image.fromarray((pred*255).round().astype('uint8')).save(a.output)
    mask_path=a.output.with_name(a.output.stem+'_effective_mask.png');Image.fromarray(effective.astype('uint8')*255).save(mask_path)
    a.output.with_suffix('.json').write_text(json.dumps({'backbone':a.backbone,'refiner':str(a.refiner) if a.refiner else None,'refiner_sha256':hashlib.sha256(a.refiner.read_bytes()).hexdigest() if a.refiner else None,'threshold':a.threshold,'seed':a.seed,'input_size':original_size,'output_size':[256,256],'mask_white':'replace','scope':'Research inference; generated hidden content is not verified ground truth.'},indent=2));print(a.output)


if __name__=='__main__':main()
