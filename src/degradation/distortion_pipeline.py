"""Seeded synthetic damage. Clean RGB is used only to construct evaluation cases."""
import io
import numpy as np
from PIL import Image, ImageFilter
from .severity import SEVERITIES
from .face_region_masks import damage_mask

KINDS=('removal','gaussian_blur','motion_blur','noise','jpeg','downsample','mixed','illumination')

def degrade(rgb, kind='removal', severity='medium', seed=17, mask=None, region='eyes', bbox=None, landmarks=None):
    rgb=np.asarray(rgb)
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('Expected uint8 RGB')
    if kind not in KINDS or severity not in SEVERITIES:raise ValueError('Unknown distortion/severity')
    mask=damage_mask(rgb.shape[:2],region,severity,seed,bbox,landmarks) if mask is None else np.asarray(mask,dtype=bool)
    if mask.shape!=rgb.shape[:2] or not mask.any():raise ValueError('Nonempty matching mask required')
    p=SEVERITIES[severity];rng=np.random.default_rng(seed);im=Image.fromarray(rgb)
    if kind=='removal':damaged=np.full_like(rgb,[90,100,110])
    elif kind=='gaussian_blur':damaged=np.asarray(im.filter(ImageFilter.GaussianBlur(p['sigma'])))
    elif kind=='motion_blur':
        from scipy.ndimage import uniform_filter1d
        damaged=np.rint(uniform_filter1d(rgb.astype(float),p['motion'],axis=1)).astype('uint8')
    elif kind=='noise':damaged=np.clip(rgb.astype(float)+rng.normal(0,p['noise'],rgb.shape),0,255).astype('uint8')
    elif kind=='illumination':damaged=np.clip(rgb.astype(float)*{'mild':.8,'medium':.55,'severe':.3}[severity],0,255).astype('uint8')
    else:
        if kind in ('downsample','mixed'):
            im=im.resize((max(1,im.width//p['downsample']),max(1,im.height//p['downsample'])),Image.Resampling.BOX).resize(im.size,Image.Resampling.BICUBIC)
        if kind=='mixed':
            im=im.filter(ImageFilter.GaussianBlur(p['sigma']))
            im=Image.fromarray(np.clip(np.asarray(im).astype(float)+rng.normal(0,p['noise'],rgb.shape),0,255).astype('uint8'))
        if kind in ('jpeg','mixed'):
            buf=io.BytesIO();im.save(buf,format='JPEG',quality=p['jpeg']);buf.seek(0)
            with Image.open(buf) as decoded:im=decoded.convert('RGB').copy()
        damaged=np.asarray(im)
    result=rgb.copy();result[mask]=damaged[mask]
    return result,mask,{'kind':kind,'severity':severity,'seed':seed,'region':region,
                      'parameters':p,'mask_fraction':float(mask.mean()),'outside_mask_unchanged':True,
                      'construction_only_clean_input':True}
