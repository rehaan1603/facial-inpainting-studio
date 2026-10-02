"""Explicit ROI diagnostics; crop LPIPS is not falsely labelled masked LPIPS."""
import numpy as np
import torch
from scipy.ndimage import binary_erosion


def pixel_statistics(output, target, region):
    count=int(region.sum())
    if not count:
        return dict(pixels=0,mae=None,psnr=None,psnr_status='empty_region')
    error=(output.astype(np.float64)-target.astype(np.float64))[region]/255
    mse=float(np.square(error).mean())
    return dict(pixels=count,mae=float(np.abs(error).mean()),psnr=-10*np.log10(mse) if mse else None,
                psnr_status='finite' if mse else 'perfect_infinite')


def region_statistics(metrics, output, target, region, ssim_map=None):
    result=pixel_statistics(output,target,region)
    valid=binary_erosion(region,structure=np.ones((11,11),bool),border_value=0)
    result['ssim_valid_centres']=int(valid.sum())
    result['ssim']=float(ssim_map[valid].mean()) if ssim_map is not None and valid.any() else None
    result['ssim_status']='valid_windows' if result['ssim'] is not None else 'no_fully_contained_11x11_window'
    result['bbox_lpips']=None
    if region.any():
        ys,xs=np.nonzero(region)
        # Fixed ten-pixel context; bounds are shared across every method in this case.
        y0,y1=max(0,int(ys.min())-10),min(region.shape[0],int(ys.max())+11)
        x0,x1=max(0,int(xs.min())-10),min(region.shape[1],int(xs.max())+11)
        result['bbox_xyxy']=[x0,y0,x1,y1]
        if min(y1-y0,x1-x0)>=32:
            with torch.inference_mode():
                result['bbox_lpips']=float(metrics.lpips(metrics.tensor(output[y0:y1,x0:x1])*2-1,
                    metrics.tensor(target[y0:y1,x0:x1])*2-1).item())
            result['bbox_lpips_status']='crop_with_context_not_pixel_masked'
        else:result['bbox_lpips_status']='crop_too_small_no_resizing'
    else:result['bbox_lpips_status']='empty_region'
    return result


def score_regions(metrics, output, target, observed, mask, components):
    assert output.shape==target.shape==observed.shape
    assert mask.shape==target.shape[:2] and mask.dtype==bool
    a,b=output.astype(np.float64)/255,target.astype(np.float64)/255
    _,ssim_map=metrics.ssim(a,b,data_range=1,channel_axis=2,gaussian_weights=True,
        sigma=1.5,use_sample_covariance=False,win_size=11,full=True)
    if ssim_map.ndim==3:ssim_map=ssim_map.mean(2)
    regions={'whole':np.ones(mask.shape,bool),'masked':mask,'visible':~mask}
    for name,component in components.items():regions[name]=component & mask
    result={name:region_statistics(metrics,output,target,region,ssim_map) for name,region in regions.items()}
    unchanged=~mask
    result['visible']['changed_pixels_vs_input']=int(np.any(output!=observed,axis=2)[unchanged].sum())
    result['visible']['vs_input']=pixel_statistics(output,observed,unchanged)
    return result
