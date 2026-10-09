"""Registered reference RGB detail, using only visible target anchors."""
import json,time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter,distance_transform_edt
from reference_inpaint import ROOT,CACHE,sha,load_rgb
from run_structure_transport import LandmarkExtractor,sanitized_observation
from src.preservation.structure_transport import build_target_shape,similarity_fit,transform_points

OUT=ROOT/'outputs/focused_eye_texture_v1';IDS=['263','6681','10149']

def aperture(points):
    width=np.linalg.norm(points[0]-points[3])
    if width<2:return 0.
    return float((np.linalg.norm(points[1]-points[5])+np.linalg.norm(points[2]-points[4]))/(2*width))

def apply(scaffold,observed,mask,warped,support,offset,alpha):
    corrected=np.clip(warped.astype(float)+np.asarray(offset),0,255)
    weight=(np.clip(support,0,1)*mask*alpha)[...,None]
    result=np.rint(scaffold*(1-weight)+corrected*weight).clip(0,255).astype('uint8')
    result[~mask]=observed[~mask]
    return result

def main():
    cfg=json.loads((ROOT/'outputs/focused_eye_guidance_v1/config.json').read_text())
    cfg.update(expected=3,runner_sha256=sha(__file__),protocol_sha256=sha(ROOT/'research/FOCUSED_EYE_TEXTURE_PREFLIGHT_V1.md'),
               mechanism='Registered reference RGB eye/brow detail on fixed CFG3 scaffold',training=False,
               aperture_min=.16,residual_max=.06,anchor_margin=8,minanchors=6,feather_sigma=2,transfer_strength=1.,
               maximum_colour_offset=24,minimum_visible_pixels=200,minimum_support_coverage=.8)
    cases={c['identity']:c for c in json.loads((ROOT/'outputs/masked_task_v1/cases.json').read_text()) if c['location']=='eyes'}
    assert sha(ROOT/'outputs/masked_task_v1/cases.json')==cfg['cases_sha256']
    controls={r['identity']:r for r in json.loads((ROOT/'outputs/focused_eye_guidance_v1/summary.json').read_text())['rows']}
    cfg['landmark_weights']={str(p):sha(p) for p in (CACHE/'reference_models/insightface/models/buffalo_l').glob('*.onnx')}
    for ident in IDS:
        for key in ['input','mask']:assert sha(cases[ident][key])==cases[ident][key+'_sha256']
        for r in cfg['references'][ident+'/clean']:assert sha(r['path'])==r['sha256'] and r['sha256']!=cases[ident]['target_sha256']
        assert sha(ROOT/'outputs/focused_eye_guidance_v1'/controls[ident]['file'])==controls[ident]['sha256']
    OUT.mkdir(exist_ok=False);(OUT/'config.json').write_text(json.dumps(cfg,indent=2))
    extractor=LandmarkExtractor(CACHE/'reference_models/insightface/models/buffalo_l')
    rows=[]
    for ident in IDS:
        start=time.monotonic();c=cases[ident];observed=np.asarray(load_rgb(c['input']));mask=np.asarray(Image.open(c['mask']))>127
        scaffold=np.asarray(load_rgb(ROOT/'outputs/focused_eye_guidance_v1'/controls[ident]['file']))
        assert np.array_equal(scaffold[~mask],observed[~mask]);result=scaffold.copy();details={'references':[]}
        row=dict(identity=ident,location='eyes',condition='clean',seed=17,file=f'{ident}_eyes_clean_17.png',mechanism_status='fallback')
        try:
            _,bbox=extractor.shape(scaffold);obs68=extractor.observed_shape(sanitized_observation(observed,mask),bbox)
            refs=[]
            for index,r in enumerate(cfg['references'][ident+'/clean']):
                record=dict(index=index,sha256=r['sha256'],eligible=False)
                try:
                    rgb=np.asarray(load_rgb(r['path']));shape,_=extractor.shape(rgb)
                    ears=[aperture(shape[36:42]),aperture(shape[42:48])];record['apertures']=ears
                    if min(ears)<cfg['aperture_min']:raise ValueError('Eye aperture below fixed threshold')
                    refs.append((index,rgb,shape,record))
                except Exception as exc:record['reason']=str(exc)
                details['references'].append(record)
            if not refs:raise ValueError('No reference has two eligible open eyes')
            _,metadata=build_target_shape(obs68,[r[2] for r in refs],mask,anchor_margin=8,minanchors=6)
            anchors=np.asarray(metadata['anchor_indices']);details['anchors']=metadata
            eligible=[]
            for index,rgb,shape,record in refs:
                matrix=similarity_fit(shape[anchors],obs68[anchors]);registered=transform_points(shape,matrix)
                residual=float(np.sqrt(np.square(registered[anchors]-obs68[anchors]).sum(1).mean())/(bbox[2]-bbox[0]))
                record.update(normalized_residual=residual,transform=matrix.tolist())
                if residual>cfg['residual_max']:record['reason']='Visible-anchor residual above threshold';continue
                record['eligible']=True;eligible.append((residual,index,rgb,registered,matrix))
            if not eligible:raise ValueError('No reference passes visible-anchor alignment')
            residual,index,rgb,points,matrix=min(eligible,key=lambda r:(r[0],r[1]))
            warped=cv2.warpAffine(rgb,matrix,(512,512),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            valid=cv2.warpAffine(np.ones(rgb.shape[:2],np.uint8)*255,matrix,(512,512),flags=cv2.INTER_NEAREST)>127
            support=np.zeros(mask.shape,np.float64);coverage=[]
            for eye,brow in [(points[36:42],points[17:22]),(points[42:48],points[22:27])]:
                group=np.concatenate([eye,brow]);lo=group.min(0);hi=group.max(0);center=(lo+hi)/2;radius=(hi-lo)*.55
                if min(radius)<2:raise ValueError('Degenerate eye/brow support')
                yy,xx=np.mgrid[:512,:512];ellipse=((xx-center[0])/radius[0])**2+((yy-center[1])/radius[1])**2<=1
                fraction=float((ellipse&mask&valid).sum()/max(1,ellipse.sum()));coverage.append(fraction)
                if fraction<cfg['minimum_support_coverage']:raise ValueError('Insufficient valid masked support')
                support=np.maximum(support,gaussian_filter(ellipse.astype(float),sigma=2))
            hull=np.zeros(mask.shape,np.uint8);cv2.fillConvexPoly(hull,cv2.convexHull(obs68[anchors].round().astype('int32')),1)
            safe=(hull>0)&(~mask)&valid&(distance_transform_edt(~mask)>=8)
            if safe.sum()<200:raise ValueError('Insufficient visible pixels for bounded colour offset')
            offset=np.clip(np.median(observed[safe].astype(float)-warped[safe].astype(float),axis=0),-24,24)
            zero=apply(scaffold,observed,mask,warped,support,offset,0.);assert np.array_equal(zero,scaffold)
            result=apply(scaffold,observed,mask,warped,support,offset,1.)
            details.update(selected_index=index,residual=residual,coverage=coverage,colour_offset=offset.tolist(),safe_visible_pixels=int(safe.sum()),zero_exact=True)
            np.savez_compressed(OUT/f'{ident}_geometry.npz',observed_anchors=obs68[anchors],registered_reference=points,transform=matrix,support=support,valid=valid)
            Image.fromarray(warped).save(OUT/f'{ident}_registered_reference.png')
            row['mechanism_status']='transferred'
        except Exception as exc:details['fallback_reason']=str(exc)
        assert np.array_equal(result[~mask],observed[~mask])
        p=OUT/row['file'];Image.fromarray(result).save(p);Image.fromarray(result).save(p.with_name(p.stem+'_raw.png'))
        details.update(scaffold_sha256=controls[ident]['sha256'],raw_role='Post-transfer image; not a new diffusion generation')
        p.with_suffix('.json').write_text(json.dumps(details,indent=2));row.update(status='complete',sha256=sha(p),raw_sha256=sha(p.with_name(p.stem+'_raw.png')),seconds=time.monotonic()-start,details=details)
        rows.append(row);(OUT/'rows.json').write_text(json.dumps(rows,indent=2));print('TEXTURE',ident,row['mechanism_status'],details.get('fallback_reason',''),flush=True)

if __name__=='__main__':main()
