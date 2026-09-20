"""Fixed landmark normalization and frozen spatial CNN region descriptors."""
import json
from pathlib import Path
import numpy as np
from src.research_integrity import ROOT,sha

CANONICAL = np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],[41.5493,92.3655],[70.7299,92.2041]],dtype=np.float64)*2
BOXES = {'left_eye':(48,80,105,122),'right_eye':(119,80,176,122),
         'nose':(88,116,138,165),'mouth':(66,163,158,205),
         'left_cheek':(35,121,85,171),'right_cheek':(139,121,189,171),
         'forehead':(58,38,165,79),'chin':(76,199,150,224)}


def similarity_transform(source,destination=CANONICAL):
    source,destination=np.asarray(source,float),np.asarray(destination,float)
    if source.shape!=(5,2) or destination.shape!=(5,2) or not np.isfinite(source).all():
        raise ValueError('Five finite corresponding landmarks required')
    a,b=source-source.mean(0),destination-destination.mean(0)
    variance=np.sum(a*a)
    if variance<1e-8:raise ValueError('Degenerate landmarks')
    u,s,vt=np.linalg.svd(a.T@b);correction=np.eye(2);correction[-1,-1]=np.sign(np.linalg.det(u@vt))
    rotation=u@correction@vt;scale=float(np.sum(s*np.diag(correction))/variance)
    if scale<=0:raise ValueError('Invalid scale')
    linear=scale*rotation;translation=destination.mean(0)-source.mean(0)@linear
    matrix=np.concatenate([linear.T,translation[:,None]],axis=1)
    residual=np.sqrt(np.mean(np.sum((source@linear+translation-destination)**2,axis=1)))
    return matrix,float(residual/np.linalg.norm(destination[0]-destination[1]))


def warp(rgb,matrix,nearest=False):
    import cv2
    return cv2.warpAffine(rgb,matrix,(224,224),flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)


class LocalEncoder:
    def __init__(self):
        import torch
        from torchvision.models import alexnet
        torch.set_num_threads(2)
        cache=Path(json.loads((ROOT/'configs/local.json').read_text())['cache'])
        weight=cache/'torch/checkpoints/alexnet-owt-7be5be79.pth'
        network=alexnet(weights=None);network.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True))
        self.network=network.features[:5].eval().requires_grad_(False)
        self.weight_sha256=sha(weight)
        self.parameter_count=sum(p.numel() for p in self.network.parameters())

    def encode(self,aligned):
        import torch
        import torch.nn.functional as F
        if aligned.shape!=(224,224,3):raise ValueError('Canonical 224 RGB required')
        t=torch.from_numpy(aligned.copy()).permute(2,0,1).float()[None]/255
        t=(t-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
        with torch.inference_mode():
            features=self.network(t);h,w=features.shape[-2:];regions=[]
            for x0,y0,x1,y1 in BOXES.values():
                crop=features[:,:,int(y0*h/224):max(int(y1*h/224),int(y0*h/224)+1),int(x0*w/224):max(int(x1*w/224),int(x0*w/224)+1)]
                regions.append(F.adaptive_avg_pool2d(crop,(3,3)).flatten().numpy())
        return np.stack(regions)


def extract(rgb,face,encoder,mask=None):
    from src.reference_selection.quality_estimator import quality
    from src.reference_selection.pose_estimator import pose_proxy
    matrix,residual=similarity_transform(face.kps)
    aligned=warp(rgb,matrix);valid=warp(np.ones(rgb.shape[:2],np.uint8),matrix,True)>0
    aligned_mask=warp(np.asarray(mask,np.uint8),matrix,True)>0 if mask is not None else np.zeros((224,224),bool)
    records={}
    for name,(x0,y0,x1,y1) in BOXES.items():
        area=np.zeros((224,224),bool);area[y0:y1,x0:x1]=True
        records[name]={'quality':quality(aligned,area)['quality'],'in_bounds_fraction':float(valid[area].mean()),
                       'damage_fraction':float(aligned_mask[area].mean()),'alignment_confidence':float(np.exp(-residual/.1))}
    return {'features':encoder.encode(aligned),'global_identity':np.asarray(face.normed_embedding),
            'aligned':aligned,'matrix':matrix,'normalized_landmark_residual':residual,
            'pose_proxy':pose_proxy(face.kps),'regions':records,
            'limitation':'2D similarity normalization cannot correct 3D self-occlusion or expression; visibility is in-bounds only'}
