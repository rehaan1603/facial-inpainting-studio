"""Estimate a restoration blend from corrupted reference proxies, not target truth.

The fitted weight is a constrained mixing coefficient, not a probability of accuracy.
"""
import numpy as np


def box(a, radius):
    if radius==0:return np.asarray(a,dtype=float).copy()
    a=np.asarray(a,dtype=float);p=np.pad(a,((radius,radius),(radius,radius)),mode='reflect')
    s=np.pad(p,((1,0),(1,0))).cumsum(0).cumsum(1);k=2*radius+1
    return (s[k:,k:]-s[:-k,k:]-s[k:,:-k]+s[:-k,:-k])/(k*k)


def features(observed, generated):
    if observed.shape!=generated.shape or observed.ndim!=3 or observed.shape[2]!=3:raise ValueError('Matching RGB images required')
    o=observed.astype(float)/255;g=generated.astype(float)/255
    gray=o.mean(2);outgray=g.mean(2);edit=np.abs(g-o).mean(2)
    mean=box(gray,2);contrast=np.sqrt(np.maximum(0,box(gray**2,2)-mean**2))
    outmean=box(outgray,2);outcontrast=np.sqrt(np.maximum(0,box(outgray**2,2)-outmean**2))
    dy,dx=np.gradient(gray);gradient=box(np.abs(dx)+np.abs(dy),2)
    return np.stack([edit,box(edit,2),contrast,outcontrast,gradient,
                     box(np.abs(gray-box(gray,1)),2),np.abs(box(outgray-gray,4)),mean],axis=-1)


def fit(proxies, ridge=.01):
    """Each proxy is (damaged reference, restored reference, original reference, mask)."""
    xs=[];weights=[];cross=[]
    for o,g,t,mask in proxies:
        if o.dtype!=np.uint8 or g.dtype!=np.uint8 or t.dtype!=np.uint8:raise ValueError('uint8 required')
        if o.shape!=t.shape or mask.shape!=o.shape[:2] or not mask.any():raise ValueError('Invalid proxy geometry')
        x=features(o,g)[mask][::4]
        delta=(g.astype(float)-o.astype(float))[mask][::4]/255
        desired=(t.astype(float)-o.astype(float))[mask][::4]/255
        xs.append(x);weights.append((delta*delta).sum(1));cross.append((delta*desired).sum(1))
    x=np.concatenate(xs);w=np.concatenate(weights);b=np.concatenate(cross)
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-5);x=np.column_stack([np.ones(len(x)),(x-mean)/scale])
    mass=float(w.sum())
    if mass<1e-12:return {'mean':mean.tolist(),'scale':scale.tolist(),'coefficients':[0.]*x.shape[1],'global_weight':0.,'samples':len(x),'ridge':ridge}
    penalty=np.eye(x.shape[1])*ridge;penalty[0,0]=0
    coef=np.linalg.solve((x.T*w)@x/mass+penalty,x.T@b/mass)
    return {'mean':mean.tolist(),'scale':scale.tolist(),'coefficients':coef.tolist(),
            'global_weight':float(np.clip(b.sum()/mass,0,1)),'samples':len(x),'ridge':ridge}


def mix(observed,generated,mask,weight):
    if mask.shape!=observed.shape[:2] or generated.shape!=observed.shape:raise ValueError('Invalid geometry')
    weight=np.broadcast_to(np.asarray(weight,float),mask.shape)
    if not np.isfinite(weight).all() or np.any((weight<0)|(weight>1)):raise ValueError('Invalid mixing weights')
    out=observed.copy();out[mask]=np.rint(observed[mask].astype(float)+weight[mask,None]*(generated[mask].astype(float)-observed[mask])).clip(0,255).astype('uint8')
    return out


def predict(observed,generated,mask,model):
    x=(features(observed,generated)-np.asarray(model['mean']))/np.asarray(model['scale']);coef=np.asarray(model['coefficients'])
    raw=np.clip(coef[0]+x@coef[1:],0,1)
    # Normalize smoothing at mask boundaries so outside pixels never set its weight.
    weight=box(raw*mask,2)/np.maximum(box(mask.astype(float),2),1e-8);weight=np.clip(weight,0,1)
    weight[~mask]=0
    return mix(observed,generated,mask,weight),weight
