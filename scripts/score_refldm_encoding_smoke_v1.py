"""Score paired encoding smoke with frozen local metrics, not author benchmark metrics."""
import json
import numpy as np
import torch
from extended_evaluation_metrics import ROOT,Metrics,read_rgb,sha
m=Metrics();out={}
a=ROOT/'outputs/refldm_author_smoke_v1';b=ROOT/'outputs/refldm_separate_encoding_smoke_v1'
ca=json.loads((a/'config.json').read_text());cb=json.loads((b/'config.json').read_text())
for k in ['identity','target','steps','seed','precision','references','lq_sha256','hq_sha256','guidance','output_conversion']:assert ca[k]==cb[k],k
truth=read_rgb(a/'hq.png');tv,td=m.features(truth)
for label,folder in [('author_rgb_encoding',a),('separate_encoding',b)]:
 receipt=json.loads((folder/'receipt.json').read_text());assert receipt['status']=='complete'
 rgb=read_rgb(folder/'result.png',receipt['result_sha256']);v,d=m.features(rgb)
 a=rgb.astype(np.float64)/255;b=truth.astype(np.float64)/255
 mse=float(np.square(a-b).mean());values={'psnr':float(-10*np.log10(mse)),'ssim':float(m.ssim(a,b,data_range=1,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False))}
 with torch.inference_mode():values['lpips']=float(m.lpips(m.tensor(rgb)*2-1,m.tensor(truth)*2-1))
 values.update({k+'_cosine':float(np.dot(x,tv[k])) if x is not None and tv[k] is not None else None for k,x in v.items()})
 out[label]={'metrics':values,'receipt':receipt,'target_detections':td,'output_detections':d}
result={'scope':'One already-observed identity, one official target, seed17; only reference encoding changes. Local metrics, not published benchmark reproduction.','config':ca,'results':out,'script_sha256':sha(__file__)}
(ROOT/'research/refldm_encoding_smoke_v1.json').write_text(json.dumps(result,indent=2));print(json.dumps(out,indent=2))
