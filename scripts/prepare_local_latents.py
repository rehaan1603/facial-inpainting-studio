"""Encode aligned runtime references in SDXL's own feature space on CPU."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha,write_new
from src.local_correspondence.features import similarity_transform,BOXES
from src.local_correspondence.latent_fusion import region_weights,fuse_latents
from src.reference_selection.reference_analyzer import ReferenceAnalyzer,load_rgb
from src.degradation.face_region_masks import region_masks
from distortion_aware_study import BASE,RESERVED


def main():
    import torch,cv2
    from diffusers import AutoencoderKL
    from scipy.ndimage import gaussian_filter
    torch.set_num_threads(2);cv2.setNumThreads(2)
    out=ROOT/'outputs/local_latent_v1';out.mkdir(exist_ok=False)
    cache=Path(json.loads((ROOT/'configs/local.json').read_text())['cache'])
    vae=AutoencoderKL.from_pretrained(str(cache/'osor_models/sdxl/vae'),local_files_only=True,torch_dtype=torch.float32).eval().requires_grad_(False)
    detector=ReferenceAnalyzer().detector
    features=json.loads((ROOT/'outputs/local_correspondence_v1/manifest.json').read_text())
    cases=[c for c in json.loads((BASE/'inference_manifest.json').read_text())['cases'] if c['kind']=='mixed']
    protocol={'version':1,'date':'2026-09-20','scope':'Four observed development identities, mixed damage, two seeds; mechanism diagnostic, no promotion',
              'policies':['global','regional_global','single','equal','quality','damage'],'seeds':[17,29],
              'strength':.99,'scale':.8,'steps':30,'blend':'poisson','latent_injection_gain':.25,
              'weights':'Single argmax regional quality; equal uniform; quality softmax(q/.15); damage softmax((q+.25*compatibility*reliability)/.15). q includes in-bounds and alignment confidence.',
              'alignment':'Five-point least-squares similarity from each reference to damaged target; no clean target geometry',
              'feature_space':'Frozen native SDXL VAE posterior means, scaling_factor applied, 4x64x64; region maps on damaged target',
              'injection':'After first unchanged denoising step, blend masked-image conditioning latents inside damage with aligned fused features, gain .25. Same global FaceID tokens in every arm. No output reference-pixel pasting.',
              'controls':'Global and prior global-regional routing on identical inputs; zero-gain equivalence tested separately',
              'limitations':['Out-of-training-distribution context for inpainting UNet','No 3D pose or expression correction','Heuristic visibility and confidence','Only mixed damage in generative diagnostic','AlexNet local compatibility shares backbone with LPIPS; final FaceNet remains separate'],
              'trainable_parameters':0,'final_test_used':False}
    write_new(ROOT/'research/protocols/local_latent_v1.json',protocol)
    def encode(image):
        tensor=torch.from_numpy(image.copy()).permute(2,0,1).float()[None]/127.5-1
        with torch.inference_mode():return (vae.encode(tensor).latent_dist.mean*vae.config.scaling_factor)[0].numpy()
    rows=[]
    for c in cases:
        assert c['identity'] not in RESERVED
        row={'case_id':c['case_id'],'identity':c['identity'],'status':'failed'}
        try:
            observed=load_rgb(c['observed']);mask=load_rgb(c['mask'])[:,:,0]>=128
            faces=detector.get(observed[:,:,::-1].copy())
            if len(faces)!=1:raise ValueError('Damaged target geometry unavailable')
            face=faces[0];local=next(r for r in features['rows'] if r['case_id']==c['case_id'])
            if local['status']!='complete':raise ValueError('Local correspondence unavailable')
            assert sha(local['features'])==local['features_sha256']
            bank=np.load(local['features']);ref_features=bank['references'];observed_features=bank['observed']
            norms=lambda x:x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)
            compatibility=np.sum(norms(ref_features)*norms(observed_features)[None],axis=-1)
            reliability=np.array([1-local['observed']['regions'][r]['damage_fraction'] for r in BOXES])
            quality=[];latents=[];alignments=[]
            for i,ref in enumerate(c['references']):
                assert sha(ref['path'])==ref['sha256']
                image=load_rgb(ref['path']);detected=detector.get(image[:,:,::-1].copy())
                if len(detected)!=1:raise ValueError('Reference geometry unavailable')
                matrix,error=similarity_transform(detected[0].kps,face.kps)
                warped=cv2.warpAffine(image,matrix,(512,512),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                coverage=cv2.warpAffine(np.ones(image.shape[:2],np.uint8),matrix,(512,512),flags=cv2.INTER_NEAREST)>0
                latents.append(encode(warped));alignments.append({'matrix':matrix.tolist(),'normalized_residual':error,'damaged_coverage':float(coverage[mask].mean())})
                quality.append([local['references'][i]['regions'][r]['quality']*local['references'][i]['regions'][r]['in_bounds_fraction']*np.exp(-error/.1) for r in BOXES])
            basis=np.stack([cv2.resize(gaussian_filter(m.astype(float),8),(64,64),interpolation=cv2.INTER_AREA) for m in region_masks(mask.shape,face.bbox,face.kps).values()])
            gate=cv2.resize(mask.astype('float32'),(64,64),interpolation=cv2.INTER_NEAREST)*.25
            folder=out/c['case_id'];folder.mkdir();payload={'reference_latents':np.stack(latents),'basis':basis,'gate':gate}
            weight_records={}
            for policy in ['single','equal','quality','damage']:
                weights=region_weights(np.array(quality),compatibility,reliability,policy)
                fused,spatial=fuse_latents(np.stack(latents),weights,basis)
                payload[policy]=fused;weight_records[policy]=weights.tolist()
            np.savez_compressed(folder/'latents.npz',**payload)
            row.update(status='complete',bank=str(folder/'latents.npz'),bank_sha256=sha(folder/'latents.npz'),alignment=alignments,region_weights=weight_records,
                       observed_sha256=c['observed_sha256'],reference_sha256=[r['sha256'] for r in c['references']])
        except Exception as error:row['error']=f'{type(error).__name__}: {error}'
        rows.append(row);print('NATIVE LOCAL FEATURES',len(rows),'/4',row['status'],row.get('error',''),flush=True)
    signature={'sources':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/local_correspondence/features.py',ROOT/'src/local_correspondence/latent_fusion.py']},
               'protocol_sha256':sha(ROOT/'research/protocols/local_latent_v1.json'),'local_feature_manifest_sha256':sha(ROOT/'outputs/local_correspondence_v1/manifest.json'),
               'vae_files':{str(p.relative_to(cache)):sha(p) for p in (cache/'osor_models/sdxl/vae').iterdir() if p.is_file()}}
    write_new(out/'feature_manifest.json',{'signature':signature,'rows':rows,'final_test_used':False})


if __name__=='__main__':main()
