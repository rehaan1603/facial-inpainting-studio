"""Reference-only surrogate calibration. Target/gallery evaluation manifests are never opened."""
import json,sys,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha,write_new
from src.degradation.distortion_pipeline import degrade
from src.preservation.proxy_calibration import fit,predict,mix

def main():
    protocol=ROOT/'research/protocols/proxy_calibration_v1.json';cfg=json.loads(protocol.read_text())
    manifest=ROOT/'outputs/distortion_aware_v1/inference_manifest.json';allcases=json.loads(manifest.read_text())['cases']
    cases=[next(c for c in allcases if c['case_id']==name) for name in cfg['cases']]
    assert {c['identity'] for c in cases}=={'1306','2790','1043','787'}
    for c in cases:
        for path,digest in [(c['observed'],c['observed_sha256']),(c['mask'],c['mask_sha256'])]+[(r['path'],r['sha256']) for r in c['references']]:assert sha(path)==digest
        assert c['references'][0]['sha256'] not in {r['sha256'] for r in c['references'][1:]}
    out=ROOT/'outputs/proxy_calibration_v1';out.mkdir(exist_ok=False)
    cache=Path(json.loads((ROOT/'configs/local.json').read_text())['cache']);source=cache/'refldm_source_v1';weights=cache/'refldm_weights_v1'
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    assert commit=='af6690c19fdc6421802fd7996510bcfef259bfd1'
    patch=subprocess.check_output(['git','-C',str(source),'diff'],text=True)
    assert patch==(ROOT/'research/refldm_author_compatibility.patch').read_text()
    for f in json.loads((ROOT/'research/refldm_downloads_v1.json').read_text())['files']:assert sha(weights/f['name'])==f['sha256']
    write_new(out/'signature.json',{'protocol_sha256':sha(protocol),'runner_sha256':sha(__file__),
        'mechanism_sha256':sha(ROOT/'src/preservation/proxy_calibration.py'),'input_manifest_sha256':sha(manifest),
        'source_commit':commit,'patch_sha256':sha(ROOT/'research/refldm_author_compatibility.patch'),'torch':torch.__version__,
        'input_scope':'observations, masks, supplied references and previous target outputs only; no target/gallery truth'})
    sys.path.insert(0,str(source))
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.ddim import DDIMSampler
    from inference import read_and_normalize_image
    from torchvision.transforms.functional import to_pil_image
    config=OmegaConf.load(source/'configs/refldm.yaml');config.model.params.first_stage_config.params.ckpt_path=str(weights/'vqgan.ckpt')
    for k in ['ckpt_path','perceptual_loss_config']:config.model.params.pop(k,None)
    model=instantiate_from_config(config.model);keys=model.load_state_dict(torch.load(weights/'refldm.ckpt',map_location='cpu',weights_only=True),strict=False)
    if keys.missing_keys or keys.unexpected_keys:raise ValueError('Model parameter mismatch')
    model.cuda().eval();torch.set_grad_enabled(False);sampler=DDIMSampler(model,print_tqdm=False,schedule='uniform_trailing')
    shape=[model.model.diffusion_model.out_channels,model.image_size,model.image_size]
    baseline=json.loads((ROOT/'outputs/refldm_development_v1/comparison.json').read_text())['rows'];rows=[]
    for c in cases:
        dest=out/c['case_id'];dest.mkdir();common={k:c[k] for k in ['case_id','identity','kind']}
        observed=np.asarray(Image.open(c['observed']).convert('RGB'));mask=np.asarray(Image.open(c['mask']).convert('L'))>=128
        base=next(r for r in baseline if r['case_id']==c['case_id'] and r['mode']=='composited');assert sha(base['output'])==base['output_sha256']
        generated=np.asarray(Image.open(base['output']).convert('RGB'))
        arrays={'observed':observed,'all_reference':generated,'fixed_half':mix(observed,generated,mask,.5)}
        error=None
        try:
            proxy_truth=np.asarray(Image.open(c['references'][0]['path']).convert('RGB'));proxies=[];records=[]
            refs=[r['path'] for r in c['references'][1:]]
            for kind in cfg['proxy_degradation_bank']:
                degraded,_,metadata=degrade(proxy_truth,kind,'medium',29,mask=mask,region='central_face')
                input_path=dest/(kind+'_input.png');Image.fromarray(degraded).save(input_path);torch.manual_seed(17)
                noise=torch.randn([1,*shape],device='cuda')
                condition={'lq_image':read_and_normalize_image(input_path).unsqueeze(0).cuda(),
                    'ref_image':torch.cat([read_and_normalize_image(p) for p in refs],dim=-1).unsqueeze(0).cuda()}
                condition=model.get_learned_conditioning(condition);null={k:v.detach().clone() for k,v in condition.items()};null['ref_image']*=0
                start=time.monotonic()
                with model.ema_scope():
                    latent,_=sampler.sample(S=50,unconditional_guidance_scale=1.5,conditioning=condition,unconditional_conditioning=null,
                        shape=shape,x_T=noise,batch_size=1,verbose=False)
                decoded=((model.decode_first_stage(latent)+1)/2).clamp(0,1)
                if not torch.isfinite(decoded).all():raise ValueError('Nonfinite proxy output')
                output_path=dest/(kind+'_output.png');to_pil_image(decoded.squeeze(0).cpu()).save(output_path)
                restored=np.asarray(Image.open(output_path).convert('RGB'));proxies.append((degraded,restored,proxy_truth,mask))
                records.append({'kind':kind,'input_sha256':sha(input_path),'output_sha256':sha(output_path),'seconds':time.monotonic()-start})
                write_new(dest/(kind+'_receipt.json'),records[-1]);print(c['case_id'],kind,'proxy complete',flush=True)
            fitted=fit(proxies);write_new(dest/'calibration.json',fitted)
            arrays['proxy_global']=mix(observed,generated,mask,fitted['global_weight'])
            arrays['proxy_spatial'],weight=predict(observed,generated,mask,fitted);np.save(dest/'spatial_weights.npy',weight)
            write_new(dest/'proxy_provenance.json',{'calibration_reference':c['references'][0],
                'conditioning_references':c['references'][1:],'self_reference_excluded':True,'records':records,'target_truth_used':False})
        except Exception as e:error=f'{type(e).__name__}: {e}'
        for name in cfg['arms']:
            if name in arrays:
                path=dest/(name+'.png');Image.fromarray(arrays[name]).save(path)
                rows.append(dict(common,key=c['case_id']+'_'+name,mode=name,status='complete',output=str(path),output_sha256=sha(path)))
            else:rows.append(dict(common,key=c['case_id']+'_'+name,mode=name,status='failed',error=error))
        write_new(dest/'rows.json',rows[-5:])
    write_new(out/'comparison.json',{'rows':rows,'final_test_used':False})
if __name__=='__main__':main()
