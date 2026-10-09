"""One fixed lower reference-strength probe on three erased-eye cases."""
import json,time
from pathlib import Path
from reference_inpaint import reconstruct,sha,ROOT


def main():
    out=ROOT/'outputs/focused_reference_strength_v1';out.mkdir(exist_ok=False)
    source=ROOT/'outputs/masked_sdxl_resolution_v1';cfg=json.loads((source/'config.json').read_text())
    cfg.update(expected=3,scale=.4,selected_identities=['263','6681','10149'],locations=['eyes'],blend='poisson',
               source_config_sha256=sha(source/'config.json'),
               hypothesis='Reducing reference attention strength may reduce over-dominant reference appearance and improve agreement with visible target context',
               single_change='FaceID scale0.8 to0.4 versus1024 blended control; all other settings fixed')
    (out/'config.json').write_text(json.dumps(cfg,indent=2));runtime={};rows=[]
    cases={c['identity']:c for c in json.loads((ROOT/'outputs/masked_task_v1/cases.json').read_text()) if c['location']=='eyes'}
    for ident in cfg['selected_identities']:
        c=cases[ident];refs=cfg['references'][ident+'/clean']
        assert sha(c['input'])==c['input_sha256'] and sha(c['mask'])==c['mask_sha256']
        for r in refs:assert sha(r['path'])==r['sha256']
        p=out/f'{ident}_eyes_clean_17.png';start=time.monotonic()
        meta=reconstruct(Path(c['input']),Path(c['mask']),[Path(r['path']) for r in refs],p,
                         seed=17,steps=30,scale=.4,strength=1.,memory='model',blend='poisson',runtime=runtime,model_resolution=1024)
        rows.append(dict(identity=ident,location='eyes',condition='clean',seed=17,file=p.name,status='complete',
                         sha256=sha(p),raw_sha256=meta['raw_output_sha256'],seconds=time.monotonic()-start,
                         peak_allocated_bytes=meta['peak_allocated_bytes']))
        (out/'rows.json').write_text(json.dumps(rows,indent=2));print('OUTPUT',ident,flush=True)


if __name__=='__main__':main()
