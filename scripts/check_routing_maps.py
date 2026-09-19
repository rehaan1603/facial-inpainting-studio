"""Verify real-case routing maps at the resolutions used by stock attention."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from diffusers.image_processor import IPAdapterMaskProcessor
from src.reference_fusion.feature_bank import analyze_runtime
from src.reference_fusion.regional_fusion import build_masks
from src.research_integrity import ROOT,write_new

def main():
    cases=json.loads((ROOT/'outputs/generalization_v1/cases/manifest.json').read_text())['cases'];analyzer=None;records=[]
    for c in cases:
        analysis,geometry,analyzer=analyze_runtime(c['observed'],c['mask'],[r['path'] for r in c['references']],analyzer)
        mask=np.asarray(Image.open(c['mask']).convert('L'))>=128
        maps,record=build_masks(analysis,mask,geometry)
        levels=[]
        for queries in [64,256,1024,4096]:
            down=torch.stack([IPAdapterMaskProcessor.downsample(torch.from_numpy(m)[None],2,queries,1) for m in maps])
            minimum=float(down.min());maximum=float(down.max());error=float((down.sum(0)-1).abs().max())
            if minimum<0 or maximum>1 or error>1e-5:raise ValueError(f'Interpolation violates routing bounds for {c["case_id"]}: {minimum}, {maximum}, {error}')
            levels.append({'queries':queries,'min':minimum,'max':maximum,'partition_max_error':error})
        records.append({'case_id':c['case_id'],'levels':levels,'details':record})
    write_new(ROOT/'outputs/regional_routing_v1/map_checks.json',{'status':'passed','cases':records})
    print('All 12 cases preserve bounded, normalized routing at four attention resolutions')

if __name__=='__main__':main()
