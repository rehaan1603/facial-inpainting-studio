import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.degradation import degrade
from src.degradation.distortion_pipeline import KINDS
from src.reference_selection.mask_aware_scorer import score_reference
from src.reference_selection.selection_baselines import select
from src.reference_selection.reference_analyzer import ReferenceAnalyzer
from src.degradation.face_region_masks import REGIONS
from src.research_integrity import ROOT

class UpgradeTests(unittest.TestCase):
    def test_distortions_deterministic_and_preserve_known_pixels(self):
        rgb=np.random.default_rng(3).integers(0,256,(64,64,3),dtype='uint8')
        for kind in KINDS:
            for severity in ['mild','medium','severe']:
                a,m,meta=degrade(rgb,kind,severity);b,n,_=degrade(rgb,kind,severity)
                self.assertTrue(np.array_equal(a,b));self.assertTrue(np.array_equal(m,n));self.assertTrue(np.array_equal(a[~m],rgb[~m]))
                self.assertTrue(m.any());self.assertFalse(m.all());self.assertEqual(a.dtype,np.uint8)
    def test_removal_does_not_reveal_hidden_pixels(self):
        rgb=np.zeros((64,64,3),dtype='uint8');a,m,_=degrade(rgb)
        rgb[m]=255;b,_,_=degrade(rgb);self.assertTrue(np.array_equal(a,b))
    def test_mask_changes_selection(self):
        weights={'regional_quality':1.}
        records=[]
        for index in range(2):
            regions={r:dict(quality=.5,visibility_proxy=1.,exposure=1.) for r in REGIONS}
            regions['left_eye']['quality']=1-index;regions['mouth']['quality']=index
            records.append(dict(path=str(index),valid=True,regions=regions,global_quality=.5,pose_similarity=.5,identity_compatibility=.5))
        for region,expected in [('left_eye','0'),('mouth','1')]:
            demand={r:float(r==region) for r in REGIONS}
            for r in records:r['mask_aware_score']=score_reference(r,demand,weights)[0]
            self.assertEqual(select({'references':records}),[expected])
    def test_invalid_references_and_needs_mask(self):
        class NoFaces:
            def get(self,rgb):return []
        analyzer=ReferenceAnalyzer(NoFaces())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);Image.new('RGB',(64,64)).save(p/'input.png');Image.new('L',(64,64),255).save(p/'mask.png')
            result=analyzer.analyze(p/'input.png',p/'mask.png',[p/'missing.png',p/'input.png'])
            self.assertEqual(select(result),[]);self.assertEqual(len(result['references']),2)
            Image.new('L',(64,64)).save(p/'mask.png')
            with self.assertRaisesRegex(ValueError,'needs_mask'):analyzer.analyze(p/'input.png',p/'mask.png',[p/'input.png'])
    def test_frozen_protocol_disjoint(self):
        p=ROOT/'research/protocols/unseen_identity_protocol_v1.json'
        if not p.exists():self.skipTest('Protocol not yet prepared')
        data=json.loads(p.read_text());excluded=set().union(*(set(v['identity_labels']) for v in data['exclusion_ledger'].values()))
        ids=[c['identity'] for c in data['cases']];self.assertEqual(len(ids),len(set(ids)));self.assertFalse(set(ids)&excluded)
        hashes=[r['decoded_rgb_sha256'] for c in data['cases'] for r in c['images']];self.assertEqual(len(hashes),len(set(hashes)))
        self.assertEqual(sum(c['split']=='reserved_final' for c in data['cases']),8)
        for case in data['cases']:
            self.assertEqual({r['role'] for r in case['images']},{'target',*[f'reference_{i}' for i in range(1,5)],*[f'gallery_{i}' for i in range(1,4)]})
        fingerprints=[int(r['phash'],16) for c in data['cases'] for r in c['images']]
        self.assertTrue(all((a^b).bit_count()>6 for i,a in enumerate(fingerprints) for b in fingerprints[i+1:]))
    def test_distortion_severity_and_region_masks(self):
        rgb=np.random.default_rng(4).integers(0,256,(128,128,3),dtype='uint8');original=rgb.copy()
        for region in ['rectangle','brush','eyes','nose','mouth','half_face','central_face']:
            _,mask,_=degrade(rgb,region=region)
            self.assertTrue(mask.any());self.assertFalse(mask.all())
        mild,_,_=degrade(rgb,'noise','mild');severe,_,_=degrade(rgb,'noise','severe')
        self.assertFalse(np.array_equal(mild,severe));self.assertTrue(np.array_equal(rgb,original))
    def test_duplicate_reference_and_disagreement_diagnostics(self):
        from types import SimpleNamespace
        class Faces:
            def get(self,rgb):
                # Distinct sign-coded descriptors exercise warnings, not identity accuracy.
                sign=1 if rgb.mean()<100 else -1
                return [SimpleNamespace(bbox=np.array([8,8,56,60]),kps=np.array([[22,24],[42,24],[32,35],[25,46],[39,46]]),normed_embedding=np.array([float(sign),0.]),det_score=.99)]
        analyzer=ReferenceAnalyzer(Faces())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for name,value in [('input',90),('a',50),('b',200)]:Image.new('RGB',(64,64),(value,)*3).save(p/(name+'.png'))
            mask=np.zeros((64,64),dtype='uint8');mask[20:30,20:44]=255;Image.fromarray(mask).save(p/'mask.png')
            r=analyzer.analyze(p/'input.png',p/'mask.png',[p/'a.png',p/'a.png',p/'b.png'])
            self.assertEqual(sum(x['valid'] for x in r['references']),2)
            self.assertIn('duplicate',r['references'][1]['rejection_reason'])
            self.assertTrue(any('disagree' in w for w in r['warnings']))
            self.assertEqual(select(r,'random',17),select(r,'random',17))

if __name__=='__main__':unittest.main()
