import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from apply_saved_preservation import apply
from src.research_integrity import sha

class SavedPreservationTests(unittest.TestCase):
    def test_verified_saved_output_and_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);observed=np.full((512,512,3),100,np.uint8);mask=np.zeros((512,512),np.uint8);mask[100:200,100:200]=255
            generated=observed.copy();generated[mask>0]=200;confidence=np.full((512,512),255,np.uint8);confidence[mask>0]=128
            for name,array in [('image',observed),('mask',mask),('generated',generated),('confidence',confidence)]:Image.fromarray(array).save(p/f'{name}.png')
            metadata={k:sha(p/f'{n}.png') for k,n in [('input_sha256','image'),('mask_sha256','mask'),('result_sha256','generated')]}
            (p/'metadata.json').write_text(json.dumps(metadata))
            args={n:p/f'{n}.png' for n in ['image','mask','generated','confidence']};args.update(metadata=p/'metadata.json',output=p/'result.png')
            r=apply(**args);self.assertTrue(r['known_pixels_unchanged']);self.assertFalse(r['new_generation'])
            result=np.asarray(Image.open(p/'result.png'));self.assertEqual(int(result[150,150,0]),150)
            with self.assertRaisesRegex(ValueError,'new output path'):apply(**args)
            metadata['result_sha256']='wrong';(p/'metadata.json').write_text(json.dumps(metadata));args['output']=p/'another.png'
            with self.assertRaisesRegex(ValueError,'result_sha256'):apply(**args)
            self.assertFalse((p/'another.png').exists())

if __name__=='__main__':unittest.main()
