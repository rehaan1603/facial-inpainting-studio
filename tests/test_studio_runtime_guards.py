import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from webapp.geometry import require_reference_mask_support
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import studio_inference_worker as worker


class StudioRuntimeGuardTests(unittest.TestCase):
    def test_thin_disconnected_mark_cannot_be_silently_lost(self):
        for size, x in [(256,200),(512,201),(1024,201)]:
            pixels=np.zeros((512,512),np.uint8);pixels[100:400,x]=255
            pixels[400:450,400:450]=255
            with self.subTest(size=size), self.assertRaisesRegex(ValueError,'too thin'):
                require_reference_mask_support(Image.fromarray(pixels),size)
            pixels[92:408,x-8:x+9]=255
            require_reference_mask_support(Image.fromarray(pixels),size)

    def test_nested_reference_analyzer_cannot_reset_thread_cap(self):
        state={'threads':24}
        cv=SimpleNamespace(__version__='test',setNumThreads=lambda value:state.update(threads=value),getNumThreads=lambda:state['threads'])
        def entry(*args,**kwargs):
            cv.setNumThreads(2)
            self.assertEqual(cv.getNumThreads(),1)
        target=Path(worker.__file__).with_name('mask_aware_inpaint.py')
        with patch.dict(sys.modules,{'cv2':cv}), patch.object(sys,'argv',['worker',str(target)]), patch.object(worker.runpy,'run_path',side_effect=entry):
            worker.main()

    def test_refldm_worker_does_not_require_opencv(self):
        target=Path(worker.__file__).with_name('refldm_restore.py')
        with patch.dict(sys.modules,{'cv2':None}), patch.object(sys,'argv',['worker',str(target)]), patch.object(worker.runpy,'run_path') as run:
            worker.main()
            run.assert_called_once_with(str(target),run_name='__main__')


if __name__=='__main__':unittest.main()
