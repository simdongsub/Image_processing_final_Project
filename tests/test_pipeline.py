"""합성 영상으로 실행 경로를 검증합니다. 성능 검증용 데이터가 아닙니다."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataset import create_lol_dataloaders
from model import ZeroDCE, MODES
from loss import ZeroDCELoss
from eval import metrics

ROOT = Path(__file__).resolve().parents[1]

class PipelineTests(unittest.TestCase):
    def test_modes_and_gradients(self):
        torch.set_num_threads(1)
        x=torch.rand(1,3,24,24)*0.4
        count=[]
        for mode in MODES:
            model=ZeroDCE(mode)
            output,curves,_=model(x)
            self.assertEqual(output.shape,x.shape)
            self.assertTrue(bool(((output>=0)&(output<=1)).all()))
            loss,_=ZeroDCELoss()(x,output,curves)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            count.append(sum(p.numel() for p in model.parameters()))
        self.assertEqual(len(set(count)),1)
        self.assertAlmostEqual(metrics(x,x)['ssim'],1,places=5)

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for split,n in [('train',5),('test',2)]:
                for name in ('low','high'): (root/split/name).mkdir(parents=True)
                for i in range(n):
                    rng=np.random.default_rng(i)
                    image=rng.integers(5,65,(24,24,3),dtype=np.uint8)
                    Image.fromarray(image).save(root/split/'low'/f'{i}.png')
                    Image.fromarray(image*3).save(root/split/'high'/f'{i}.png')
            kwargs=dict(low_dir=root/'train/low',high_dir=root/'train/high',test_low_dir=root/'test/low',
                        test_high_dir=root/'test/high',crop_size=16,batch_size=2,num_workers=0)
            a=create_lol_dataloaders(**kwargs,seed=0,split_seed=42)
            b=create_lol_dataloaders(**kwargs,seed=1,split_seed=42)
            self.assertEqual(a['train'].dataset.pairs,b['train'].dataset.pairs)
            self.assertEqual(a['test'].batch_size,1)
            args=['--low-dir',str(root/'train/low'),'--high-dir',str(root/'train/high'),
                  '--test-low-dir',str(root/'test/low'),'--test-high-dir',str(root/'test/high'),
                  '--epochs','1','--crop-size','16','--batch-size','2','--output',str(root/'runs')]
            def command(script,*params,success=True):
                import os
                env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
                result=subprocess.run([sys.executable,str(ROOT/script),*params],capture_output=True,text=True,env=env)
                if success: self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                else: self.assertNotEqual(result.returncode,0)
                return result
            for mode in MODES: command('train.py',*args,'--mode',mode)
            command('train.py',*args,'--mode','none','--resume')
            command('eval.py','--runs',str(root/'runs'),'--output',str(root/'eval'),'--warmup','0','--repeats','1')
            summary=json.loads((root/'eval/test/summary.json').read_text())
            self.assertEqual(len(summary['runs']),4)
            self.assertEqual(len(list((root/'eval/test/seed0').glob('*_comparison.png'))),2)
            checkpoint=root/'runs/both_seed0/last.pt'
            saved=torch.load(checkpoint,weights_only=False); saved['complete']=False; torch.save(saved,checkpoint)
            command('eval.py','--runs',str(root/'runs'),'--output',str(root/'rejected'),success=False)

if __name__=='__main__': unittest.main()
