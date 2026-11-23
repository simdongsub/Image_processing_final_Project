"""합성 영상으로 실행 경로를 검증합니다. 성능 검증용 데이터가 아닙니다."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch.utils.data import RandomSampler, SequentialSampler

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataset import (ImagePair, LowLightDataset, audit_lol_splits,
                     create_lol_dataloaders, create_lol_datasets)
from model import ZeroDCE, MODES
from loss import ZeroDCELoss
from eval import metrics

ROOT = Path(__file__).resolve().parents[1]

class PipelineTests(unittest.TestCase):
    def test_dataset_integrity_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ('train', 'test'):
                for role in ('low', 'high'):
                    (root/split/role).mkdir(parents=True)
            train_low = root/'train/low/a.png'
            train_high = root/'train/high/a.png'
            test_low = root/'test/low/b.png'
            test_high = root/'test/high/b.png'
            for path, value in ((train_low, 10), (train_high, 100),
                                (root/'train/low/c.png', 11),
                                (root/'train/high/c.png', 101),
                                (test_low, 20), (test_high, 120)):
                Image.new('RGB', (24, 24), (value,)*3).save(path)

            def audit():
                return audit_lol_splits(root/'train/low', root/'train/high',
                                        root/'test/low', root/'test/high', crop_size=16)

            self.assertEqual(audit()['issues'], [])
            self.assertEqual(audit()['splits']['train']['valid_pairs'], 2)
            self.assertEqual(audit()['partition']['counts'], {'train': 1, 'val': 1, 'test': 1})

            train_high.unlink()
            self.assertEqual(audit()['splits']['train']['missing_high'], 1)
            Image.new('RGB', (24, 24), (100,)*3).save(train_high)

            train_low.unlink()
            self.assertEqual(audit()['splits']['train']['missing_low'], 1)
            Image.new('RGB', (24, 24), (10,)*3).save(train_low)

            train_high.write_bytes(b'not an image')
            self.assertEqual(audit()['splits']['train']['corrupt_files'], 1)
            Image.new('RGB', (25, 24), (100,)*3).save(train_high)
            self.assertEqual(audit()['splits']['train']['size_mismatches'], 1)
            Image.new('RGB', (24, 24), (100,)*3).save(train_high)

            test_low.write_bytes(train_low.read_bytes())
            self.assertTrue(any('train/test 파일 중복' in issue for issue in audit()['issues']))
            Image.new('RGB', (24, 24), (20,)*3).save(test_low)

            Image.new('RGB', (24, 24)).save(root/'train/low/a.jpg')
            self.assertEqual(audit()['splits']['train']['duplicate_names'], 1)
            (root/'train/low/a.jpg').unlink()
            (root/'train/low/unknown.txt').write_text('unsupported')
            self.assertEqual(audit()['splits']['train']['unsupported_files'], 1)

            absent = audit_lol_splits(root/'absent', root/'absent',
                                      root/'absent', root/'absent')
            self.assertIsNone(absent['splits']['train']['matched'])
            self.assertTrue(absent['issues'])

    def test_split_contract_and_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, count, start in (('train', 20, 10), ('test', 3, 130)):
                for role in ('low', 'high'):
                    (root/split/role).mkdir(parents=True)
                for index in range(count):
                    value = start + index
                    Image.new('RGB', (24, 24), (value,)*3).save(root/split/'low'/f'{index}.png')
                    Image.new('RGB', (24, 24), (value+40,)*3).save(root/split/'high'/f'{index}.png')

            args = dict(low_dir=root/'train/low', high_dir=root/'train/high',
                        test_low_dir=root/'test/low', test_high_dir=root/'test/high',
                        val_ratio=0.1, crop_size=16, batch_size=4, num_workers=0)
            first = create_lol_dataloaders(**args, split_seed=7, seed=0)
            second = create_lol_dataloaders(**args, split_seed=7, seed=1)
            self.assertEqual([len(first[name].dataset) for name in ('train','val','test')],
                             [18, 2, 3])
            self.assertEqual(first['train'].dataset.pairs, second['train'].dataset.pairs)
            self.assertEqual(first['val'].dataset.pairs, second['val'].dataset.pairs)
            self.assertEqual(first['test'].dataset.pairs, second['test'].dataset.pairs)
            different_split = create_lol_dataloaders(**args, split_seed=8, seed=0)
            self.assertNotEqual(first['val'].dataset.pairs, different_split['val'].dataset.pairs)

            pair_paths = [{pair.low.resolve() for pair in first[name].dataset.pairs}
                          for name in ('train', 'val', 'test')]
            self.assertTrue(all(not pair_paths[i] & pair_paths[j]
                                for i, j in ((0, 1), (0, 2), (1, 2))))
            self.assertEqual(pair_paths[2], {path.resolve() for path in (root/'test/low').glob('*.png')})
            self.assertIsInstance(first['train'].sampler, RandomSampler)
            self.assertIsInstance(first['val'].sampler, SequentialSampler)
            self.assertIsInstance(first['test'].sampler, SequentialSampler)
            self.assertTrue(first['train'].dataset.augment)
            self.assertFalse(first['train'].dataset.return_reference)
            self.assertFalse(first['val'].dataset.augment)
            self.assertFalse(first['test'].dataset.augment)
            self.assertTrue(first['val'].dataset.return_reference)
            self.assertTrue(first['test'].dataset.return_reference)
            self.assertIsNone(first['val'].dataset.crop_size)
            self.assertIsNone(first['test'].dataset.crop_size)
            self.assertFalse(first['train'].drop_last)

            batches = list(first['train'])
            self.assertEqual([batch.shape[0] for batch in batches], [4, 4, 4, 4, 2])
            self.assertEqual(tuple(batches[0].shape[1:]), (3, 16, 16))
            self.assertEqual(batches[0].dtype, torch.float32)
            self.assertTrue(bool(torch.isfinite(batches[0]).all()))
            self.assertGreaterEqual(float(batches[0].min()), 0)
            self.assertLessEqual(float(batches[0].max()), 1)
            for name in ('val', 'test'):
                low, high = next(iter(first[name]))
                self.assertEqual(tuple(low.shape), (1, 3, 24, 24))
                self.assertEqual(tuple(high.shape), (1, 3, 24, 24))
                self.assertEqual(first[name].batch_size, 1)
            large_batch = create_lol_dataloaders(**dict(args, batch_size=100), split_seed=7)
            self.assertEqual(next(iter(large_batch['train'])).shape[0], 18)
            with self.assertRaises(ValueError):
                create_lol_datasets(args['low_dir'], args['high_dir'],
                                    args['low_dir'], args['high_dir'])

            def audit():
                return audit_lol_splits(args['low_dir'], args['high_dir'],
                                        args['test_low_dir'], args['test_high_dir'],
                                        crop_size=16, val_ratio=0.1, split_seed=7)
            result = audit()
            self.assertEqual(result['issues'], [])
            self.assertEqual(result['partition']['counts'], {'train': 18, 'val': 2, 'test': 3})
            self.assertTrue(all(not any(overlap.values())
                                for overlap in result['partition']['overlap'].values()))
            train_pair = first['train'].dataset.pairs[0]
            val_pair = first['val'].dataset.pairs[0]
            val_pair.low.write_bytes(train_pair.low.read_bytes())
            val_pair.high.write_bytes(train_pair.high.read_bytes())
            self.assertTrue(any('train/val 파일 중복' in issue for issue in audit()['issues']))

    def test_paired_spatial_augmentation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = np.arange(24*24, dtype=np.uint8).reshape(24, 24) % 200
            low_path, high_path = root/'low.png', root/'high.png'
            Image.fromarray(base).save(low_path)
            Image.fromarray(base + 20).save(high_path)
            pair = ImagePair(low_path, high_path)
            dataset = LowLightDataset([pair], crop_size=16, augment=True,
                                      return_reference=True)
            with patch('dataset.random.randint', side_effect=[3, 5]), \
                 patch('dataset.random.random', return_value=0.0):
                low, high = dataset[0]
            expected = base[5:21, 3:19][:, ::-1].copy().astype(np.float32) / 255
            np.testing.assert_allclose(low[0].numpy(), expected, atol=1e-6)
            np.testing.assert_allclose((high-low).numpy(), 20/255, atol=1e-6)
            self.assertEqual(tuple(low.shape), (3, 16, 16))
            self.assertEqual(low.dtype, torch.float32)
            self.assertEqual(high.dtype, torch.float32)
            self.assertTrue(bool(torch.isfinite(low).all() and torch.isfinite(high).all()))
            self.assertGreaterEqual(float(high.min()), 0)
            self.assertLessEqual(float(high.max()), 1)
            val_low, val_high = LowLightDataset([pair], return_reference=True)[0]
            self.assertEqual(tuple(val_low.shape), (3, 24, 24))
            self.assertEqual(tuple(val_high.shape), (3, 24, 24))
            with self.assertRaises(ValueError):
                LowLightDataset([pair], crop_size=25)[0]
            with self.assertRaises(ValueError):
                LowLightDataset([])

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
            run_dir=root/'runs/none_seed0'
            config=json.loads((run_dir/'config.json').read_text())
            manifest=json.loads((run_dir/'manifest.json').read_text())
            self.assertEqual(config['settings']['split_seed'],0)
            self.assertEqual([len(manifest[name]) for name in ('train','val','test')],[4,1,2])
            self.assertFalse({item['low'] for item in manifest['train']}
                             & {item['low'] for item in manifest['val']})
            command('train.py',*args,'--mode','none','--seed','1')
            second_manifest=json.loads((root/'runs/none_seed1/manifest.json').read_text())
            self.assertEqual(manifest,second_manifest)
            command('train.py',*args,'--mode','none','--resume')
            command('eval.py','--runs',str(root/'runs'),'--output',str(root/'eval'),'--warmup','0','--repeats','1')
            summary=json.loads((root/'eval/test/summary.json').read_text())
            self.assertEqual(len(summary['runs']),4)
            self.assertEqual(len(list((root/'eval/test/seed0').glob('*_comparison.png'))),2)
            checkpoint=root/'runs/both_seed0/last.pt'
            saved=torch.load(checkpoint,weights_only=False); saved['complete']=False; torch.save(saved,checkpoint)
            command('eval.py','--runs',str(root/'runs'),'--output',str(root/'rejected'),success=False)

if __name__=='__main__': unittest.main()
