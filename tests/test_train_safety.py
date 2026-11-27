"""TASK-010(중단·재개)과 TASK-011(시간 한도·미완료 거부) 검증.

train.py·eval.py는 수정하지 않는다. 합성 영상을 쓰므로 성능 수치가 아니다.
- 중단: 3번째 에폭 학습 도중 예외를 일으켜 세션 종료를 흉내 낸다.
- 시간: train.time을 가짜 시계로 바꿔 정확한 시점에 예산을 넘기게 한다.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train
from loss import ZeroDCELoss

TRAIN_BATCHES = 5          # train 9장(풀 10장 중 val 1장), batch 2 → 2,2,2,2,1
CALLS_PER_EPOCH = TRAIN_BATCHES + 1   # + val 1장


def make_data(root):
    for split, count in (('train', 10), ('test', 2)):
        for role in ('low', 'high'):
            (root / split / role).mkdir(parents=True)
        for index in range(count):
            image = np.random.default_rng(index + 100 * (split == 'test')).integers(5, 70, (24, 24, 3), dtype=np.uint8)
            Image.fromarray(image).save(root / split / 'low' / f'{index}.png')
            Image.fromarray(np.clip(image.astype(int) * 3, 0, 255).astype(np.uint8)).save(root / split / 'high' / f'{index}.png')


def cli_args(root, output, epochs=3, max_hours=3.0, **extra):
    args = ['--low-dir', str(root / 'train/low'), '--high-dir', str(root / 'train/high'),
            '--test-low-dir', str(root / 'test/low'), '--test-high-dir', str(root / 'test/high'),
            '--epochs', str(epochs), '--crop-size', '16', '--batch-size', '2',
            '--output', str(output), '--device', 'cpu', '--max_hours', repr(max_hours)]
    for key, value in extra.items():
        args += [f'--{key.replace("_", "-")}', str(value)]
    return args


def run_cli(script, *args, success=True):
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUTF8='1')
    result = subprocess.run([sys.executable, str(ROOT / script), *args],
                            capture_output=True, text=True, encoding='utf-8', env=env)
    if success:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0, result.stdout
    return result


def checkpoint(output, mode='none', seed=0):
    return torch.load(Path(output) / f'{mode}_seed{seed}' / 'last.pt', map_location='cpu', weights_only=False)


def assert_same_state(test, a, b):
    test.assertEqual(a.keys(), b.keys())
    for key in a:
        test.assertTrue(torch.equal(a[key], b[key]), key)


def assert_same_optimizer(test, a, b):
    test.assertEqual(a['param_groups'], b['param_groups'])
    for index, state in a['state'].items():
        for key, value in state.items():
            other = b['state'][index][key]
            test.assertTrue(torch.equal(value, other) if torch.is_tensor(value) else value == other, (index, key))


class FakeClock:
    """손실이 한 번 계산될 때마다 1초가 흐르는 시계."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now


def ticking_loss(clock):
    class TickingLoss(ZeroDCELoss):
        def forward(self, *args):
            clock.now += 1.0
            return super().forward(*args)
    return TickingLoss


class ResumeTests(unittest.TestCase):
    """TASK-010."""

    def test_interrupted_resume_matches_continuous_run(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_data(root)
            run_cli('train.py', *cli_args(root, root / 'continuous'))

            # 3번째 에폭의 3번째 학습 배치에서 세션이 죽은 상황
            calls = {'n': 0}
            crash_at = 2 * CALLS_PER_EPOCH + 3

            class CrashingLoss(ZeroDCELoss):
                def forward(self, *args):
                    calls['n'] += 1
                    if calls['n'] == crash_at:
                        raise KeyboardInterrupt('session lost')
                    return super().forward(*args)

            interrupted = root / 'interrupted'
            with patch('train.ZeroDCELoss', CrashingLoss), self.assertRaises(KeyboardInterrupt):
                train.run(train.parser().parse_args(cli_args(root, interrupted)))
            saved = checkpoint(interrupted)
            self.assertEqual(saved['epoch'], 2)          # 부분 에폭은 버리고 마지막 완료 에폭만
            self.assertFalse(saved['complete'])
            self.assertEqual(len(saved['history']), 2)
            self.assertEqual(json.loads((interrupted / 'none_seed0/history.json').read_text()), saved['history'])

            run_cli('eval.py', '--runs', str(interrupted), '--compare', 'none', '--device', 'cpu',
                    '--output', str(root / 'eval_rejected'), success=False)

            # 새 프로세스에서 같은 명령 + --resume
            run_cli('train.py', *cli_args(root, interrupted), '--resume')
            resumed, continuous = checkpoint(interrupted), checkpoint(root / 'continuous')
            self.assertTrue(resumed['complete'])
            self.assertEqual(resumed['epoch'], 3)
            assert_same_state(self, resumed['model'], continuous['model'])
            assert_same_optimizer(self, resumed['optimizer'], continuous['optimizer'])
            self.assertEqual(resumed['manifest'], continuous['manifest'])
            for a, b in zip(resumed['history'], continuous['history']):
                self.assertEqual((a['epoch'], a['train_loss'], a['val_loss']),
                                 (b['epoch'], b['train_loss'], b['val_loss']))

    def test_resume_rejects_changed_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_data(root)
            output = root / 'runs'
            run_cli('train.py', *cli_args(root, output, epochs=1))
            for changed in (dict(lr=0.001), dict(split_seed=1), dict(max_hours=2.0), dict(epochs=2)):
                args = cli_args(root, output, epochs=changed.pop('epochs', 1),
                                max_hours=changed.pop('max_hours', 3.0), **changed)
                result = run_cli('train.py', *args, '--resume', success=False)
                self.assertIn('재개 설정', result.stderr)
            # --resume 없이 같은 경로로 다시 시작하면 덮어쓰지 않고 거부
            result = run_cli('train.py', *cli_args(root, output, epochs=1), success=False)
            self.assertIn('FileExistsError', result.stderr)


class TimeBudgetTests(unittest.TestCase):
    """TASK-011. 가짜 시계: 손실 계산 1회 = 1초, 에폭당 6초(학습 5 + 검증 1)."""

    def run_with_clock(self, root, output, budget_seconds, epochs=2, resume=False):
        clock = FakeClock()
        args = cli_args(root, output, epochs=epochs, max_hours=budget_seconds / 3600)
        if resume:
            args.append('--resume')
        with patch('train.time', SimpleNamespace(monotonic=clock.monotonic)), \
             patch('train.ZeroDCELoss', ticking_loss(clock)):
            train.run(train.parser().parse_args(args))
        return checkpoint(output)

    def test_stop_mid_epoch_keeps_last_completed_epoch(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_data(root)
            # 2번째 에폭 학습 중 9초에 예산(8.5초) 초과 → 부분 에폭 폐기
            stopped = self.run_with_clock(root, root / 'stopped', 8.5)
            self.assertEqual(stopped['epoch'], 1)
            self.assertFalse(stopped['complete'])
            self.assertEqual(len(stopped['history']), 1)
            run_cli('train.py', *cli_args(root, root / 'one_epoch', epochs=1))
            assert_same_state(self, stopped['model'], checkpoint(root / 'one_epoch')['model'])
            run_cli('eval.py', '--runs', str(root / 'stopped'), '--compare', 'none', '--device', 'cpu',
                    '--output', str(root / 'eval_rejected'), success=False)

            # 같은 설정으로 재개해도 누적 시간이 남아 있어 다시 멈추고 완료되지 않음
            again = self.run_with_clock(root, root / 'stopped', 8.5, resume=True)
            self.assertFalse(again['complete'])
            self.assertEqual(again['epoch'], 1)

    def test_budget_exceeded_during_final_validation(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_data(root)
            # 학습 배치는 모두 예산 안(마지막 검사 10초 < 11.5초), 마지막 검증 후 12초 → 미완료
            late = self.run_with_clock(root, root / 'late', 11.5)
            self.assertEqual(late['epoch'], 2)
            self.assertFalse(late['complete'])
            self.assertGreater(late['elapsed_seconds'], 11.5)
            run_cli('eval.py', '--runs', str(root / 'late'), '--compare', 'none', '--device', 'cpu',
                    '--output', str(root / 'eval_rejected'), success=False)
            # 예산이 충분하면 같은 시계에서 complete=true
            enough = self.run_with_clock(root, root / 'enough', 12.5)
            self.assertTrue(enough['complete'])
            self.assertEqual(enough['elapsed_seconds'], 12.0)

    def test_lost_partial_epoch_time_is_not_counted(self):
        """세션이 에폭 중간에 죽으면 그 부분 에폭 시간은 elapsed에 남지 않는다(기록용 확인)."""
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_data(root)
            clock = FakeClock()
            base = ticking_loss(clock)

            class Crash(base):
                def forward(self, *args):
                    if clock.now >= 9:              # 2번째 에폭 4번째 배치 전에 종료
                        raise KeyboardInterrupt
                    return super().forward(*args)

            output = root / 'runs'
            with patch('train.time', SimpleNamespace(monotonic=clock.monotonic)), \
                 patch('train.ZeroDCELoss', Crash), self.assertRaises(KeyboardInterrupt):
                train.run(train.parser().parse_args(cli_args(root, output, epochs=2)))
            lost = clock.now - checkpoint(output)['elapsed_seconds']
            self.assertEqual(checkpoint(output)['elapsed_seconds'], 6.0)
            self.assertEqual(lost, 3.0)              # 실제로 쓴 3초가 기록되지 않음
            resumed = self.run_with_clock(root, output, 3 * 3600, resume=True)
            self.assertTrue(resumed['complete'])
            self.assertEqual(resumed['elapsed_seconds'], 12.0)   # 실제 소비 15초 중 12초만 기록


if __name__ == '__main__':
    unittest.main()
