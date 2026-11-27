"""TASK-012: eval.metrics의 PSNR·SSIM·보조지표를 기준값과 독립 구현으로 검증.

SSIM 기준 구현은 Wang et al.(2004) 정의를 NumPy float64로 직접 옮긴 것으로,
eval.py·model.py의 함수를 사용하지 않는다(11×11 Gaussian, sigma 1.5, 유효 영역,
C1=(0.01)², C2=(0.03)², 모분산, 채널별 계산 후 평균).
"""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import metrics


def reference_ssim(output, target):
    """float64 Gaussian SSIM, 유효(valid) 영역 평균. 입력은 [C,H,W] 배열."""
    axis = np.arange(-5, 6, dtype=np.float64)
    weights = np.exp(-axis ** 2 / (2 * 1.5 ** 2))
    window = np.outer(weights, weights)
    window /= window.sum()

    def blur(image):
        patches = np.lib.stride_tricks.sliding_window_view(image, (11, 11))
        return np.einsum('ijkl,kl->ij', patches, window)

    values = []
    for x, y in zip(output.astype(np.float64), target.astype(np.float64)):
        mu_x, mu_y = blur(x), blur(y)
        var_x = blur(x * x) - mu_x ** 2
        var_y = blur(y * y) - mu_y ** 2
        cov = blur(x * y) - mu_x * mu_y
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        values.append(((2 * mu_x * mu_y + c1) * (2 * cov + c2)
                       / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))))
    return float(np.mean(values))


def tensor(array):
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[None]


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_identical_images(self):
        x = tensor(self.rng.random((3, 32, 48)))
        result = metrics(x, x)
        self.assertEqual(result['psnr'], 120.0)                 # mse 0 → 최대 120dB
        self.assertAlmostEqual(result['ssim'], 1.0, places=5)
        self.assertEqual(result['color_mean_error'], 0.0)
        self.assertEqual(result['flat_residual_std'] or 0.0, 0.0)

    def test_psnr_known_mse(self):
        base = self.rng.uniform(0.1, 0.8, (3, 32, 32))
        cases = [
            (base + 0.1, 20.0),                                  # mse 0.01
            (base + np.where(self.rng.random(base.shape) < 0.5, 0.05, -0.05),
             -10 * np.log10(0.0025)),                            # mse 0.0025 → 26.0206
            (base + 0.001, 60.0),                                # mse 1e-6
        ]
        for output, expected in cases:
            self.assertAlmostEqual(metrics(tensor(output), tensor(base))['psnr'], expected, places=3)
        # mse가 1e-12보다 작으면 120dB로 표기(무한대 방지)
        tiny = metrics(tensor(base + 1e-7), tensor(base))['psnr']
        self.assertGreaterEqual(tiny, 119.99)
        self.assertLessEqual(tiny, 120.0)

    def test_ssim_matches_independent_reference(self):
        cases = []
        for size in ((32, 48), (11, 11), (17, 23), (64, 64)):
            target = self.rng.random((3, *size))
            cases.append((np.clip(target + self.rng.normal(0, 0.1, target.shape), 0, 1), target))
        # LOL처럼 어두운 입력: float32 분산 계산(E[x²]−E[x]²)의 오차 확인용
        dark = self.rng.uniform(0.0, 0.08, (3, 40, 60))
        cases.append((np.clip(dark * 1.3 + self.rng.normal(0, 0.005, dark.shape), 0, 1), dark))
        # 밝고 매끄러운 영상(분산이 작아 상대 오차가 커지기 쉬운 경우)
        ramp = np.broadcast_to(np.linspace(0.6, 0.9, 50), (3, 40, 50)).copy()
        cases.append((np.clip(ramp + self.rng.normal(0, 0.01, ramp.shape), 0, 1), ramp))
        for output, target in cases:
            ours = metrics(tensor(output), tensor(target))['ssim']
            self.assertAlmostEqual(ours, reference_ssim(output, target), delta=1e-4,
                                   msg=f'shape {target.shape}')

    def test_ssim_boundary_rule(self):
        """가로·세로 11 이상이면 유효 영역만, 10 이하이면 replicate 패딩 전체 평균."""
        target = self.rng.random((3, 11, 11))
        output = np.clip(target + 0.05, 0, 1)
        self.assertAlmostEqual(metrics(tensor(output), tensor(target))['ssim'],
                               reference_ssim(output, target), delta=1e-4)
        small = self.rng.random((3, 10, 10))
        value = metrics(tensor(np.clip(small + 0.05, 0, 1)), tensor(small))['ssim']
        self.assertTrue(np.isfinite(value))   # 유효 창이 없으므로 표준 SSIM과 비교 불가

    def test_auxiliary_metrics(self):
        # 과노출 비율: 휘도 ≥ 0.98인 화소가 1/4
        output = np.full((3, 20, 20), 0.5)
        output[:, :10, :10] = 1.0
        target = np.full((3, 20, 20), 0.5)
        self.assertAlmostEqual(metrics(tensor(output), tensor(target))['saturation_ratio'], 0.25, places=6)
        # 채널 평균 오차: (|0.5−0.4| + 0 + |0.3−0.4|) / 3
        output = np.stack([np.full((20, 20), v) for v in (0.5, 0.4, 0.3)])
        target = np.full((3, 20, 20), 0.4)
        self.assertAlmostEqual(metrics(tensor(output), tensor(target))['color_mean_error'], 0.2 / 3, places=6)
        # 평탄 잔차: 정답이 상수면 전부 평탄 → 휘도 차이(무작위 ±d)의 모표준편차
        target = np.full((3, 20, 20), 0.4)
        sign = np.where(self.rng.random((20, 20)) < 0.5, 1.0, -1.0)
        output = target + 0.02 * sign
        expected = np.std(0.02 * sign)
        self.assertAlmostEqual(metrics(tensor(output), tensor(target))['flat_residual_std'], expected, places=5)
        # 평탄 영역이 없으면 None
        self.assertIsNone(metrics(tensor(output), tensor(target), flat_threshold=0)['flat_residual_std'])


if __name__ == '__main__':
    unittest.main()
