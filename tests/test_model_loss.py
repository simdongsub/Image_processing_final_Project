"""TASK-007(DCE 구조·초기화·곡선)과 TASK-008(손실 구성·가중치) 검증.

합성 텐서만 사용합니다. 성능 수치가 아니라 코드가 Zero-DCE 공식 구현의
구조·수식과 일치하는지 확인하는 테스트입니다.
"""
import inspect
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loss import ZeroDCELoss
from model import MODES, ZeroDCE

OFFICIAL_CHANNELS = [(3, 32), (32, 32), (32, 32), (32, 32), (64, 32), (64, 32), (64, 24)]
OFFICIAL_PARAMETERS = 79416  # Zero-DCE 논문의 약 79K 파라미터


def official_losses(original, enhanced, curves):
    """저자 저장소 Myloss.py·lowlight_train.py를 그대로 옮긴 독립 참조 구현."""
    # L_color: k = sqrt(Drg² + Drb² + Dgb²), Dxy = (mean_x - mean_y)²
    mean_rgb = enhanced.mean((2, 3), keepdim=True)
    mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
    drg, drb, dgb = (mr - mg) ** 2, (mr - mb) ** 2, (mg - mb) ** 2
    color = torch.sqrt(drg ** 2 + drb ** 2 + dgb ** 2).mean()
    # L_exp(16, 0.6)
    exposure = ((F.avg_pool2d(enhanced.mean(1, keepdim=True), 16) - 0.6) ** 2).mean()
    # L_spa: 4×4 평균 풀링 후 좌·우·상·하 이웃 차이, padding=1
    org_pool = F.avg_pool2d(original.mean(1, keepdim=True), 4)
    enh_pool = F.avg_pool2d(enhanced.mean(1, keepdim=True), 4)
    kernels = {
        'left': [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
        'right': [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
        'up': [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
        'down': [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
    }
    spatial_map = 0
    for kernel in kernels.values():
        weight = original.new_tensor(kernel)[None, None]
        spatial_map = spatial_map + (F.conv2d(org_pool, weight, padding=1)
                                     - F.conv2d(enh_pool, weight, padding=1)) ** 2
    spatial = spatial_map.mean()
    # L_TV: 채널 합 / (배치 × 공간 위치 수), TVLoss_weight=1
    batch, _, h, w = curves.shape
    h_tv = ((curves[:, :, 1:, :] - curves[:, :, :h - 1, :]) ** 2).sum()
    w_tv = ((curves[:, :, :, 1:] - curves[:, :, :, :w - 1]) ** 2).sum()
    tv = 2 * (h_tv / ((h - 1) * w) + w_tv / (h * (w - 1))) / batch
    total = 200 * tv + spatial + 5 * color + 10 * exposure
    return total, {'spatial': spatial, 'exposure': exposure, 'color': color, 'tv': tv}


def saturated_model(mode, value):
    """마지막 층이 모든 곡선 계수를 tanh(value)로 내도록 고정한 모델."""
    model = ZeroDCE(mode)
    with torch.no_grad():
        model.layers[6].weight.zero_()
        model.layers[6].bias.fill_(value)
    return model


class DCEStructureTests(unittest.TestCase):
    """TASK-007: DCE 계층·초기화·곡선."""

    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)

    def test_layers_and_parameter_count(self):
        model = ZeroDCE()
        self.assertEqual([(l.in_channels, l.out_channels) for l in model.layers], OFFICIAL_CHANNELS)
        for layer in model.layers:
            self.assertEqual(layer.kernel_size, (3, 3))
            self.assertEqual(layer.padding, (1, 1))
            self.assertEqual(layer.stride, (1, 1))
        counts = {mode: sum(p.numel() for p in ZeroDCE(mode).parameters()) for mode in MODES}
        self.assertEqual(set(counts.values()), {OFFICIAL_PARAMETERS}, counts)

    def test_skip_connections(self):
        """conv5 ← cat(x3, x4), conv6 ← cat(x2, x5), conv7 ← cat(x1, x6)."""
        model = ZeroDCE()
        inputs, outputs = {}, {}
        def record(index):
            def hook(module, args, out):
                inputs[index], outputs[index] = args[0], out
            return hook

        for index, layer in enumerate(model.layers):
            layer.register_forward_hook(record(index))
        x = torch.rand(2, 3, 20, 28)
        _, curves, _ = model(x)
        act = {i: F.relu(outputs[i]) for i in range(6)}
        self.assertTrue(torch.equal(inputs[0], x))
        for i in (1, 2, 3):
            self.assertTrue(torch.equal(inputs[i], act[i - 1]))
        self.assertTrue(torch.equal(inputs[4], torch.cat((act[2], act[3]), 1)))
        self.assertTrue(torch.equal(inputs[5], torch.cat((act[1], act[4]), 1)))
        self.assertTrue(torch.equal(inputs[6], torch.cat((act[0], act[5]), 1)))
        self.assertTrue(torch.equal(curves, torch.tanh(outputs[6])))

    def test_initialization(self):
        """weight ~ N(0, 0.02), bias = 0 (계획서 규격)."""
        model = ZeroDCE()
        for index, layer in enumerate(model.layers):
            weight = layer.weight.detach()
            self.assertLess(abs(float(weight.mean())), 0.003, index)
            self.assertAlmostEqual(float(weight.std()), 0.02, delta=0.003, msg=index)
            self.assertTrue(bool((layer.bias == 0).all()), index)

    def test_output_shapes_and_curve_range(self):
        for mode in MODES:
            for size in ((20, 28), (17, 23)):
                x = torch.rand(2, 3, *size)
                enhanced, curves, maps = ZeroDCE(mode)(x)
                self.assertEqual(tuple(enhanced.shape), (2, 3, *size))
                self.assertEqual(tuple(curves.shape), (2, 24, *size))
                self.assertTrue(bool((curves.abs() <= 1).all()))
                self.assertTrue(bool(torch.isfinite(enhanced).all()))
            self.assertEqual(maps == {}, mode == 'none')

    def test_eight_iteration_curve(self):
        """none 모드 출력 = 3채널씩 8회 I + r·(I² − I), 앞 채널부터 적용."""
        model = ZeroDCE('none')
        x = torch.rand(2, 3, 20, 20)
        enhanced, curves, _ = model(x)
        expected = x
        for i in range(8):
            r = curves[:, 3 * i:3 * i + 3]
            expected = expected + r * (expected ** 2 - expected)
        torch.testing.assert_close(enhanced, expected)

    def test_curve_sign_convention(self):
        """r = +1이면 어두워지고(I²), r = −1이면 밝아진다. 제안서 α = −r."""
        x = torch.full((1, 3, 16, 16), 0.3)
        darker, curves, _ = saturated_model('none', 10.0)(x)
        self.assertGreater(float(curves.min()), 0.999)
        self.assertLess(float(darker.max()), 1e-6)            # 0.3^(2^8) ≈ 0
        brighter, curves, _ = saturated_model('none', -10.0)(x)
        self.assertLess(float(curves.max()), -0.999)
        self.assertGreater(float(brighter.min()), 1 - 1e-6)   # 1 − 0.7^(2^8) ≈ 1
        _, curves, _ = saturated_model('none', 0.0)(x)
        self.assertTrue(bool((curves == 0).all()))
        torch.testing.assert_close(saturated_model('none', 0.0)(x)[0], x)  # r = 0이면 항등

    def test_boundary_inputs(self):
        zeros, ones = torch.zeros(1, 3, 16, 16), torch.ones(1, 3, 16, 16)
        for value in (-10.0, 0.0, 10.0):
            model = saturated_model('none', value)
            self.assertTrue(torch.equal(model(zeros)[0], zeros))   # 0은 곡선의 고정점
            self.assertTrue(torch.equal(model(ones)[0], ones))     # 1도 고정점
        for mode in MODES:
            for value in (-10.0, 10.0, None):
                model = ZeroDCE(mode) if value is None else saturated_model(mode, value)
                for x in (zeros, ones, torch.rand(2, 3, 24, 24)):
                    enhanced = model(x)[0]
                    self.assertTrue(bool(torch.isfinite(enhanced).all()), (mode, value))
                    self.assertGreaterEqual(float(enhanced.min()), 0, (mode, value))
                    self.assertLessEqual(float(enhanced.max()), 1 + 1e-6, (mode, value))

    def test_gradients_finite_for_all_parameters(self):
        objective = ZeroDCELoss()
        for mode in MODES:
            for x in (torch.rand(2, 3, 24, 24) * 0.4, torch.zeros(1, 3, 16, 16),
                      torch.ones(1, 3, 16, 16)):
                model = ZeroDCE(mode)
                enhanced, curves, _ = model(x)
                loss, _ = objective(x, enhanced, curves)
                self.assertTrue(bool(torch.isfinite(loss)), mode)
                loss.backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, (mode, name))
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()), (mode, name))


class LossTests(unittest.TestCase):
    """TASK-008: 손실 구성과 가중치."""

    def setUp(self):
        torch.manual_seed(0)
        self.objective = ZeroDCELoss()

    def test_matches_official_reference(self):
        for size in ((32, 48), (20, 37), (16, 16)):
            original = torch.rand(2, 3, *size) * 0.3
            enhanced = torch.rand(2, 3, *size)
            curves = torch.rand(2, 24, *size) * 2 - 1
            total, parts = self.objective(original, enhanced, curves)
            ref_total, ref_parts = official_losses(original, enhanced, curves)
            torch.testing.assert_close(total, ref_total)
            for key in ('spatial', 'exposure', 'color', 'tv'):
                torch.testing.assert_close(parts[key], ref_parts[key], msg=f'{size} {key}')

    def test_weights_and_total(self):
        original = torch.rand(2, 3, 32, 32) * 0.3
        total, parts = self.objective(original, torch.rand(2, 3, 32, 32), torch.rand(2, 24, 32, 32))
        expected = parts['spatial'] + 10 * parts['exposure'] + 5 * parts['color'] + 200 * parts['tv']
        torch.testing.assert_close(total, expected)
        self.assertEqual(set(parts), {'spatial', 'exposure', 'color', 'tv'})

    def test_hand_computed_terms(self):
        # 노출 목표 0.6의 회색 상수 영상, 곡선 0 → 네 항 모두 0
        gray = torch.full((1, 3, 16, 16), 0.6)
        total, parts = self.objective(gray, gray, torch.zeros(1, 24, 16, 16))
        for key, value in parts.items():
            self.assertAlmostEqual(float(value), 0.0, places=7, msg=key)
        self.assertAlmostEqual(float(total), 0.0, places=6)
        # 노출: 밝기 0.2 상수 → (0.2 − 0.6)² = 0.16
        dark = torch.full((1, 3, 16, 16), 0.2)
        self.assertAlmostEqual(float(self.objective(dark, dark, torch.zeros(1, 24, 16, 16))[1]['exposure']),
                               0.16, places=6)
        # 색: RGB 평균 (0.5, 0.3, 0.2) → sqrt(0.2⁴ + 0.3⁴ + 0.1⁴)
        tinted = torch.tensor([0.5, 0.3, 0.2])[None, :, None, None].expand(1, 3, 16, 16)
        color = self.objective(tinted, tinted, torch.zeros(1, 24, 16, 16))[1]['color']
        self.assertAlmostEqual(float(color), (0.2 ** 4 + 0.3 ** 4 + 0.1 ** 4) ** 0.5, places=6)
        # TV: 세로 방향으로 0.1씩 증가하는 24채널 → 2 × (24 × 0.01) = 0.48 (채널 합 유지)
        ramp = (torch.arange(16.0) * 0.1)[None, None, :, None].expand(1, 24, 16, 16)
        self.assertAlmostEqual(float(self.objective(gray, gray, ramp)[1]['tv']), 0.48, places=5)
        # 공간: 상수 0.2 → 상수 0.6. 공식 구현의 zero padding 때문에 테두리에서만 차이가 생긴다.
        # 4×4 풀링 격자 테두리 방향 16개 × 0.4² / 16칸 = 0.16
        spatial = self.objective(dark, gray, torch.zeros(1, 24, 16, 16))[1]['spatial']
        self.assertAlmostEqual(float(spatial), 0.16, places=6)

    def test_minimum_size(self):
        self.objective(torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16), torch.rand(1, 24, 16, 16))
        for size in ((15, 16), (16, 15), (8, 8)):
            x = torch.rand(1, 3, *size)
            with self.assertRaises(ValueError):
                self.objective(x, x, torch.rand(1, 24, *size))

    def test_backward_finite(self):
        for original, enhanced in ((torch.rand(2, 3, 24, 24), torch.rand(2, 3, 24, 24)),
                                   (torch.full((1, 3, 16, 16), 0.4), torch.full((1, 3, 16, 16), 0.4))):
            enhanced = enhanced.clone().requires_grad_()
            curves = torch.rand(original.shape[0], 24, *original.shape[-2:], requires_grad=True)
            total, _ = self.objective(original, enhanced, curves)
            total.backward()
            self.assertTrue(bool(torch.isfinite(enhanced.grad).all()))
            self.assertTrue(bool(torch.isfinite(curves.grad).all()))

    def test_reference_image_not_used(self):
        """손실은 low·출력·곡선만 받으며 high(정답)를 인자로 받지 않는다."""
        self.assertEqual(list(inspect.signature(ZeroDCELoss.forward).parameters),
                         ['self', 'original', 'enhanced', 'curves'])


if __name__ == '__main__':
    unittest.main()
