"""LOL 저조도 이미지 데이터셋 로딩과 재현 가능한 분할을 제공합니다."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImagePair:
    """한 저조도 이미지와 선택적인 정상조도 정답 이미지의 경로입니다."""

    low: Path
    high: Optional[Path] = None


def _image_files(directory: str | Path) -> list[Path]:
    """폴더 안의 이미지 경로를 재귀적으로 찾고 입력 오류를 알립니다."""
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(
            f"이미지 폴더를 찾을 수 없습니다: {root}\n"
            "데이터셋 경로를 확인하고 low/high 폴더를 각각 지정해 주세요."
        )

    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(
            f"이미지 파일이 없습니다: {root}\n"
            f"지원 확장자: {', '.join(sorted(IMAGE_SUFFIXES))}"
        )
    return paths


def _relative_key(path: Path, root: Path) -> str:
    """확장자가 달라도 low/high 파일을 상대 경로와 파일명으로 매칭합니다."""
    return path.relative_to(root).with_suffix("").as_posix().casefold()


def discover_image_pairs(
    low_dir: str | Path,
    high_dir: Optional[str | Path] = None,
) -> list[ImagePair]:
    """저조도 폴더를 탐색하고, 지정된 경우 정답 폴더와 파일을 짝짓습니다.

    정답 폴더는 평가를 위해 선택적으로 지정합니다. 짝 데이터가 없어도
    저조도 입력만으로 비지도 학습용 데이터 목록을 만들 수 있습니다.
    """
    low_root = Path(low_dir).expanduser()
    low_paths = _image_files(low_root)
    if high_dir is None:
        return [ImagePair(low=path) for path in low_paths]

    high_root = Path(high_dir).expanduser()
    high_paths = _image_files(high_root)

    def make_index(paths: Sequence[Path], root: Path, label: str) -> dict[str, Path]:
        index: dict[str, Path] = {}
        for path in paths:
            key = _relative_key(path, root)
            if key in index:
                raise ValueError(
                    f"{label} 폴더에 같은 이름의 이미지가 여러 개 있습니다: {key}"
                )
            index[key] = path
        return index

    low_index = make_index(low_paths, low_root, "저조도")
    high_index = make_index(high_paths, high_root, "정상조도")
    missing = sorted(low_index.keys() - high_index.keys())
    extra = sorted(high_index.keys() - low_index.keys())
    if missing or extra:
        details = []
        if missing:
            details.append(f"정상조도 대응 파일이 없는 저조도 이미지: {missing[:5]}")
        if extra:
            details.append(f"저조도 대응 파일이 없는 정상조도 이미지: {extra[:5]}")
        raise ValueError("저조도/정상조도 파일 쌍이 맞지 않습니다. " + "; ".join(details))

    return [
        ImagePair(low=low_index[key], high=high_index[key])
        for key in sorted(low_index)
    ]


def split_pairs(
    pairs: Sequence[ImagePair],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> tuple[list[ImagePair], list[ImagePair], list[ImagePair]]:
    """고정 시드로 데이터 쌍을 학습·검증·평가 세트로 나눕니다."""
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios):
        raise ValueError("ratios는 음수가 아닌 (학습, 검증, 평가) 비율 세 개여야 합니다.")
    if not np.isclose(sum(ratios), 1.0, atol=1e-6):
        raise ValueError(f"분할 비율의 합은 1이어야 합니다. 현재 값: {ratios}")
    if not pairs:
        raise ValueError("분할할 이미지가 없습니다.")

    active_splits = sum(ratio > 0 for ratio in ratios)
    if len(pairs) < active_splits:
        raise ValueError(
            f"이미지 {len(pairs)}장으로 비어 있지 않은 {active_splits}개 분할을 "
            "만들 수 없습니다. 분할 수나 비율을 조정해 주세요."
        )

    exact_counts = [len(pairs) * ratio for ratio in ratios]
    counts = [int(value) for value in exact_counts]
    remainder = len(pairs) - sum(counts)
    order = sorted(
        range(3), key=lambda index: exact_counts[index] - counts[index], reverse=True
    )
    for index in order[:remainder]:
        counts[index] += 1

    # 양수 비율인 분할이 반올림 때문에 비지 않도록 한 장씩 보장합니다.
    for index, ratio in enumerate(ratios):
        if ratio > 0 and counts[index] == 0:
            donor = max(range(3), key=lambda candidate: counts[candidate])
            if counts[donor] <= 1:
                raise ValueError("데이터가 너무 적어 요청한 분할을 만들 수 없습니다.")
            counts[donor] -= 1
            counts[index] += 1

    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)
    train_end = counts[0]
    val_end = train_end + counts[1]
    return (
        [pairs[index] for index in indices[:train_end]],
        [pairs[index] for index in indices[train_end:val_end]],
        [pairs[index] for index in indices[val_end:]],
    )


class LowLightDataset(Dataset):
    """RGB 이미지를 0~1 텐서로 읽고 학습 시 동일한 공간 증강을 적용합니다."""

    def __init__(
        self,
        pairs: Sequence[ImagePair],
        crop_size: Optional[int] = None,
        augment: bool = False,
        return_reference: bool = False,
    ) -> None:
        if not pairs:
            raise ValueError("데이터셋이 비어 있습니다.")
        if crop_size is not None and crop_size <= 0:
            raise ValueError("crop_size는 양의 정수여야 합니다.")
        if return_reference and any(pair.high is None for pair in pairs):
            raise ValueError(
                "정상조도 정답을 반환하려면 모든 이미지 쌍에 high 경로가 있어야 합니다."
            )
        self.pairs = list(pairs)
        self.crop_size = crop_size
        self.augment = augment
        self.return_reference = return_reference

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _read_rgb(path: Path) -> Image.Image:
        """이미지를 RGB로 변환하고 손상·읽기 오류에 경로를 붙여 전달합니다."""
        try:
            with Image.open(path) as image:
                return image.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise OSError(f"이미지를 읽을 수 없습니다: {path} ({exc})") from exc

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        """RGB 픽셀을 채널 우선 float32 텐서로 변환하고 0~1로 정규화합니다."""
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        low = self._read_rgb(pair.low)
        high = self._read_rgb(pair.high) if self.return_reference and pair.high else None

        if high is not None and low.size != high.size:
            raise ValueError(
                "저조도 이미지와 정답 이미지의 크기가 다릅니다: "
                f"{pair.low}={low.size}, {pair.high}={high.size}"
            )

        if self.crop_size is not None:
            width, height = low.size
            if width < self.crop_size or height < self.crop_size:
                raise ValueError(
                    f"이미지 {pair.low} 크기 {width}x{height}가 "
                    f"crop_size={self.crop_size}보다 작습니다."
                )
            left = random.randint(0, width - self.crop_size)
            top = random.randint(0, height - self.crop_size)
            box = (left, top, left + self.crop_size, top + self.crop_size)
            low = low.crop(box)
            if high is not None:
                high = high.crop(box)

        if self.augment and random.random() < 0.5:
            low = low.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if high is not None:
                high = high.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        low_tensor = self._to_tensor(low)
        if not self.return_reference:
            return low_tensor
        if high is None:
            raise RuntimeError(f"정상조도 정답 이미지가 없습니다: {pair.low}")
        return low_tensor, self._to_tensor(high)


def _seed_worker(worker_id: int) -> None:
    """DataLoader 작업자마다 Python/NumPy 난수 생성기를 재현 가능하게 초기화합니다."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_lol_datasets(
    low_dir: str | Path,
    high_dir: Optional[str | Path] = None,
    test_low_dir: Optional[str | Path] = None,
    test_high_dir: Optional[str | Path] = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
    crop_size: Optional[int] = None,
) -> dict[str, LowLightDataset]:
    """LOL 폴더 경로를 받아 학습·검증·평가 데이터셋을 구성합니다.

    별도 test 폴더가 있으면 그 폴더는 분할에서 제외합니다. test 폴더가
    없으면 지정된 low/high 폴더를 고정 시드로 세 부분으로 나눕니다.
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio는 0과 1 사이여야 합니다.")
    if test_low_dir is None and (
        not 0 < test_ratio < 1 or val_ratio + test_ratio >= 1
    ):
        raise ValueError(
            "별도 test 폴더가 없으면 val_ratio와 test_ratio는 양수이고 "
            "두 비율의 합은 1보다 작아야 합니다."
        )
    if test_low_dir is None and test_high_dir is not None:
        raise ValueError("test_high_dir를 사용하려면 test_low_dir도 지정해야 합니다.")

    pairs = discover_image_pairs(low_dir, high_dir)
    if test_low_dir is None:
        train_pairs, val_pairs, test_pairs = split_pairs(
            pairs, (1.0 - val_ratio - test_ratio, val_ratio, test_ratio), seed
        )
    else:
        test_pairs = discover_image_pairs(test_low_dir, test_high_dir)
        train_pairs, val_pairs, _ = split_pairs(
            pairs, (1.0 - val_ratio, val_ratio, 0.0), seed
        )
        train_low_paths = {pair.low.resolve() for pair in train_pairs + val_pairs}
        test_low_paths = {pair.low.resolve() for pair in test_pairs}
        overlap = train_low_paths & test_low_paths
        if overlap:
            example = sorted(str(path) for path in overlap)[0]
            raise ValueError(f"학습/검증과 평가 폴더가 겹칩니다: {example}")

    return {
        "train": LowLightDataset(
            train_pairs, crop_size=crop_size, augment=True, return_reference=False
        ),
        "val": LowLightDataset(
            val_pairs, return_reference=all(pair.high is not None for pair in val_pairs)
        ),
        "test": LowLightDataset(
            test_pairs, return_reference=all(pair.high is not None for pair in test_pairs)
        ),
    }


def create_lol_dataloaders(
    low_dir: str | Path,
    high_dir: Optional[str | Path] = None,
    test_low_dir: Optional[str | Path] = None,
    test_high_dir: Optional[str | Path] = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
    crop_size: Optional[int] = None,
    batch_size: int = 8,
    num_workers: int = 2,
    pin_memory: Optional[bool] = None,
) -> dict[str, DataLoader]:
    """데이터셋을 만든 뒤 재현 가능한 학습·검증·평가 DataLoader를 반환합니다."""
    if batch_size <= 0:
        raise ValueError("batch_size는 양의 정수여야 합니다.")
    if num_workers < 0:
        raise ValueError("num_workers는 음수가 될 수 없습니다.")

    datasets = create_lol_datasets(
        low_dir=low_dir,
        high_dir=high_dir,
        test_low_dir=test_low_dir,
        test_high_dir=test_high_dir,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        crop_size=crop_size,
    )
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loaders: dict[str, DataLoader] = {}
    for offset, split in enumerate(("train", "val", "test")):
        generator = torch.Generator()
        generator.manual_seed(seed + offset)
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=_seed_worker,
            generator=generator,
            persistent_workers=(num_workers > 0),
        )
    return loaders
