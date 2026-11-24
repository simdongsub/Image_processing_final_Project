"""LOL 저조도 이미지 데이터셋 로딩과 재현 가능한 분할을 제공합니다."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
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


def _split_train_val_without_content_overlap(
    pairs: Sequence[ImagePair], val_ratio: float, seed: int
) -> tuple[list[ImagePair], list[ImagePair]]:
    """같은 low 또는 high 바이트를 가진 쌍을 한 분할에 묶습니다."""
    base_train, base_val, _ = split_pairs(
        pairs, (1.0 - val_ratio, val_ratio, 0.0), seed
    )
    parent = list(range(len(pairs)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen: dict[tuple[str, str], int] = {}
    for index, pair in enumerate(pairs):
        for role in ("low", "high"):
            path = getattr(pair, role)
            if path is None:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            key = (role, digest)
            if key in seen:
                parent[find(index)] = find(seen[key])
            else:
                seen[key] = index

    groups: dict[int, list[int]] = {}
    for index in range(len(pairs)):
        groups.setdefault(find(index), []).append(index)
    if len(groups) == len(pairs):
        return base_train, base_val

    shuffled_groups = list(groups.values())
    random.Random(seed).shuffle(shuffled_groups)
    val_indices: set[int] = set()
    for group in shuffled_groups:
        if len(val_indices) + len(group) <= len(base_val):
            val_indices.update(group)
    if not val_indices or len(val_indices) == len(pairs):
        raise ValueError("중복 내용을 분리하지 않고 train/val을 만들 수 없습니다.")
    train = [pair for index, pair in enumerate(pairs) if index not in val_indices]
    val = [pair for index, pair in enumerate(pairs) if index in val_indices]
    return train, val


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
        train_pairs, val_pairs = _split_train_val_without_content_overlap(
            pairs, val_ratio, seed
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
    split_seed: int = 0,
) -> dict[str, DataLoader]:
    """split_seed로 분할을 고정하고 seed로 학습 셔플 RNG를 설정합니다.

    평가·검증은 원본 크기 이미지 지원을 위해 배치 크기 1을 사용합니다.
    num_workers=0인 증강 RNG는 호출자가 random.seed로 초기화해야 합니다.
    """
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
        seed=split_seed,
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
            batch_size=batch_size if split == "train" else 1,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=_seed_worker,
            generator=generator,
            persistent_workers=(num_workers > 0),
        )
    return loaders


def audit_lol_splits(
    train_low: str | Path,
    train_high: str | Path,
    test_low: str | Path,
    test_high: str | Path,
    crop_size: int = 128,
    val_ratio: float = 0.1,
    split_seed: int = 0,
) -> dict:
    """실제 파일과 고정 분할의 쌍·손상·중복을 점검합니다."""
    if crop_size < 1:
        raise ValueError("crop_size는 양의 정수여야 합니다.")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio는 0과 1 사이여야 합니다.")

    report: dict = {"splits": {}, "partition": None, "issues": []}
    paths: dict[str, dict[str, set[Path]]] = {}
    digest_by_path: dict[Path, str] = {}
    formats = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG",
               ".bmp": "BMP", ".tif": "TIFF", ".tiff": "TIFF"}

    for split, roots in (
        ("train", {"low": Path(train_low).expanduser(), "high": Path(train_high).expanduser()}),
        ("test", {"low": Path(test_low).expanduser(), "high": Path(test_high).expanduser()}),
    ):
        indexes: dict[str, dict[str, Path]] = {}
        image_info: dict[Path, tuple[tuple[int, int], bool]] = {}
        paths[split] = {"low": set(), "high": set()}
        summary = {"low": None, "high": None, "matched": 0, "valid_pairs": 0,
                   "missing_high": 0, "missing_low": 0, "duplicate_names": 0,
                   "unsupported_files": 0, "size_mismatches": 0, "corrupt_files": 0,
                   "format_mismatches": 0, "too_small_for_crop": 0,
                   "low_resolutions": {}, "high_resolutions": {}}
        report["splits"][split] = summary

        for role, root in roots.items():
            if not root.is_dir():
                report["issues"].append(f"{split}/{role}: 폴더 없음 또는 접근 불가: {root}")
                indexes[role] = {}
                continue
            try:
                files = [path for path in root.rglob("*") if path.is_file()]
            except OSError as exc:
                report["issues"].append(f"{split}/{role}: 목록 읽기 실패: {exc}")
                indexes[role] = {}
                continue
            images = [path for path in files if path.suffix.lower() in IMAGE_SUFFIXES]
            unsupported = [path for path in files if not path.name.startswith(".")
                           and path.suffix.lower() not in IMAGE_SUFFIXES]
            summary[role] = len(images)
            summary["unsupported_files"] += len(unsupported)
            for path in unsupported:
                report["issues"].append(f"{split}/{role}: 지원하지 않는 파일: {path}")
            counts = Counter(_relative_key(path, root) for path in images)
            duplicates = [key for key, count in counts.items() if count > 1]
            summary["duplicate_names"] += len(duplicates)
            for key in duplicates:
                report["issues"].append(f"{split}/{role}: 중복 이름: {key}")
            indexes[role] = {_relative_key(path, root): path for path in images}
            paths[split][role] = {path.resolve() for path in images}
            if not images:
                report["issues"].append(f"{split}/{role}: 이미지 파일 없음: {root}")
            for path in images:
                try:
                    with Image.open(path) as image:
                        image.load()
                        size = image.size
                        valid_format = image.format == formats[path.suffix.lower()]
                        if not valid_format:
                            summary["format_mismatches"] += 1
                            report["issues"].append(
                                f"{split}/{role}: 확장자와 실제 형식 불일치: {path} ({image.format})")
                    image_info[path] = (size, valid_format)
                    resolution = f"{size[0]}x{size[1]}"
                    resolutions = summary[f"{role}_resolutions"]
                    resolutions[resolution] = resolutions.get(resolution, 0) + 1
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    digest_by_path[path.resolve()] = digest
                except (OSError, UnidentifiedImageError) as exc:
                    summary["corrupt_files"] += 1
                    report["issues"].append(f"{split}/{role}: 읽기 또는 디코딩 실패: {path} ({exc})")

        if summary["low"] is None or summary["high"] is None:
            for field in ("matched", "valid_pairs", "missing_high", "missing_low",
                          "duplicate_names", "unsupported_files", "size_mismatches",
                          "corrupt_files", "format_mismatches", "too_small_for_crop",
                          "low_resolutions", "high_resolutions"):
                summary[field] = None
            continue

        low_keys, high_keys = set(indexes["low"]), set(indexes["high"])
        missing_high, missing_low = low_keys - high_keys, high_keys - low_keys
        summary["missing_high"], summary["missing_low"] = len(missing_high), len(missing_low)
        for key in sorted(missing_high):
            report["issues"].append(f"{split}: high 누락: {key}")
        for key in sorted(missing_low):
            report["issues"].append(f"{split}: low 누락: {key}")
        summary["matched"] = len(low_keys & high_keys)

        for key in sorted(low_keys & high_keys):
            low_info = image_info.get(indexes["low"][key])
            high_info = image_info.get(indexes["high"][key])
            if low_info is None or high_info is None or not low_info[1] or not high_info[1]:
                continue
            if low_info[0] != high_info[0]:
                summary["size_mismatches"] += 1
                report["issues"].append(f"{split}: 크기 불일치: {key} low={low_info[0]}, high={high_info[0]}")
                continue
            if split == "train" and min(low_info[0]) < crop_size:
                summary["too_small_for_crop"] += 1
                report["issues"].append(f"train: crop_size={crop_size}보다 작은 이미지: {key} {low_info[0]}")
                continue
            summary["valid_pairs"] += 1

    for split in ("train", "test"):
        repeated_within_pair = paths[split]["low"] & paths[split]["high"]
        if repeated_within_pair:
            report["issues"].append(f"{split}: low/high가 같은 파일을 참조: {len(repeated_within_pair)}개")

    if not report["issues"]:
        try:
            datasets = create_lol_datasets(train_low, train_high, test_low, test_high,
                                           val_ratio=val_ratio, seed=split_seed,
                                           crop_size=crop_size)
        except (ValueError, FileNotFoundError) as exc:
            report["issues"].append(f"분할 생성 실패: {exc}")
        else:
            partition = {"split_seed": split_seed, "val_ratio": val_ratio,
                         "counts": {name: len(dataset) for name, dataset in datasets.items()},
                         "overlap": {}}
            report["partition"] = partition
            for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
                details = {}
                for role in ("low", "high"):
                    left_paths = {getattr(pair, role).resolve() for pair in datasets[left].pairs}
                    right_paths = {getattr(pair, role).resolve() for pair in datasets[right].pairs}
                    details[f"{role}_same_path"] = len(left_paths & right_paths)
                    left_hashes = {digest_by_path[path] for path in left_paths}
                    details[f"{role}_same_content"] = sum(
                        digest_by_path[path] in left_hashes for path in right_paths)
                partition["overlap"][f"{left}_{right}"] = details
                if any(details.values()):
                    report["issues"].append(f"{left}/{right} 파일 중복: {details}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LOL 학습/평가 이미지 무결성 점검")
    parser.add_argument("--train-low", required=True)
    parser.add_argument("--train-high", required=True)
    parser.add_argument("--test-low", required=True)
    parser.add_argument("--test-high", required=True)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()
    result = audit_lol_splits(args.train_low, args.train_high, args.test_low,
                              args.test_high, args.crop_size, args.val_ratio,
                              args.split_seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(bool(result["issues"]))
