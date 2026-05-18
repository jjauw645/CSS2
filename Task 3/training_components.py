# Reusable training components for the CSS2 gingivitis segmentation notebooks
# These are designed to replace fragile notebook cells while keeping the current
# PyTorch + segmentation-models-pytorch workflow

from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import cv2
except ModuleNotFoundError:  # Allows local audits when OpenCV is only installed in Colab.
    cv2 = None


IGNORE_INDEX = 255

CLASS_MAPPINGS = {
    # Current project setup: healthy/mild, moderate, severe/very severe.
    "severity_3": {0: 0, 1: 0, 2: 1, 3: 2, 4: 2},
    # Closer to the paper-style grouping mentioned in team chat:
    # healthy, questionable, diseased.
    "paper_3": {0: 0, 1: 1, 2: 1, 3: 2, 4: 2},
    # Useful sanity baseline. If this fails, the masks are the main bottleneck.
    "binary_disease": {0: 0, 1: 1, 2: 1, 3: 1, 4: 1},
    # Original five severity classes.
    "severity_5": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
}

CLASS_NAMES = {
    "severity_3": ["healthy_or_mild", "moderate", "severe_or_very_severe"],
    "paper_3": ["healthy", "questionable", "diseased"],
    "binary_disease": ["healthy", "diseased"],
    "severity_5": ["healthy", "mild", "moderate", "severe", "very_severe"],
}


def load_confidence_scores(path: str | Path, *, required: bool = False) -> dict[str, float]:
    path = Path(path)
    if not path.exists():
        message = f"Confidence score file does not exist: {path}"
        if required:
            raise FileNotFoundError(message)
        warnings.warn(message)
        return {}

    data = json.loads(path.read_text())
    scores = {str(k): float(v) for k, v in data.items() if isinstance(v, (int, float))}

    if required and not scores:
        raise ValueError(
            f"Confidence score file is empty or invalid: {path}. "
            "Curriculum learning cannot run without training scores."
        )
    if not scores:
        warnings.warn(f"No usable confidence scores found in {path}")
    return scores


def remap_index_mask(mask: np.ndarray, mode: str = "severity_3") -> np.ndarray:
    if mode not in CLASS_MAPPINGS:
        raise ValueError(f"Unknown class mapping: {mode}. Options: {sorted(CLASS_MAPPINGS)}")

    remap = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for src, dst in CLASS_MAPPINGS[mode].items():
        remap[src] = dst
    return remap[mask]


def read_image_and_mask(image_path: Path, mask_path: Path, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if cv2 is not None:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
    else:
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
    return image, remap_index_mask(mask, mode)


def compute_pixel_counts(mask_folder: str | Path, mode: str = "severity_3") -> Counter[int]:
    counts: Counter[int] = Counter()
    for path in sorted(Path(mask_folder).glob("*.png")):
        if cv2 is not None:
            raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                continue
        else:
            raw = np.array(Image.open(path).convert("L"))
        mask = remap_index_mask(raw, mode)
        values, value_counts = np.unique(mask[mask != IGNORE_INDEX], return_counts=True)
        counts.update(dict(zip(values.tolist(), value_counts.tolist())))
    return counts


def compute_class_weights(
    mask_folder: str | Path,
    mode: str = "severity_3",
    *,
    max_weight: float = 6.0,
    power: float = 0.5,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    # Return smoothed inverse-frequency weights
    # The square-root smoothing avoids the extreme weights that raw inverse
    # frequency produces for tiny healthy regions

    counts = compute_pixel_counts(mask_folder, mode)
    num_classes = len(CLASS_NAMES[mode])
    freqs = np.array([counts.get(i, 0) for i in range(num_classes)], dtype=np.float64)
    if np.any(freqs == 0):
        missing = [i for i, v in enumerate(freqs) if v == 0]
        raise ValueError(f"No pixels found for classes {missing}; cannot compute weights.")

    inv = (freqs.sum() / freqs) ** power
    inv = inv / inv.mean()
    inv = np.clip(inv, 1.0 / max_weight, max_weight)
    return torch.tensor(inv, dtype=torch.float32, device=device)


def _pad_to_crop(image: np.ndarray, mask: np.ndarray, crop_size: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    pad_h = max(0, crop_size - h)
    pad_w = max(0, crop_size - w)
    if pad_h == 0 and pad_w == 0:
        return image, mask

    if cv2 is not None:
        image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
        )
        mask = cv2.copyMakeBorder(
            mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=IGNORE_INDEX
        )
    else:
        image = np.pad(
            image,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="reflect",
        )
        mask = np.pad(
            mask,
            ((0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=IGNORE_INDEX,
        )
    return image, mask


def class_balanced_crop(
    image: np.ndarray,
    mask: np.ndarray,
    crop_size: int,
    class_weights: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    image, mask = _pad_to_crop(image, mask, crop_size)
    h, w = mask.shape

    present = [int(c) for c in np.unique(mask) if c != IGNORE_INDEX]
    if not present:
        y = np.random.randint(0, h - crop_size + 1)
        x = np.random.randint(0, w - crop_size + 1)
        return image[y : y + crop_size, x : x + crop_size], mask[y : y + crop_size, x : x + crop_size]

    weights = np.array([class_weights.get(c, 1.0) if class_weights else 1.0 for c in present])
    weights = weights / weights.sum()
    target_cls = int(np.random.choice(present, p=weights))
    pixels = np.argwhere(mask == target_cls)
    cy, cx = pixels[np.random.randint(len(pixels))]

    y = int(np.clip(cy - crop_size // 2, 0, h - crop_size))
    x = int(np.clip(cx - crop_size // 2, 0, w - crop_size))
    return image[y : y + crop_size, x : x + crop_size], mask[y : y + crop_size, x : x + crop_size]


class GingivitisSegmentationDataset(Dataset):
    def __init__(
        self,
        image_folder: str | Path,
        mask_folder: str | Path,
        *,
        transform: Callable | None = None,
        mode: str = "severity_3",
        confidence_scores: dict[str, float] | None = None,
        min_confidence: float = 0.0,
        crop_size: int | None = None,
        balanced_crop: bool = False,
    ) -> None:
        self.image_folder = Path(image_folder)
        self.mask_folder = Path(mask_folder)
        self.transform = transform
        self.mode = mode
        self.crop_size = crop_size
        self.balanced_crop = balanced_crop

        self.image_paths = sorted(self.image_folder.glob("*.jpg"))
        if min_confidence > 0:
            if not confidence_scores:
                raise ValueError(
                    "min_confidence was set but no confidence scores were provided. "
                    "The previous notebook silently skipped curriculum learning here."
                )
            self.image_paths = [
                p for p in self.image_paths
                if confidence_scores.get(p.stem, -1.0) >= min_confidence
            ]

        self.mask_paths = [self.mask_folder / f"{p.stem}.png" for p in self.image_paths]
        missing = [p for p in self.mask_paths if not p.exists()]
        if missing:
            preview = ", ".join(str(p) for p in missing[:5])
            raise FileNotFoundError(f"{len(missing)} masks are missing. Examples: {preview}")

        self.crop_class_weights = None
        if balanced_crop:
            counts = compute_pixel_counts(mask_folder, mode)
            self.crop_class_weights = {
                cls: 1.0 / max(count, 1) for cls, count in counts.items()
            }

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image, mask = read_image_and_mask(image_path, mask_path, self.mode)

        if self.crop_size is not None:
            if self.balanced_crop:
                image, mask = class_balanced_crop(
                    image, mask, self.crop_size, self.crop_class_weights
                )
            else:
                image, mask = _pad_to_crop(image, mask, self.crop_size)
                h, w = mask.shape
                y = np.random.randint(0, h - self.crop_size + 1)
                x = np.random.randint(0, w - self.crop_size + 1)
                image = image[y : y + self.crop_size, x : x + self.crop_size]
                mask = mask[y : y + self.crop_size, x : x + self.crop_size]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask, image_path.name


class SegmentationMeter:
    def __init__(self, num_classes: int, ignore_index: int = IGNORE_INDEX) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = torch.argmax(logits, dim=1).detach().cpu()
        targets = targets.detach().cpu()
        valid = targets != self.ignore_index
        preds = preds[valid].view(-1)
        targets = targets[valid].view(-1)
        for true, pred in zip(targets.tolist(), preds.tolist()):
            if 0 <= true < self.num_classes and 0 <= pred < self.num_classes:
                self.confusion[true, pred] += 1

    def summary(self) -> dict[str, object]:
        cm = self.confusion.float()
        tp = torch.diag(cm)
        pred_sum = cm.sum(dim=0)
        true_sum = cm.sum(dim=1)
        union = pred_sum + true_sum - tp
        dice_den = pred_sum + true_sum

        iou = torch.where(union > 0, tp / union.clamp_min(1), torch.zeros_like(tp))
        dice = torch.where(dice_den > 0, 2 * tp / dice_den.clamp_min(1), torch.zeros_like(tp))
        recall = torch.where(true_sum > 0, tp / true_sum.clamp_min(1), torch.zeros_like(tp))
        support = true_sum.long().tolist()
        total = cm.sum().item()
        accuracy = tp.sum().item() / total if total else 0.0

        return {
            "accuracy": accuracy,
            "macro_iou": iou.mean().item(),
            "macro_dice": dice.mean().item(),
            "macro_recall": recall.mean().item(),
            "per_class_iou": iou.tolist(),
            "per_class_dice": dice.tolist(),
            "per_class_recall": recall.tolist(),
            "support": support,
            "confusion_matrix": self.confusion.tolist(),
        }
