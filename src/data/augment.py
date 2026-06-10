"""Create augmented training images from data/raw."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
from tqdm import tqdm

try:
    from config import DATA_AUG_DIR, DATA_RAW_DIR
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import DATA_AUG_DIR, DATA_RAW_DIR


TRANSFORM = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.3, p=0.7),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.Rotate(limit=15, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, p=0.4),
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_limit=(1, 2),
            shadow_dimension=3,
            p=0.2,
        ),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(8, 20),
            hole_width_range=(8, 20),
            fill=0,
            p=0.2,
        ),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.25),
    ]
)


def augment(input_dir: str = DATA_RAW_DIR, output_dir: str = DATA_AUG_DIR, n_augments: int = 8) -> None:
    source_root = Path(input_dir)
    output_root = Path(output_dir)
    if not source_root.exists():
        print(f"[!] Input directory does not exist: {source_root}")
        return

    person_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
    if not person_dirs:
        print(f"[!] No employee folders found in {source_root}")
        return

    total = 0
    for person_dir in person_dirs:
        out_dir = output_root / person_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.jpeg")) + list(person_dir.glob("*.png"))
        count = 0

        for image_path in tqdm(images, desc=f"  {person_dir.name}", leave=False):
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            cv2.imwrite(str(out_dir / image_path.name), image)
            count += 1
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            for idx in range(n_augments):
                augmented = TRANSFORM(image=rgb)["image"]
                out_path = out_dir / f"{image_path.stem}_aug{idx:02d}.jpg"
                cv2.imwrite(str(out_path), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
                count += 1

        print(f"  {person_dir.name}: {len(images)} -> {count} images")
        total += count

    print(f"[OK] Augmentation complete: {total} images")
    print(f"     Saved to: {output_root}")


if __name__ == "__main__":
    raw_n = input("Augmented variants per image [8]: ").strip()
    augment(n_augments=int(raw_n) if raw_n.isdigit() else 8)
