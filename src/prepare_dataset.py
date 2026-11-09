import hashlib
import random
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import IMAGE_SIZE


SOURCE_CLASSES = {
    "benign_qr_images_500": "real",
    "malicious_qr_images_500": "fake",
}
SPLIT_RATIOS = (("train", 0.70), ("val", 0.15), ("test", 0.15))
RANDOM_SEED = 42
SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def split_items(items: list[Path], seed: int = RANDOM_SEED) -> dict[str, list[Path]]:
    """Split one class into deterministic train, validation, and test subsets."""
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)

    train_end = int(len(shuffled) * SPLIT_RATIOS[0][1])
    validation_end = train_end + int(len(shuffled) * SPLIT_RATIOS[1][1])
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }


def collect_source_files(source_dir: Path) -> list[Path]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Required source class folder not found: {source_dir}")

    files = sorted(path for path in source_dir.rglob("*") if path.is_file())
    unsupported = [path for path in files if path.suffix.lower() not in SUPPORTED_EXTENSIONS]
    if unsupported:
        examples = ", ".join(str(path) for path in unsupported[:5])
        raise ValueError(f"Unsupported files found in {source_dir}: {examples}")

    images = [path for path in files if path.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not images:
        raise ValueError(f"No supported images found in {source_dir}")
    return images


def validate_source_images(source_files: dict[str, list[Path]]) -> None:
    seen_hashes: dict[str, Path] = {}
    for label, files in source_files.items():
        for path in files:
            if cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is None:
                raise ValueError(f"Image could not be decoded: {path}")

            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            previous_path = seen_hashes.get(digest)
            if previous_path is not None:
                raise ValueError(
                    "Exact duplicate images can leak across dataset splits: "
                    f"{previous_path} and {path}"
                )
            seen_hashes[digest] = path


def prepare_dataset() -> None:
    dataset_dir = PROJECT_ROOT / "dataset"
    source_files = {
        label: collect_source_files(dataset_dir / source_name)
        for source_name, label in SOURCE_CLASSES.items()
    }
    validate_source_images(source_files)

    output_files: dict[str, dict[str, list[Path]]] = {}
    for label, files in source_files.items():
        output_files[label] = split_items(files)

    output_dirs = [
        dataset_dir / split / label
        for split, _ in SPLIT_RATIOS
        for label in SOURCE_CLASSES.values()
    ]
    for output_dir in output_dirs:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output folder is not empty; refusing to overwrite files: {output_dir}"
            )

    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)

    width, height = IMAGE_SIZE
    print("Dataset preparation plan")
    print(f"Image size: {width} x {height}")
    print(f"Split ratios: train 70%, val 15%, test 15% (seed {RANDOM_SEED})")
    for label, splits in output_files.items():
        print(f"{label}: " + ", ".join(f"{name}={len(items)}" for name, items in splits.items()))

    for label, splits in output_files.items():
        for split, files in splits.items():
            output_dir = dataset_dir / split / label
            for source_path in files:
                image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Image could not be decoded: {source_path}")

                resized = cv2.resize(
                    image,
                    (width, height),
                    interpolation=cv2.INTER_AREA,
                )
                output_path = output_dir / f"{source_path.stem}.png"
                if not cv2.imwrite(str(output_path), resized):
                    raise OSError(f"Could not write processed image: {output_path}")

    total = sum(len(files) for splits in output_files.values() for files in splits.values())
    print(f"Prepared {total} images. Original source images were left unchanged.")


if __name__ == "__main__":
    prepare_dataset()
