import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import DATASET_DIR, DEVICE, IMAGE_SIZE
from src.scan_qr import DEFAULT_CHECKPOINT, EXPECTED_CLASSES, load_classifier
from src.train import calculate_metrics


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
COMPARISON_DIRS = (
    DATASET_DIR / "train",
    DATASET_DIR / "val",
    DATASET_DIR / "test",
    DATASET_DIR / "benign_qr_images_500",
    DATASET_DIR / "malicious_qr_images_500",
)


def validate_external_root(dataset_root: Path) -> Path:
    """Require an external class-folder dataset outside the project dataset."""
    root = dataset_root.expanduser().resolve()
    project_dataset = DATASET_DIR.resolve()
    if root == project_dataset or project_dataset in root.parents or root in project_dataset.parents:
        raise ValueError(
            "External evaluation data must be outside the project's dataset directory."
        )
    if not root.is_dir():
        raise FileNotFoundError(f"External dataset folder not found: {root}")

    class_directories = {
        entry.name for entry in root.iterdir() if entry.is_dir()
    }
    if class_directories != EXPECTED_CLASSES:
        raise ValueError(
            f"External dataset must contain exactly {sorted(EXPECTED_CLASSES)} "
            f"class folders; found {sorted(class_directories)}."
        )
    return root


def image_hashes(dataset_root: Path) -> dict[str, Path]:
    """Hash supported image files so exact duplicates can be detected."""
    hashes: dict[str, Path] = {}
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:
            raise ValueError(
                f"Exact duplicate images found: {hashes[digest]} and {path}"
            )
        hashes[digest] = path
    return hashes


def check_no_training_overlap(
    external_hashes: dict[str, Path],
    comparison_dirs: tuple[Path, ...] = COMPARISON_DIRS,
) -> None:
    """Reject byte-identical images found in the existing project dataset."""
    for directory in comparison_dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            external_path = external_hashes.get(digest)
            if external_path is not None:
                raise ValueError(
                    "Exact image overlap with existing project data detected: "
                    f"{external_path} matches {path}"
                )


def evaluate_external_dataset(
    dataset_root: Path,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
) -> dict[str, Any]:
    """Evaluate the saved checkpoint on labeled external images without training."""
    root = validate_external_root(dataset_root)
    external_hashes = image_hashes(root)
    if not external_hashes:
        raise ValueError(f"No supported images found in external dataset: {root}")
    check_no_training_overlap(external_hashes)

    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.ImageFolder(root, transform=transform)
    if not dataset.samples:
        raise ValueError(f"No supported images found in external dataset: {root}")
    if dataset.class_to_idx.keys() != EXPECTED_CLASSES:
        raise ValueError(
            f"External class mapping must include {sorted(EXPECTED_CLASSES)}, "
            f"found {dataset.class_to_idx}."
        )

    model, class_to_idx = load_classifier(checkpoint_path)
    if dataset.class_to_idx != class_to_idx:
        raise ValueError(
            "External dataset class mapping does not match the checkpoint: "
            f"{dataset.class_to_idx} != {class_to_idx}"
        )

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    confusion = torch.zeros((len(class_to_idx), len(class_to_idx)), dtype=torch.int64)
    loss_function = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    predictions: list[dict[str, Any]] = []
    sample_offset = 0
    index_to_class = {index: name for name, index in class_to_idx.items()}

    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(DEVICE))
            labels_on_device = labels.to(DEVICE)
            total_loss += float(loss_function(logits, labels_on_device).item())
            probabilities = torch.softmax(logits, dim=1).cpu()
            predicted_labels = probabilities.argmax(dim=1)

            for batch_index, (true_index, predicted_index) in enumerate(
                zip(labels.tolist(), predicted_labels.tolist())
            ):
                image_path, _ = dataset.samples[sample_offset + batch_index]
                confidence = float(probabilities[batch_index, predicted_index])
                confusion[true_index, predicted_index] += 1
                predictions.append(
                    {
                        "image": Path(image_path).relative_to(root).as_posix(),
                        "true_label": index_to_class[true_index],
                        "predicted_label": index_to_class[predicted_index],
                        "correct": true_index == predicted_index,
                        "predicted_score": confidence,
                    }
                )
            sample_offset += len(labels)

    metrics = calculate_metrics(confusion, class_to_idx)
    return {
        "dataset": str(root),
        "checkpoint": str(checkpoint_path.resolve()),
        "sample_count": len(dataset),
        "class_counts": {
            class_name: sum(label == class_to_idx[class_name] for _, label in dataset.samples)
            for class_name in class_to_idx
        },
        "loss": total_loss / len(dataset),
        "accuracy": metrics["accuracy"],
        "per_class": metrics["per_class"],
        "confusion_matrix": {
            "class_index": class_to_idx,
            "rows_true_columns_predicted": confusion.tolist(),
        },
        "independence_checks": {
            "dataset_path_outside_project_dataset": True,
            "no_exact_hash_overlap_with_project_images": True,
            "external_provenance_or_label_correctness_verified": False,
        },
        "limitations": [
            "Exact file hashes detect only byte-identical image overlap, not resized or altered copies.",
            "Dataset source and labels are supplied by the user and are not independently verified.",
            "Predicted scores are uncalibrated and are not safety probabilities.",
            "This evaluation does not establish real-world security performance.",
        ],
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained QR classifier on a separate labeled dataset."
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="External dataset root containing exactly fake/ and real/ folders",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Model checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the JSON metrics and per-image predictions",
    )
    args = parser.parse_args()

    report = evaluate_external_dataset(args.dataset, args.checkpoint)
    report_json = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_json + "\n", encoding="utf-8")
        print(f"External evaluation report saved to: {args.output}")
    print(report_json)


if __name__ == "__main__":
    main()
