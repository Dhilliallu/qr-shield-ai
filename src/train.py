import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import (
    BATCH_SIZE,
    DATASET_DIR,
    DEVICE,
    IMAGE_SIZE,
    LEARNING_RATE,
    MODEL_DIR,
    NUM_CLASSES,
    NUM_EPOCHS,
)
from src.model import QRClassifier


RANDOM_SEED = 42
EXPECTED_CLASSES = {"fake", "real"}


def build_data_loaders() -> tuple[dict[str, DataLoader], dict[str, int]]:
    """Load prepared image folders and apply consistent image normalization."""
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    datasets_by_split: dict[str, datasets.ImageFolder] = {}
    for split in ("train", "val", "test"):
        split_dir = DATASET_DIR / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Dataset split folder not found: {split_dir}")

        dataset = datasets.ImageFolder(split_dir, transform=transform)
        if not dataset.samples:
            raise ValueError(f"No images found in dataset split: {split_dir}")
        if set(dataset.classes) != EXPECTED_CLASSES:
            raise ValueError(
                f"Expected class folders {sorted(EXPECTED_CLASSES)} in {split_dir}, "
                f"found {dataset.classes}"
            )
        datasets_by_split[split] = dataset

    class_to_idx = datasets_by_split["train"].class_to_idx
    for split, dataset in datasets_by_split.items():
        if dataset.class_to_idx != class_to_idx:
            raise ValueError(f"Class mapping differs in the {split} dataset.")

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=(DEVICE.type == "cuda"),
            generator=generator if split == "train" else None,
        )
        for split, dataset in datasets_by_split.items()
    }
    return loaders, class_to_idx


def run_epoch(
    model: QRClassifier,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, torch.Tensor]:
    """Run a training or evaluation epoch and return loss, accuracy, and counts."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))
        labels = labels.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_function(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

        batch_counts = torch.bincount(
            labels * NUM_CLASSES + predictions,
            minlength=NUM_CLASSES * NUM_CLASSES,
        )
        confusion += batch_counts.reshape(NUM_CLASSES, NUM_CLASSES).cpu()

    if total_samples == 0:
        raise ValueError("Cannot evaluate an empty data loader.")

    return total_loss / total_samples, total_correct / total_samples, confusion


def calculate_metrics(
    confusion: torch.Tensor,
    class_to_idx: dict[str, int],
) -> dict[str, object]:
    """Calculate accuracy and per-class precision, recall, and F1."""
    total = int(confusion.sum().item())
    if total == 0:
        raise ValueError("Cannot calculate metrics for an empty dataset.")

    per_class: dict[str, dict[str, float]] = {}
    for class_name, index in class_to_idx.items():
        true_positive = int(confusion[index, index].item())
        predicted_positive = int(confusion[:, index].sum().item())
        actual_positive = int(confusion[index, :].sum().item())
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    correct = int(confusion.diag().sum().item())
    return {"accuracy": correct / total, "per_class": per_class}


def train_model() -> None:
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    loaders, class_to_idx = build_data_loaders()
    model = QRClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODEL_DIR / "baseline_cnn.pth"

    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = 0

    print("QR Shield AI - Baseline CNN")
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
    print(f"Classes: {class_to_idx}")
    print(f"Batch size: {BATCH_SIZE}; epochs: {NUM_EPOCHS}; learning rate: {LEARNING_RATE}")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_accuracy, _ = run_epoch(
            model, loaders["train"], loss_function, optimizer
        )
        val_loss, val_accuracy, _ = run_epoch(
            model, loaders["val"], loss_function
        )
        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train loss {train_loss:.4f}, accuracy {train_accuracy:.1%} | "
            f"val loss {val_loss:.4f}, accuracy {val_accuracy:.1%}"
        )

        if val_accuracy > best_val_accuracy or (
            val_accuracy == best_val_accuracy and val_loss < best_val_loss
        ):
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "image_size": IMAGE_SIZE,
                    "epoch": epoch,
                    "validation_accuracy": val_accuracy,
                    "validation_loss": val_loss,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy, confusion = run_epoch(
        model, loaders["test"], loss_function
    )
    test_metrics = calculate_metrics(confusion, class_to_idx)

    print(f"Best checkpoint: epoch {best_epoch}, validation accuracy {best_val_accuracy:.1%}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.1%}")
    print("Test metrics by class (precision / recall / F1):")
    for class_name, metrics in test_metrics["per_class"].items():
        print(
            f"  {class_name}: {metrics['precision']:.1%} / "
            f"{metrics['recall']:.1%} / {metrics['f1']:.1%}"
        )
    print("Test confusion matrix (rows=true, columns=predicted; class index order):")
    print(f"  {class_to_idx}")
    print(confusion.tolist())
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    train_model()
